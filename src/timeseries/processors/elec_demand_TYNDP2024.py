import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.timeseries.processors.base_processor import BaseProcessor, SourceDataError


class elec_demand_TYNDP2024(BaseProcessor):
    """
    Electricity demand for each node, from one TYNDP 2024 profile per country.

    The same two steps as district heating, with the profile read from a workbook
    instead of derived from temperature: normalise so the mean climate year sums
    to one, then split each node's annual energy into a profile-driven and a flat
    part.

    Everything file-shaped about the workbook is proved **once**, before the
    parquet cache is written, and recorded in the cache's own metadata. Reading
    the workbook takes over a minute and reading the parquet under a second,
    which is what makes a thorough check affordable. Later runs read the receipt
    and trust it; only the config-shaped questions -- which countries, which
    climate years, which nodes -- are asked every run, where they cost nothing.

    docs/elec-demand-timeseries.md is the documentation and carries the reasoning:
    what `twh/year` means, why the source has no leap years, why an output zero is
    an alarm and a source zero is a gap, and what the cache's receipt covers.

    Parameters
    ----------
    input_folder : str
        Folder holding the TYNDP workbooks, pre-joined by ProcessorRunner.
    country_codes : list of str
        Countries the run includes. The authority on spelling.
    start_year, end_year : int
        First and last climate year.
    df_annual_demands : pd.DataFrame
        Demand rows already filtered to this processor's `demand_grid`.
    demand_grid : str
        The grid written into the output, e.g. `elec`.

    `scenario_year` and `exclude_nodes` also arrive in the kwargs and are
    deliberately not read. `scenario_year` has had nothing to choose since
    `PROFILE_WORKBOOK` became a constant, and still reaches the output through
    each node's `twh/year`, which SourceDataPipeline has whitelisted by it;
    `exclude_nodes` is likewise applied upstream.
    """

    #: Demand is written as negative `ts_influx`. No value_range to go with it:
    #: the meaningful bound is each zone's own peak MW, which no class-level tuple
    #: can state, and a maximum of 0.0 would only restate the sign.
    value_sign = "non_positive"

    #: Divisor for the flat share, deliberately the nominal year rather than the
    #: real one. `DH_demand_fromTemperature` uses the identical formula, so the
    #: two have to move together or not at all.
    HOURS_PER_YEAR = 8760

    #: Hours in one standardised source year. The same number as HOURS_PER_YEAR
    #: and a different fact: this one is a property of the workbook's calendar,
    #: and the checks below compare row counts against it.
    STANDARD_YEAR_HOURS = 8760

    TWH_TO_MWH = 1e6

    #: Zero-indexed row holding the column titles in every country sheet.
    HEADER_ROW = 7

    #: The one workbook every scenario year reads, 2040 included.
    #:
    #: `elec_2040_National_Trends.xlsx` also exists and is deliberately not used:
    #: its SE04 and UK00 sheets carry hours of *negative* demand, which reach
    #: Backbone as free generation into those nodes. Every scenario year takes
    #: the 2030 shape and is scaled to its own `TWh/year` instead -- the trade,
    #: and what it costs, are in docs/elec-demand-timeseries.md.
    PROFILE_WORKBOOK = 'elec_2030_National_Trends.xlsx'

    #: The columns the reshaped cache is keyed by. Everything else in it is a
    #: country.
    INDEX_COLUMNS = ('year', 'month', 'day', 'hour')

    #: Countries the cache is built for -- a hard gate, not a preference. The
    #: workbook holds many more zones, and this is the set that has been checked
    #: by hand. Widening the tuple is a deliberate act that invalidates the cache.
    #:
    #: Deliberately wider than any single config's country_codes, so that one
    #: cache serves every config and adding a country to a run costs nothing.
    ALLOWED_COUNTRIES = (
        'AT00', 'BE00', 'CH00', 'DE00', 'DKW1', 'DKE1', 'EE00', 'ES00',
        'FI00', 'FR00', 'LT00', 'LV00', 'NL00', 'NOS0', 'NOM1',
        'NON1', 'PL00', 'SE01', 'SE02', 'SE03', 'SE04', 'UK00',
        'ITN1', 'ITCN', 'ITCS', 'PT00',
    )

    #: Bumped when what the cache holds, or what is proved before it is written,
    #: changes. A cache whose receipt names a different version is rebuilt.
    PARQUET_CONTRACT_VERSION = 1

    #: Where the receipt lives in the parquet's schema metadata.
    RECEIPT_KEY = b'elec_demand_receipt'

    def __init__(self, **kwargs_processor):
        """Initialize the processor. Required kwargs are listed below."""
        super().__init__(**kwargs_processor)

        required_params = [
            'input_folder',
            'country_codes',
            'start_year',
            'end_year',
            'df_annual_demands',
            'demand_grid',
        ]

        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Named one by one rather than through setattr, so that the attributes
        # are visible to a reader and to a checker.
        self.input_folder      = kwargs_processor['input_folder']
        self.country_codes     = kwargs_processor['country_codes']
        self.start_year        = kwargs_processor['start_year']
        self.end_year          = kwargs_processor['end_year']
        self.df_annual_demands = kwargs_processor['df_annual_demands']
        self.demand_grid       = kwargs_processor['demand_grid']

        # ProcessorRunner rounds the output to this many decimals after the
        # processor returns, so it decides whether a small value survives as a
        # number or reaches GAMS as a zero. _check_no_zero_hours needs it.
        try:
            self.rounding_precision = int(kwargs_processor.get('rounding_precision') or 0)
        except (TypeError, ValueError):
            self.rounding_precision = 0

        self.start_date = pd.Timestamp(f"{self.start_year}-01-01")
        self.end_date   = pd.Timestamp(f"{self.end_year}-12-31 23:00")

        # The cache sits beside the workbook and takes its name, so the two are
        # obviously a pair. It is keyed on nothing else -- not the config, not
        # the country set, not the climate range -- which is what lets one cache
        # serve every run.
        self.input_file = os.path.join(self.input_folder, self.PROFILE_WORKBOOK)
        self.parquet_file = os.path.join(
            self.input_folder, self.PROFILE_WORKBOOK.replace('.xlsx', '.parquet')
        )

        #: (label, reason, level) for everything that could not be built, said
        #: once at the end rather than as it happens.
        self.unbuilt_nodes = []

        #: country -> why it has no usable profile, expanded to node names when
        #: reporting.
        self.country_problems = {}

        #: node -> its constant_share, so the zero-hour alarm can name the cause
        #: it most often has.
        self.node_shares = {}

        #: configured country code -> the cache column that serves it. Filled by
        #: `_country_problems`, which is the one place that resolves spelling.
        self.country_columns = {}

        #: country and node -> how many source hours are negative. Counted from
        #: the loaded frame rather than the receipt, so it cannot disagree with
        #: the data being used. It lets the zero-hour alarm tell a profile that
        #: genuinely crosses zero from a fabricated zero.
        self.country_negative_hours = {}
        self.node_negative_hours = {}

    def output_index(self) -> pd.DatetimeIndex:
        """Every hour the processor has to produce."""
        return pd.date_range(self.start_date, self.end_date, freq='60min')

    def requested_years(self) -> list:
        """Every climate year the run asks for."""
        return list(range(int(self.start_year), int(self.end_year) + 1))

    # ------------------------------------------------------------------
    # The receipt, and whether the cache can be trusted
    # ------------------------------------------------------------------

    def read_receipt(self):
        """The receipt stored in the cache's schema metadata, or None.

        `read_schema` reads the parquet footer only, so this costs nothing next
        to loading the file -- which is what makes it affordable to ask on every
        run.
        """
        try:
            metadata = pq.read_schema(self.parquet_file).metadata or {}
            raw = metadata.get(self.RECEIPT_KEY)
            if raw is None:
                return None
            receipt = json.loads(bytes(raw).decode('utf-8'))
        except Exception:
            # A receipt that cannot be read says nothing, and the caller rebuilds
            # -- the same answer as for a missing one, so nothing to distinguish.
            return None
        return receipt if isinstance(receipt, dict) else None

    def _rebuild_reason(self):
        """Why the cache cannot be trusted, or None if it can.

        Deliberately silent -- the caller decides what to say, because the same
        finding reads differently depending on whether the workbook is there to
        rebuild from.

        A shortfall of climate *years* is never a reason. The receipt was written
        by a reader that had the workbook open; if it says a country lacks 2017,
        re-reading will not produce one. Those are reported against what the run
        asks for instead, in `_country_problems`.
        """
        parquet_path = Path(self.parquet_file)
        excel_path = Path(self.input_file)

        if not parquet_path.exists():
            return "there is no cache file yet"

        receipt = self.read_receipt()
        if receipt is None:
            return (
                "it carries no readable receipt, so nothing says which checks "
                "were run when it was written"
            )

        if receipt.get('contract_version') != self.PARQUET_CONTRACT_VERSION:
            return (
                f"its receipt was written by contract version "
                f"{receipt.get('contract_version')!r}, and this is version "
                f"{self.PARQUET_CONTRACT_VERSION}"
            )

        known = set(receipt.get('countries') or {}) | set(receipt.get('sheets_rejected') or {})
        unaccounted = sorted(set(self.ALLOWED_COUNTRIES) - known)
        if unaccounted:
            return (
                f"it accounts for neither the data nor the absence of "
                f"{', '.join(unaccounted)}, which ALLOWED_COUNTRIES names"
            )

        # Everything below compares the cache against the workbook, so it needs
        # one. With none there is nothing to rebuild from either, and answering
        # "stale" would refuse a cache that is the only copy of the data --
        # `load_profiles` says instead that freshness could not be confirmed.
        if excel_path.exists():
            stat = excel_path.stat()
            if receipt.get('source_bytes') != stat.st_size:
                return (
                    f"the workbook is now {stat.st_size} bytes and the receipt "
                    f"records {receipt.get('source_bytes')!r}"
                )
            if receipt.get('source_mtime_ns') != stat.st_mtime_ns:
                return "the workbook has been modified since the cache was written"
            if Path(__file__).stat().st_mtime > parquet_path.stat().st_mtime:
                return "this processor has been edited since the cache was written"

        return None

    def need_to_create_parquet_cache(self) -> bool:
        """Whether the cache has to be rebuilt from the workbook, saying why."""
        reason = self._rebuild_reason()
        if reason is None:
            self.logger.log_status("Using parquet cache.", level="none")
            return False
        self.logger.log_status(
            f"The parquet cache '{self.parquet_file}' has to be rebuilt because {reason}.",
            level="none",
        )
        return True

    # ------------------------------------------------------------------
    # Reading and proving the workbook
    # ------------------------------------------------------------------

    def _year_coverage(self, values: pd.DataFrame) -> dict:
        """Classify each year column of one sheet as complete, partial or empty.

        The three-way split is the point. An absent year is ordinary raggedness,
        and a run that does not ask for it should hear nothing. A *partial* year
        is the dangerous one: invisible downstream, and it biases normalisation
        because the year counts in full while its sum covers only the hours that
        are there.

        An all-zero year is recorded separately again and is no more usable than
        an empty one -- electricity demand is never zero, so a column of zeros is
        a gap the workbook spelled with a number -- but the two say different
        things about the source.
        """
        complete, partial, empty, all_zero = [], {}, [], []
        for year in values.columns:
            column = pd.to_numeric(values[year], errors='coerce')
            n_values = int(column.notna().sum())
            if n_values == 0:
                empty.append(int(year))
            elif n_values < self.STANDARD_YEAR_HOURS:
                partial[int(year)] = n_values
            elif not (column.fillna(0) != 0).any():
                all_zero.append(int(year))
            else:
                complete.append(int(year))
        return {
            'complete': complete,
            'partial': partial,
            'empty': empty,
            'all_zero': all_zero,
        }

    def _read_sheet(self, sheet_name: str, calendar):
        """One country's sheet, proved and reshaped to long format.

        Returns
        -------
        (pd.DataFrame, dict, np.ndarray) or (None, str, None)
            On success the long frame `['month', 'day', 'hour', 'year', 'value']`,
            its year coverage, and the standardised calendar it uses. On failure
            None and the reason it could not be used.

        `calendar` is the calendar proved by the first sheet that succeeded, or
        None for the first one. Every sheet has to agree about it, because the
        pivot that follows silently fills a disagreeing country with NaN.
        """
        df = self.read_input_excel(self.input_file, sheet_name=sheet_name,
                                   header=self.HEADER_ROW)

        if 'Date' not in df.columns:
            return None, (
                f"its sheet has no 'Date' column, so the header row is not where "
                f"row {self.HEADER_ROW + 1} is expected to be "
                f"(columns found: {', '.join(str(c) for c in df.columns[:8])})"
            ), None
        df = df.drop(labels='Date', axis=1)

        if len(df.columns) < 4:
            return None, (
                f"its sheet has only {len(df.columns)} column(s) after 'Date', so "
                f"it carries no climate years"
            ), None

        # The first three columns are the calendar. Renamed by position because
        # their titles vary, then checked by value -- which is the only way to
        # notice that the layout has moved.
        df = df.rename(columns={
            df.columns[0]: 'month',
            df.columns[1]: 'day',
            df.columns[2]: 'hour',
        })

        for name, low, high in (('month', 0, 11), ('day', 1, 31), ('hour', 0, 23)):
            column = pd.to_numeric(df[name], errors='coerce')
            if column.isna().any() or column.min() < low or column.max() > high:
                return None, (
                    f"its '{name}' column runs {column.min()}..{column.max()} where "
                    f"{low}..{high} is expected, so the calendar columns are not "
                    f"where they are read from"
                ), None
            df[name] = column.astype(int)

        # Months are 0-indexed in the source; make them calendar months here so
        # nothing downstream has to remember.
        df['month'] = df['month'] + 1

        # Year columns by complement, not by a positional slice: an off-by-one
        # slice here silently drops the first climate year from every sheet, and
        # taking everything that is not a calendar column makes that class of
        # mistake impossible rather than merely detectable.
        year_columns = [c for c in df.columns if c not in ('month', 'day', 'hour')]
        years = {}
        for column in year_columns:
            try:
                year = int(str(column).strip())
            except (TypeError, ValueError):
                return None, (
                    f"its column {column!r} is neither a calendar column nor a "
                    f"year, so the sheet's layout is not what is read"
                ), None
            years[column] = year
        if not years:
            return None, "its sheet carries no year columns", None

        current_calendar = df[['month', 'day', 'hour']].to_numpy()
        if len(current_calendar) != self.STANDARD_YEAR_HOURS:
            return None, (
                f"its sheet has {len(current_calendar)} rows where a standardised "
                f"year has {self.STANDARD_YEAR_HOURS}"
            ), None
        if len(np.unique(current_calendar, axis=0)) != self.STANDARD_YEAR_HOURS:
            return None, (
                f"its sheet repeats or skips hours: {self.STANDARD_YEAR_HOURS} rows "
                f"name only {len(np.unique(current_calendar, axis=0))} distinct hours"
            ), None
        if calendar is not None and not np.array_equal(current_calendar, calendar):
            return None, (
                "its sheet uses a different hour ordering from the sheets before "
                "it, which the reshape below cannot combine"
            ), None

        values = df[year_columns].rename(columns=years)
        coverage = self._year_coverage(values)

        negatives = int((values.apply(pd.to_numeric, errors='coerce') < 0).to_numpy().sum())
        if negatives:
            self.logger.log_status(
                f"[{sheet_name}] {negatives} negative demand value(s) in the source "
                f"workbook. They reach the model as generation rather than demand.",
                level="warn",
            )

        long = df[['month', 'day', 'hour']].join(values).melt(
            id_vars=['month', 'day', 'hour'],
            var_name='year',
            value_name='value',
        )
        long['country'] = sheet_name
        return long, coverage, current_calendar

    def read_excel_to_parquet(self) -> bool:
        """Read every allowed country out of the workbook and write the cache.

        This is the expensive path and the only one that sees the workbook, so it
        is where everything file-shaped is proved. A sheet that fails is excluded
        and its reason recorded in the receipt -- not skipped silently, and not
        allowed to stop the write, because re-reading hundreds of megabytes to
        rediscover the same failure is what the receipt exists to prevent.

        `SourceDataError` out of `read_input_excel` is the one exception and
        propagates: a malformed number in a generated workbook means the producer
        changed format, and no part of the file can be trusted after that.

        Returns whether a cache was written.
        """
        self.logger.log_status(
            f"Reading the whole workbook '{self.input_file}' -- this takes a minute or so, "
            f"and only happens when the cache cannot be trusted.",
            level="none",
        )

        try:
            with pd.ExcelFile(self.input_file) as workbook:
                available_sheets = list(workbook.sheet_names)
        except (FileNotFoundError, OSError, ValueError) as e:
            self.logger.log_status(
                f"Unable to open '{self.input_file}': {e}. No electricity demand "
                f"time series can be built.",
                level="warn",
            )
            return False

        rejected = {}
        coverages = {}
        pieces = []
        calendar = None

        # Sorted rather than a set intersection, so the read order -- and any
        # message about it -- is the same on every machine.
        for sheet_name in sorted(self.ALLOWED_COUNTRIES):
            if sheet_name not in available_sheets:
                rejected[sheet_name] = "the workbook has no sheet of that name"
                continue
            try:
                long, verdict, calendar_seen = self._read_sheet(sheet_name, calendar)
            except SourceDataError:
                # Reported in detail by the reader. Let it stop the processor.
                raise
            except Exception as e:
                rejected[sheet_name] = f"reading its sheet failed: {e}"
                continue

            # `verdict` is the year coverage when the sheet was usable and the
            # reason it was not when it was not -- the two never both apply.
            if long is None:
                rejected[sheet_name] = verdict
                continue

            calendar = calendar_seen
            coverages[sheet_name] = verdict
            pieces.append(long)

        # Two findings, said separately. A country the workbook does not carry is
        # one fact repeated; a country whose sheet failed a check is a different
        # fact each time, and burying the second in a list of the first is how it
        # gets skimmed past.
        absent = sorted(n for n, why in rejected.items() if n not in available_sheets)
        if absent:
            self.logger.log_status(
                f"'{self.input_file}' has no sheet for {len(absent)} of the "
                f"{len(self.ALLOWED_COUNTRIES)} allowed country(-ies): "
                f"{', '.join(absent)}. A run that asks for one of them is told so.",
                level="warn",
            )
        unusable = sorted((n, why) for n, why in rejected.items() if n in available_sheets)
        if unusable:
            detail = ', '.join(f"{name} ({why})" for name, why in unusable)
            self.logger.log_status(
                f"{len(unusable)} country sheet(s) in '{self.input_file}' could not be "
                f"read: {detail}. The reason is recorded in the cache, so a run that asks "
                f"for one of them is told why.",
                level="warn",
            )

        partial = {
            name: coverage['partial']
            for name, coverage in coverages.items() if coverage['partial']
        }
        if partial:
            detail = ', '.join(
                f"{name} {sorted(years)}" for name, years in sorted(partial.items())
            )
            self.logger.log_status(
                f"{len(partial)} country(-ies) have climate year(s) with some but not all "
                f"{self.STANDARD_YEAR_HOURS} hours: {detail}. A part-year would inflate every "
                f"year of that country when normalising, so it is refused: a run asking for "
                f"one of these years does not get that country.",
                level="warn",
            )

        if not pieces:
            self.logger.log_status(
                f"No usable country sheet in '{self.input_file}'. No electricity demand "
                f"time series can be built.",
                level="warn",
            )
            return False

        self.logger.log_status("Combining data from all sheets...", level="none")
        combined = pd.concat(pieces, ignore_index=True)

        self.logger.log_status("Pivoting to wide format...", level="none")
        # A plain pivot, not pivot_table: the calendar checks have already proved
        # there are no duplicate keys to aggregate, and pivot_table's default
        # dropna would quietly delete a country whose every value is missing --
        # exactly the case that has to stay visible.
        wide = combined.pivot(
            index=['year', 'month', 'day', 'hour'],
            columns='country',
            values='value',
        ).reset_index()
        wide.columns.name = None

        stat = Path(self.input_file).stat()
        receipt = {
            'contract_version': self.PARQUET_CONTRACT_VERSION,
            'source_name': os.path.basename(self.input_file),
            'source_bytes': stat.st_size,
            'source_mtime_ns': stat.st_mtime_ns,
            'standard_year_hours': self.STANDARD_YEAR_HOURS,
            'sheets_seen': sorted(available_sheets),
            'sheets_rejected': rejected,
            'countries': coverages,
        }

        self.logger.log_status(f"Saving parquet cache to: '{self.parquet_file}'...", level="none")
        table = pa.Table.from_pandas(wide, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[self.RECEIPT_KEY] = json.dumps(receipt).encode('utf-8')
        pq.write_table(
            table.replace_schema_metadata(metadata),
            self.parquet_file,
            compression='snappy',
        )
        return True

    # ------------------------------------------------------------------
    # Reading the cache
    # ------------------------------------------------------------------

    def load_data_from_parquet(self):
        """The whole cache, or None if it cannot be read.

        Every column is loaded rather than only the configured countries: the
        per-country checks need to say *which* countries the cache does hold, and
        subsetting saves less than that message is worth.
        """
        self.logger.log_status(f"Loading parquet cache: '{self.parquet_file}'...", level="none")
        try:
            return pd.read_parquet(self.parquet_file, engine='pyarrow')
        except Exception as e:
            self.logger.log_status(
                f"Unable to read the parquet cache '{self.parquet_file}': {e}. "
                f"No electricity demand time series can be built. Deleting the file "
                f"will rebuild it from the workbook.",
                level="warn",
            )
            return None

    def load_profiles(self):
        """The cached profiles, narrowed to the requested climate years.

        Rebuilds the cache first if it cannot be trusted. Never raises: a source
        that cannot be read at all returns None and the run continues with no
        electricity demand.

        The year filter runs on the complete row structure, before any datetime
        is computed, so a country that lacks a year keeps its rows and shows the
        gap as missing values rather than as absent rows.
        """
        workbook_exists = Path(self.input_file).exists()

        if self.need_to_create_parquet_cache():
            if not workbook_exists:
                self.logger.log_status(
                    f"The source workbook '{self.input_file}' is not present, so the "
                    f"parquet cache cannot be rebuilt. No electricity demand time series "
                    f"can be built. See the README for where to download the workbook.",
                    level="warn",
                )
                return None
            if not self.read_excel_to_parquet():
                return None
        elif not workbook_exists:
            self.logger.log_status(
                f"The source workbook is not present, so the cache's freshness could not "
                f"be confirmed -- its receipt says it was written from "
                f"{(self.read_receipt() or {}).get('source_name', 'an unknown file')}, and "
                f"the checks recorded there are being trusted.",
                level="info",
            )

        df_wide = self.load_data_from_parquet()
        if df_wide is None:
            return None

        missing_index = [c for c in self.INDEX_COLUMNS if c not in df_wide.columns]
        if missing_index:
            self.logger.log_status(
                f"The parquet cache '{self.parquet_file}' has no "
                f"{', '.join(missing_index)} column, so its rows cannot be dated. "
                f"No electricity demand time series can be built. Deleting the file "
                f"will rebuild it from the workbook.",
                level="error",
            )
            return None

        wanted = self.requested_years()
        df_wide = df_wide[df_wide['year'].isin(wanted)].copy()
        self.logger.log_status(
            f"Filtered to climate years {self.start_year}-{self.end_year}: "
            f"{len(df_wide)} rows.",
            level="info",
        )
        return df_wide

    # ------------------------------------------------------------------
    # Which countries can be built
    # ------------------------------------------------------------------

    def _country_problems(self, df_wide, countries) -> dict:
        """country -> why it cannot be built, for those that cannot.

        The year check reads the loaded frame rather than the receipt. Both know
        the answer, and the frame is the one that will actually be used.

        Only years the run asks for are judged: coverage is ragged at the tail
        for most countries in the workbook, and a country that stops at 2016 is
        complete as far as a 1982-2016 run is concerned.

        Spelling is resolved here and only here -- a configured `'fi00'` finds
        the `FI00` sheet -- with `self.country_columns` recording the resolution
        rather than each caller repeating it. Case-folding twice with different
        answers is what let a demand row go missing in the heat processor.
        """
        problems = {}
        allowed_by_code = {c.upper(): c for c in self.ALLOWED_COUNTRIES}
        columns_by_code = {
            str(c).strip().upper(): c
            for c in df_wide.columns if c not in self.INDEX_COLUMNS
        }
        wanted = self.requested_years()
        receipt = self.read_receipt() or {}
        rejected = {str(k).strip().upper(): v
                    for k, v in (receipt.get('sheets_rejected') or {}).items()}

        for country in countries:
            code = str(country).strip().upper()
            if code not in allowed_by_code:
                problems[country] = (
                    f"it is not in elec_demand_TYNDP2024.ALLOWED_COUNTRIES, so the "
                    f"cache was never built for it. Add it there to include it"
                )
                continue
            if code in rejected:
                problems[country] = (
                    f"the workbook could not be read for it -- {rejected[code]}"
                )
                continue
            column = columns_by_code.get(code)
            if column is None:
                problems[country] = "the parquet cache has no column for it"
                continue

            present = df_wide.groupby('year')[column].count()
            short = [
                year for year in wanted
                if int(present.get(year, 0)) != self.STANDARD_YEAR_HOURS
            ]
            if short:
                problems[country] = (
                    f"the workbook has no complete profile for climate year(s) "
                    f"{', '.join(str(y) for y in short)}, which this run asks for"
                )
                continue

            # A year of zeros passes the count above and is still not data:
            # electricity demand is never zero, so it is a gap the workbook
            # spelled with a number instead of a blank.
            nonzero = (df_wide[column].fillna(0) != 0).groupby(df_wide['year']).sum()
            flat = [year for year in wanted if int(nonzero.get(year, 0)) == 0]
            if flat:
                problems[country] = (
                    f"every hour of climate year(s) "
                    f"{', '.join(str(y) for y in flat)} is zero in the workbook, which "
                    f"is a gap rather than a demand"
                )
                continue

            self.country_columns[country] = column

        return problems

    def get_demand_profiles(self, countries) -> pd.DataFrame:
        """Hourly profile per country, indexed by every hour of the run.

        A country that cannot be built keeps a column of NaN, which the GDX gate
        turns into zeros; its reason is recorded for `_report_coverage` and
        alarmed about by `_check_no_zero_hours`. The column is kept rather than
        dropped so that the node still appears in the output it was asked for.
        """
        out_index = self.output_index()
        df_profiles = pd.DataFrame(index=out_index)

        df_wide = self.load_profiles()
        if df_wide is None or df_wide.empty:
            if df_wide is not None:
                self.logger.log_status(
                    f"The parquet cache holds no rows for climate years "
                    f"{self.start_year}-{self.end_year}. No electricity demand time "
                    f"series can be built.",
                    level="warn",
                )
            for country in countries:
                self.country_problems[country] = (
                    "no electricity demand profile source could be read"
                )
                df_profiles[country] = np.nan
            return df_profiles

        self.country_problems.update(self._country_problems(df_wide, countries))
        buildable = [c for c in countries if c not in self.country_problems]

        for country in countries:
            if country in self.country_problems:
                df_profiles[country] = np.nan

        if not buildable:
            return df_profiles

        self.logger.log_status("Dating the profiles and inserting leap days...")
        # dict.fromkeys rather than a list: two configured codes can resolve to
        # one cache column, and a duplicated selection makes `dated[column]` a
        # frame instead of a series.
        columns = list(self.INDEX_COLUMNS) + list(
            dict.fromkeys(self.country_columns[c] for c in buildable)
        )
        dated = self.process_datetime_index(df_wide[columns])

        for country in buildable:
            column = self.country_columns[country]
            # Counted on the requested window, which is what will be built -- not
            # on the whole cache, where a year nobody asked for could make a clean
            # run look dirty.
            self.country_negative_hours[country] = int((df_wide[column] < 0).sum())
            series = dated[column].reindex(out_index)
            if series.isna().any():
                # Unreachable: the calendar was proved complete before the cache
                # was written and _country_problems has just proved every
                # requested year whole. Guarded because both promises are made
                # elsewhere.
                self.logger.log_status(
                    f"{country}: the demand profile has {int(series.isna().sum())} hour(s) "
                    f"with no value after the source was proved complete. Not building it.",
                    level="error",
                )
                self.country_problems[country] = "its demand profile came out incomplete"
                df_profiles[country] = np.nan
                continue
            df_profiles[country] = series

        return df_profiles

    # ------------------------------------------------------------------
    # Demand rows
    # ------------------------------------------------------------------

    def _column(self, df, name):
        """A column by case-folded name, or None.

        `merge_row_by_row` compares headers case-insensitively and keeps the
        first spelling it saw, so the case here is decided by whichever workbook
        was read first. Looking it up folded is the only stable way.
        """
        for col in df.columns:
            if str(col).strip().lower() == name:
                return col
        return None

    def _demand_rows_by_country(self):
        """
        Group the demand rows by case-folded country code.

        Case folding happens here and only here. It used to be split -- `process`
        did no country filtering at all while `build_demands` matched folded --
        so which spellings survived depended on which method you read.

        Returns
        -------
        dict[str, list[dict]] or None
            None if the table has no usable `country`, `node` or `twh/year` column.
        """
        df = self.df_annual_demands
        country_col = self._column(df, 'country')
        node_col = self._column(df, 'node')
        if country_col is None or node_col is None:
            missing = [n for n, c in (('country', country_col), ('node', node_col))
                       if c is None]
            self.logger.log_status(
                f"The {self.demand_grid} demand table has no {' or '.join(missing)} "
                f"column, so its rows cannot be turned into nodes. Columns found: "
                f"{', '.join(str(c) for c in df.columns)}. "
                f"No electricity demand can be built.",
                level="error",
            )
            return None

        twh_col = self._column(df, 'twh/year')
        if twh_col is None:
            self.logger.log_status(
                f"The {self.demand_grid} demand table has no 'twh/year' column, so there "
                f"is no annual energy to distribute. Columns found: "
                f"{', '.join(str(c) for c in df.columns)}. "
                f"No electricity demand can be built.",
                level="error",
            )
            return None

        # Optional, unlike the three above, and blank for every electricity node
        # shipped today: an absent constant_share becomes 0.0 in build_demands,
        # which leaves the profile carrying the whole demand.
        share_col = self._column(df, 'constant_share')

        grouped = {}
        for _, row in df.iterrows():
            country = row[country_col]
            if pd.isna(country):
                continue
            grouped.setdefault(str(country).strip().lower(), []).append({
                'node': str(row[node_col]),
                'twh': row[twh_col],
                'share': row[share_col] if share_col is not None else pd.NA,
            })
        return grouped

    # ------------------------------------------------------------------
    # The three stages
    # ------------------------------------------------------------------

    def process_datetime_index(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Give the pivoted frame a real hourly datetime index.

        Every source year is a standardised 365-day, 8760-hour calendar with
        1-indexed months, so a leap year's 8784 output hours have to be built
        from 8760 source hours: standard day 60 becomes Feb 29, days 61-365 shift
        to Mar 1 - Dec 30, and day 365 is used a second time for Dec 31. That
        keeps the order chronological. See "The source has no leap years" in
        docs/elec-demand-timeseries.md.

        Parameters
        ----------
        input_df : pd.DataFrame
            ['year', 'month', 'day', 'hour'] plus one column per country.

        Returns
        -------
        pd.DataFrame
            Indexed by every hour of the run, country columns only.
        """
        if input_df.empty:
            raise ValueError("Input DataFrame is empty - cannot process datetime index")

        input_df = input_df.copy()

        # Position in the 365-day standardised year, via a non-leap reference.
        ref_dates = pd.to_datetime(dict(
            year=2001,
            month=input_df['month'].astype(int),
            day=input_df['day'].astype(int)
        ))
        input_df['std_doy'] = ref_dates.dt.dayofyear

        # Jan 1 + (std_doy - 1) days + hour. pd.Timedelta respects the real
        # calendar, so in a leap year std_doy 60 lands on Feb 29 and everything
        # after it shifts by a day without any special case here.
        year_starts = pd.to_datetime(dict(
            year=input_df['year'].astype(int),
            month=1,
            day=1
        ))
        input_df['datetime'] = (
            year_starts
            + pd.to_timedelta(input_df['std_doy'].astype(int) - 1, unit='D')
            + pd.to_timedelta(input_df['hour'].astype(int), unit='h')
        )

        # Which leaves Dec 31 of a leap year with no source row, since std_doy
        # 365 has landed on Dec 30. It gets day 365 a second time.
        years = input_df['year'].astype(int)
        is_leap = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
        leap_last_day_mask = is_leap & (input_df['std_doy'] == 365)

        if leap_last_day_mask.any():
            extra_rows = input_df[leap_last_day_mask].copy()
            extra_rows['datetime'] = pd.to_datetime(dict(
                year=extra_rows['year'].astype(int),
                month=12,
                day=31,
                hour=extra_rows['hour'].astype(int)
            ))
            input_df = pd.concat([input_df, extra_rows], ignore_index=True)

        input_df = input_df.set_index('datetime').sort_index()
        input_df = input_df.drop(columns=['year', 'month', 'day', 'hour', 'std_doy'])

        if not input_df.index.is_unique:
            n_dupes = input_df.index.duplicated().sum()
            self.logger.log_status(
                f"{n_dupes} duplicate timestamps in the demand profiles after the leap-day "
                f"mapping, which should not be possible once the calendar has been proved. "
                f"Keeping the first of each.",
                level="warn",
            )
            input_df = input_df[~input_df.index.duplicated(keep='first')]

        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        return input_df.reindex(full_index)

    def normalize_profiles(self, df_profiles, countries) -> pd.DataFrame:
        """
        Scale each country's profile so the *mean* year sums to 1.

        Across the whole climate range, not within each year -- see "What
        TWh/year means" in docs/elec-demand-timeseries.md. How far a year
        departs from the mean is a property of the weather and of the country, so
        the run measures its own range and prints it rather than this claiming
        one.

        The year count is the number of calendar years in the index, not the
        number holding at least one positive hour. The old rule inflated demand
        whenever data was partial: a year with a single positive hour counted as
        a whole year while the sum covered only the hours present, so *every*
        year received more energy than the table said. A country reaching here
        now has a complete series across the requested range, so this states the
        intended rule directly instead of approximating it.
        """
        for country in countries:
            if country not in df_profiles.columns:
                continue
            s = df_profiles[country]
            if s.isna().all():
                continue

            if s.isna().any():
                # Unreachable: get_demand_profiles emits a complete column or an
                # all-NaN one. Guarded because that promise is made elsewhere.
                self.logger.log_status(
                    f"{country}: demand profile has {int(s.isna().sum())} hour(s) with no "
                    f"value after the source was proved complete. Not building it.",
                    level="error",
                )
                self.country_problems.setdefault(
                    country, "its demand profile came out incomplete"
                )
                df_profiles[country] = np.nan
                continue

            total = float(s.sum())
            if total <= 0:
                self.country_problems.setdefault(country, (
                    "its profile sums to zero or less over the requested climate "
                    "years, so there is no shape to scale"
                ))
                df_profiles[country] = np.nan
                continue

            n_years = int(s.index.year.nunique())
            df_profiles[country] = s * (n_years / total)

        return df_profiles

    def build_demands(self, df_profiles_norm, rows_by_country, countries) -> pd.DataFrame:
        """
        Scale normalized profiles to absolute hourly demand, one column per node.

            demand(t) = A * profile(t) + B
            A = annual_demand * (1 - constant_share)     weather-driven
            B = annual_demand * constant_share / 8760    flat base load

        A row that cannot be used -- an unusable `twh/year`, a `constant_share`
        outside [0, 1] -- costs that one node and is reported. It used to raise,
        and ProcessorRunner catches at whole-processor level, so one bad cell
        cost every node in the run its time series.
        """
        columns = {}
        claimed_by = {}
        self.node_shares = {}
        self.node_negative_hours = {}

        for country in countries:
            profile = df_profiles_norm.get(country)
            problem = self.country_problems.get(country)
            if profile is None and problem is None:
                # Unreachable: a column exists per country either way. Guarded
                # because the alternative is a TypeError further down rather
                # than a stated reason.
                problem = "no demand profile was produced for it"

            for row in rows_by_country.get(str(country).strip().lower(), []):
                node = row['node']

                if node in claimed_by:
                    self.logger.log_status(
                        f"Node '{node}' is named by more than one {self.demand_grid} demand "
                        f"row -- countries {claimed_by[node]} and {country}. Keeping the "
                        f"first; the second is ignored. One of the two country cells is "
                        f"almost certainly wrong.",
                        level="warn",
                    )
                    continue
                claimed_by[node] = country

                if problem is not None:
                    self.unbuilt_nodes.append((node, problem, "warn"))
                    columns[node] = pd.Series(np.nan, index=df_profiles_norm.index)
                    continue

                annual = pd.to_numeric(row['twh'], errors='coerce')
                if pd.isna(annual):
                    self.unbuilt_nodes.append((
                        node, f"its twh/year is {row['twh']!r}, which is not a number", "warn"
                    ))
                    columns[node] = pd.Series(np.nan, index=df_profiles_norm.index)
                    continue
                if annual < 0:
                    self.unbuilt_nodes.append((
                        node, f"its twh/year is negative ({annual})", "warn"
                    ))
                    columns[node] = pd.Series(np.nan, index=df_profiles_norm.index)
                    continue

                share = pd.to_numeric(row['share'], errors='coerce')
                if pd.isna(share):
                    share = 0.0
                share = float(share)
                if not (0 <= share <= 1):
                    self.unbuilt_nodes.append((
                        node, f"its constant_share is {share}, which is not between 0 and 1",
                        "warn"
                    ))
                    columns[node] = pd.Series(np.nan, index=df_profiles_norm.index)
                    continue

                annual_demand = float(annual) * self.TWH_TO_MWH
                A = annual_demand * (1 - share)
                B = annual_demand * share / self.HOURS_PER_YEAR
                columns[node] = A * profile + B
                self.node_shares[node] = share
                self.node_negative_hours[node] = self.country_negative_hours.get(country, 0)

        if not columns:
            return pd.DataFrame(index=df_profiles_norm.index)
        return pd.DataFrame(columns, index=df_profiles_norm.index)

    # ------------------------------------------------------------------
    # Saying what happened
    # ------------------------------------------------------------------

    def _check_no_zero_hours(self, df_demands):
        """
        Alarm about any hour of any node that comes out with no demand.

        Every zero here is a symptom -- a country outside ALLOWED_COUNTRIES, a
        climate year the workbook does not cover, a `twh/year` that is not a
        number -- and every one would otherwise reach GAMS looking like a
        modelled hour of nothing. See the zero hours section of
        docs/elec-demand-timeseries.md.

        The test is against what will actually be *written*: ProcessorRunner
        rounds to `rounding_precision` after this returns, so a node at 0.4 MWh/h
        leaves here non-zero and arrives at GAMS as nothing. NaN counts as zero
        for the same reason, the GDX gate filling it with one.

        One line per node, because a broken node has hundreds of thousands of bad
        hours, and never raised. Graded by cause: a node whose source profile goes
        negative has to cross zero to get there, so those hours are a warning,
        while anything else is an error that forces a full rerun next build.
        """
        if df_demands.empty:
            return

        tolerance = 0.5 * (10.0 ** -self.rounding_precision)
        for node in df_demands.columns:
            series = df_demands[node]
            empty = series.isna() | (series.abs() < tolerance)
            if not empty.any():
                continue
            first = series.index[empty][0]
            reason = next(
                (why for name, why, _ in self.unbuilt_nodes if name == node), None
            )
            negatives = self.node_negative_hours.get(node, 0)
            if reason is None and negatives:
                self.logger.log_status(
                    f"Electricity node '{node}' has no demand in {int(empty.sum())} of "
                    f"{len(series)} hour(s), first {first}. Its source profile is negative "
                    f"for {negatives} hour(s), so it crosses zero on the way -- these are "
                    f"the crossings, not missing data. Backbone still reads them as "
                    f"'not set' rather than as a low demand.",
                    level="warn",
                )
                continue
            detail = f" -- {reason}" if reason else ""
            self.logger.log_status(
                f"Electricity node '{node}' has no demand in {int(empty.sum())} of "
                f"{len(series)} hour(s), first {first}{detail}. Backbone reads a zero as "
                f"'not set', so those hours arrive as no demand at all rather than as a low "
                f"one. Electricity demand does not stop; check the node's twh/year and "
                f"constant_share, and whether its country has profile data for every "
                f"climate year the run asks for.",
                level="error",
            )

    def _report_coverage(self, built_nodes):
        """
        Say once what was built and what was not, instead of leaving it to be noticed.

        Split by whether the user can act on it. A configured country with no
        electricity demand rows at all is information. A node someone wrote a
        demand row for and that could not be built is a warning: that row is a
        request, and the node will otherwise sit in the model with a balance
        penalty and nothing to serve.
        """
        self.logger.log_status(
            f"Electricity demand built for {len(built_nodes)} node(s).",
            level="info",
        )

        for level in ("warn", "info"):
            entries = sorted((n, r) for n, r, lvl in self.unbuilt_nodes if lvl == level)
            if not entries:
                continue
            # Grouped by reason, which is what a reader acts on and what the
            # nodes usually share. Names stay in full: a node reaching GAMS as
            # zero demand is exactly what someone has to go and look up.
            by_reason = {}
            for node, reason in entries:
                by_reason.setdefault(reason, []).append(node)
            detail = '; '.join(
                f"{', '.join(nodes)}: {reason}"
                for reason, nodes in sorted(by_reason.items())
            )
            self.logger.log_status(
                f"No electricity demand for {len(entries)} node(s) -- {detail}. "
                f"These reach GAMS as zero demand.",
                level=level,
            )

    def _report_climate_spread(self, df_demands, built_nodes):
        """
        State the normal-year property once, with this run's own numbers.

        Anyone comparing a single climate year against the workbook's `twh/year`
        will find a mismatch. It is not one -- the table figure is a
        weather-normalised normal year -- but nothing else in a build log says
        so, and the question comes back every time.
        """
        usable = [n for n in built_nodes if n in df_demands.columns]
        if df_demands.empty or not usable:
            return

        # Per node against its own mean, not summed across nodes first: a warm
        # year in one country offsets a cold one in another, and the total would
        # report no spread while every individual node swung by a tenth.
        totals = df_demands[usable].groupby(df_demands.index.year).sum()
        if len(totals) < 2:
            return
        means = totals.mean()
        usable = [n for n in usable if means[n] > 0]
        if not usable:
            return
        ratios = totals[usable] / means[usable]

        self.logger.log_status(
            f"Annual electricity demand matches the workbook's normal-year twh/year as a "
            f"{len(totals)}-year mean; individual climate years range "
            f"{ratios.min().min():.0%} to {ratios.max().max():.0%} of it. "
            f"A single-year build is not meant to reproduce the table figure.",
            level="info",
        )

    # ------------------------------------------------------------------

    def process(self) -> pd.DataFrame:
        """
        Run the full electricity demand pipeline.

        Returns
        -------
        pd.DataFrame
            Long format ['grid', 'node', 'time', 'value'], hourly, value negative.
        """
        rows_by_country = self._demand_rows_by_country()
        if rows_by_country is None:
            return pd.DataFrame(columns=['grid', 'node', 'time', 'value'])

        # The config is the authority on country-code spelling, here as
        # everywhere else: iterate it, look the demand rows up case-folded.
        processed_countries = [
            code for code in self.country_codes
            if str(code).strip().lower() in rows_by_country
        ]
        no_rows = [c for c in self.country_codes if c not in processed_countries]
        if no_rows:
            self.logger.log_status(
                f"No {self.demand_grid} demand rows for {len(no_rows)} configured "
                f"country(-ies), which therefore have no time series here: "
                f"{', '.join(map(str, no_rows))}.",
                level="info",
            )
        if not processed_countries:
            self.logger.log_status(
                f"No configured country has any {self.demand_grid} demand row. "
                f"Nothing to build.",
                level="warn",
            )
            return pd.DataFrame(columns=['grid', 'node', 'time', 'value'])

        self.logger.log_status("Reading electricity demand profiles...")
        out_df = self.get_demand_profiles(processed_countries)

        self.logger.log_status("Normalizing demand profiles...")
        out_df = self.normalize_profiles(out_df, processed_countries)

        self.logger.log_status("Building demand time series...")
        out_df = self.build_demands(out_df, rows_by_country, processed_countries)

        unbuilt = {node for node, _, _ in self.unbuilt_nodes}
        built_nodes = [c for c in out_df.columns if c not in unbuilt]

        self._check_no_zero_hours(out_df)
        self._report_coverage(built_nodes)
        self._report_climate_spread(out_df, built_nodes)

        self.logger.log_status("Demand time series built.", level="info")

        # Long format, and negated on the way out: demand is a negative ts_influx.
        result = out_df.reset_index(names='time')
        result = result.melt(id_vars=['time'], var_name='node', value_name='value')
        result['value'] = -result['value']
        result['grid'] = self.demand_grid
        return result[['grid', 'node', 'time', 'value']]
