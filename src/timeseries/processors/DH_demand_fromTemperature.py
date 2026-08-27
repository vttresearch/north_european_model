import os
import pandas as pd
import numpy as np
from src.timeseries.processors.base_processor import BaseProcessor, SourceDataError
from src.timeseries.timeseries_helpers import complete_native_grid


class DH_demand_fromTemperature(BaseProcessor):
    """
    District heating demand for each node, from one temperature series per country.

    Trailing 24-hour mean outdoor temperature, subtracted from a 17 C balance
    point and clipped at zero, normalised so the mean climate year sums to one,
    then split per node into a weather-driven and a flat part.

    `docs/dh-demand-timeseries.md` is the documentation and carries the reasoning
    -- what `twh/year` means, why the normalisation spans the whole climate range
    rather than each year, why an output zero is an alarm, and which countries
    can be built. Read it before changing anything here.

    Parameters
    ----------
    input_folder : str
        Folder holding `Temperature.csv`, pre-joined by ProcessorRunner.
    country_codes : list of str
        Countries the run includes. The authority on spelling.
    start_year, end_year : int
        First and last climate year.
    df_annual_demands : pd.DataFrame
        Demand rows already filtered to this processor's `demand_grid`.
    demand_grid : str
        The grid written into the output, e.g. `dheat`.

    `scenario_year` and `exclude_nodes` also arrive in the kwargs and are
    deliberately not read: SourceDataPipeline has already applied both, and
    filtering again would be a second place to keep in step.
    """

    #: Demand is written as negative `ts_influx`. No value_range to go with it:
    #: the meaningful bound is each network's own peak MW, which no class-level
    #: tuple can state, and a maximum of 0.0 would only restate the sign.
    value_sign = "non_positive"

    #: Outdoor temperature above which no space heating is needed. A modelling
    #: choice, not a tuning constant.
    BALANCE_POINT_C = 17.0

    #: Length of the trailing mean that stands in for building thermal mass.
    SMOOTHING_HOURS = 24

    #: Divisor for the flat share, deliberately the nominal year rather than the
    #: real one. `elec_demand_TYNDP2024` uses the identical formula, so the two
    #: have to move together or not at all.
    HOURS_PER_YEAR = 8760

    TWH_TO_MWH = 1e6

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

        # The rolling mean needs SMOOTHING_HOURS - 1 hours before the first
        # output hour, and no more. Derived rather than hardcoded so that the
        # coverage check states the actual requirement: hardcoding two days
        # silently costs a run its first 24 hours when the temperature file
        # starts at the same year as the run.
        self.warmup = pd.Timedelta(self.SMOOTHING_HOURS - 1, unit="h")

        self.temperature_file = os.path.join(self.input_folder, 'Temperature.csv')

        #: (label, reason, level) for everything that could not be built, said
        #: once at the end rather than as it happens.
        self.unbuilt_nodes = []

        #: country -> why it has no usable profile, expanded to node names when
        #: reporting.
        self.country_problems = {}

        #: node -> its constant_share, so the zero-hour alarm can name the cause
        #: it most often has.
        self.node_shares = {}


    # ------------------------------------------------------------------
    # Reading and proving the temperature source
    # ------------------------------------------------------------------

    def required_index(self) -> pd.DatetimeIndex:
        """Every hour the temperature file has to supply, warm-up included."""
        return pd.date_range(self.start_date - self.warmup, self.end_date, freq='60min')

    def output_index(self) -> pd.DatetimeIndex:
        """Every hour the processor has to produce."""
        return pd.date_range(self.start_date, self.end_date, freq='60min')

    def _read_temperature_grid(self):
        """
        Read Temperature.csv and return it indexed by exactly the hours needed.

        A file that cannot be used at all returns None and the run continues with
        no district heating demand; a merely damaged one is repaired where that
        is honest, reported where it is not, and handed on with holes left as NaN
        for the per-country pass to see. The only raise comes from
        `read_input_csv`, and it means the file has changed format.

        Returns
        -------
        pd.DataFrame or None
            Indexed by `required_index()`, one column per country in the file.
        """
        try:
            df = self.read_input_csv(self.temperature_file)
        except SourceDataError:
            # Reported in detail by the reader. Let it stop the processor.
            raise
        except (FileNotFoundError, OSError) as e:
            self.logger.log_status(
                f"Unable to open {self.temperature_file}: {e}. "
                f"No district heating demand can be built.",
                level="error",
            )
            return None

        time_col = next(
            (c for c in df.columns if str(c).strip().lower() == 'time'), None
        )
        if time_col is None:
            self.logger.log_status(
                f"[{self.temperature_file}] has no 'Time' column, so its rows cannot be "
                f"placed on a calendar. Columns found: "
                f"{', '.join(str(c) for c in df.columns)}. "
                f"No district heating demand can be built.",
                level="error",
            )
            return None

        # errors='coerce': read_input_csv only checks columns that look numeric,
        # so a malformed timestamp is caught here or nowhere.
        stamps = pd.to_datetime(df[time_col], errors='coerce')
        unparsed = stamps.isna()
        if unparsed.any():
            example = df.loc[unparsed, time_col].iloc[0]
            self.logger.log_status(
                f"[{self.temperature_file}] {int(unparsed.sum())} row(s) carry a timestamp "
                f"that is not a date, for example {example!r}. Those rows are dropped and "
                f"the hours they should have carried count as missing data.",
                level="warn",
            )
            df = df.loc[~unparsed]
            stamps = stamps.loc[~unparsed]

        df = df.drop(columns=[time_col])
        df.index = pd.DatetimeIndex(stamps.to_numpy(), name='time')

        duplicated = df.index.duplicated()
        if duplicated.any():
            self.logger.log_status(
                f"[{self.temperature_file}] {int(duplicated.sum())} timestamp(s) appear more "
                f"than once, first {df.index[duplicated][0]}. The first row of each is kept.",
                level="warn",
            )
            df = df[~duplicated]

        df = df.sort_index()

        required = self.required_index()
        absent = required.difference(df.index)
        if len(absent):
            self.logger.log_status(
                f"[{self.temperature_file}] {len(absent)} of the {len(required)} hour(s) "
                f"between {required[0]} and {required[-1]} are not in the file, first "
                f"{absent[0]}, last {absent[-1]}. They count as missing data for every "
                f"country. The window starts {self.warmup} before {self.start_date} because "
                f"the {self.SMOOTHING_HOURS}-hour mean needs that much history.",
                level="warn",
            )

        # Reindexing turns 'row absent' into 'value NaN', the form the
        # per-country completion below can act on.
        return df.reindex(required)

    def _temperature_column(self, country, columns_by_code):
        """
        The temperature column for a country code, or None.

        The rule is the first two characters -- 'FI00' -> 'FI', 'NOS0' -> 'NO',
        'DKW1' -> 'DK' -- so splitting a country into regions costs nothing. The
        price is that 'UK00' looks for 'UK' where the file says 'GB'; that is
        reported rather than aliased, for the reasons in "Which countries can be
        built" in docs/dh-demand-timeseries.md.
        """
        return columns_by_code.get(str(country)[:2].strip().upper())

    def _complete_country_series(self, series, required, label):
        """
        Make one country's hourly temperature series whole, or say why not.

        Returns (series, None) when the country can be built and (None, reason)
        when it cannot.

        Both of `complete_native_grid`'s zero flags are off here, the one place
        this caller diverges from the hydro ones: a zero in a reservoir level is
        a dropped value wearing a plausible costume, but a zero here is 0 C, an
        ordinary winter hour everywhere in the run.
        """
        if series is None or series.dropna().empty:
            return None, "the temperature file holds no values for it"

        filled, report = complete_native_grid(
            series,
            required,
            label=label,
            zero_is_missing=False,
            isolated_zero_is_missing=False,
        )

        if filled is None or filled.empty:
            return None, "the temperature file holds no values for it"

        # complete_native_grid drops everything before the first real value --
        # right for a record that simply begins later, wrong for the warm-up
        # hours, which are the ones most likely to be absent.
        head_gap = int((required < filled.index[0]).sum())
        if head_gap:
            return None, (
                f"the {head_gap} hour(s) from {required[0]} are absent, and the "
                f"{self.SMOOTHING_HOURS}-hour mean cannot start without them"
            )

        if report.n_left:
            return None, (
                f"{report.n_left} hour(s) have no temperature, longest run "
                f"{report.longest_run_left} hour(s) from {report.first_left}"
            )

        if report.n_autofilled:
            self.logger.log_status(
                f"{label}: filled {report.n_autofilled} single missing hour(s) in the "
                f"temperature series by interpolation.",
                level="info",
            )

        return filled.reindex(required), None


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

        Case folding happens here and only here. It used to happen twice with
        different answers -- `process` filtering the country list case-sensitively
        while `build_demands` matched folded -- so a workbook cell reading 'fi00'
        was dropped before the tolerant matcher ever saw it.

        Returns
        -------
        dict[str, list[dict]] or None
            None if the table has no usable `country` or `node` column.
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
                f"No district heating demand can be built.",
                level="error",
            )
            return None

        twh_col = self._column(df, 'twh/year')
        if twh_col is None:
            self.logger.log_status(
                f"The {self.demand_grid} demand table has no 'twh/year' column, so there is "
                f"no annual energy to distribute. Columns found: "
                f"{', '.join(str(c) for c in df.columns)}. "
                f"No district heating demand can be built.",
                level="error",
            )
            return None

        # Optional, unlike the three above: an absent constant_share becomes 0.0
        # in build_demands, which is legal and alarmed about as a zero hour.
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

    def get_temperature_profile(self, processed_countries, df_temperature):
        """
        Hourly heating profile per country: max(0, BALANCE_POINT_C - 24h mean).

        A country that cannot be built keeps a column of NaN, which the GDX gate
        turns into zeros; its reason is recorded for `_report_coverage` and
        alarmed about by `_check_no_zero_hours`. The column is kept rather than
        dropped so that the node still appears in the output it was asked for.

        Parameters
        ----------
        processed_countries : list[str]
            Country codes with at least one demand row.
        df_temperature : pd.DataFrame or None
            Output of `_read_temperature_grid`.

        Returns
        -------
        pd.DataFrame
            Indexed by `output_index()`, one column per country.
        """
        required = self.required_index()
        out_index = self.output_index()
        df_heating_profile = pd.DataFrame(index=out_index)

        if df_temperature is None:
            for country in processed_countries:
                self.country_problems[country] = "the temperature file could not be read"
                df_heating_profile[country] = np.nan
            return df_heating_profile

        columns_by_code = {
            str(c).strip().upper(): c for c in df_temperature.columns
        }

        for country in processed_countries:
            column = self._temperature_column(country, columns_by_code)
            if column is None:
                self.country_problems[country] = (
                    f"the temperature file has no '{str(country)[:2]}' column "
                    f"(it has {', '.join(sorted(columns_by_code))})"
                )
                df_heating_profile[country] = np.nan
                continue

            filled, reason = self._complete_country_series(
                df_temperature[column], required, country
            )
            if filled is None:
                self.country_problems[country] = reason
                df_heating_profile[country] = np.nan
                continue

            # Trailing mean, so the profile reflects the day leading up to an
            # hour rather than that instant. min_periods is the full window --
            # the warm-up hours in `required` exist to fill the first one.
            rolling_avg = filled.rolling(
                window=self.SMOOTHING_HOURS, min_periods=self.SMOOTHING_HOURS
            ).mean()
            profile = (self.BALANCE_POINT_C - rolling_avg).clip(lower=0)
            df_heating_profile[country] = profile.reindex(out_index)

        return df_heating_profile

    def normalize_profiles(self, df_profiles, processed_countries):
        """
        Scale each country's profile so the *mean* year sums to 1.

        Across the whole climate range, not within each year -- see "What
        TWh/year means" in docs/dh-demand-timeseries.md. How far a year
        departs from the mean is a property of the weather and of the node, so
        the run measures its own range and prints it rather than this claiming
        one.

        The year count is the number of calendar years in the index, not the
        number holding at least one positive hour. The old rule inflated demand
        whenever data was partial: a year with a single positive hour counted as
        a whole year while the sum covered only the hours present, so *every*
        year received more energy than the table said. A country reaching here
        now has a complete series, so this states the intended rule directly
        instead of approximating it.
        """
        for country in processed_countries:
            if country not in df_profiles.columns:
                continue
            s = df_profiles[country]
            if s.isna().all():
                continue

            if s.isna().any():
                # Unreachable: get_temperature_profile emits a complete column or
                # an all-NaN one. Guarded because that promise is made elsewhere.
                self.logger.log_status(
                    f"{country}: heating profile has {int(s.isna().sum())} hour(s) with no "
                    f"value after the temperature series was proved complete. "
                    f"Not building it.",
                    level="error",
                )
                self.country_problems.setdefault(
                    country, "its heating profile came out incomplete"
                )
                df_profiles[country] = np.nan
                continue

            total = float(s.sum())
            if total <= 0:
                self.country_problems.setdefault(country, (
                    f"its {self.SMOOTHING_HOURS}-hour mean temperature never falls below "
                    f"{self.BALANCE_POINT_C} C, so there is no weather-driven heating demand"
                ))
                df_profiles[country] = np.nan
                continue

            n_years = int(s.index.year.nunique())
            df_profiles[country] = s * (n_years / total)

        return df_profiles

    def build_demands(self, df_profiles_norm, rows_by_country, processed_countries):
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

        for country in processed_countries:
            profile = df_profiles_norm.get(country)
            problem = self.country_problems.get(country)
            if profile is None and problem is None:
                # Unreachable: a column exists per country either way. Guarded
                # because the alternative is a TypeError further down rather
                # than a stated reason.
                problem = "no heating profile was produced for it"

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

        if not columns:
            return pd.DataFrame(index=df_profiles_norm.index)
        return pd.DataFrame(columns, index=df_profiles_norm.index)


    # ------------------------------------------------------------------
    # Saying what happened
    # ------------------------------------------------------------------

    def _check_no_zero_hours(self, df_demands):
        """
        Alarm about any hour of any node that comes out with no demand.

        Every zero here is a symptom -- a country with no temperature data, a
        `twh/year` that is not a number, a `constant_share` of zero -- and every
        one would otherwise reach GAMS looking exactly like a modelled summer.
        See the zero hours section of docs/dh-demand-timeseries.md.

        The test is against what will actually be *written*: ProcessorRunner
        rounds to `rounding_precision` after this returns, so a node at 0.4 MWh/h
        leaves here non-zero and arrives at GAMS as nothing. NaN counts as zero
        for the same reason, the GDX gate filling it with one.

        Reported at error level and never raised -- one line per node, because a
        broken node has hundreds of thousands of bad hours.
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
            if reason is None and self.node_shares.get(node) == 0:
                reason = (
                    f"its constant_share is 0, so nothing floors the hours when the "
                    f"{self.SMOOTHING_HOURS}-hour mean is at or above "
                    f"{self.BALANCE_POINT_C} C"
                )
            detail = f" -- {reason}" if reason else ""
            self.logger.log_status(
                f"District heating node '{node}' has no demand in {int(empty.sum())} of "
                f"{len(series)} hour(s), first {first}{detail}. Backbone reads a zero as "
                f"'not set', so those hours arrive as no demand at all rather than as a low "
                f"one. A heat network's hot water and losses do not stop; check the node's "
                f"twh/year and constant_share, and whether its country has temperature data.",
                level="error",
            )

    def _report_coverage(self, built_nodes):
        """
        Say once what was built and what was not, instead of leaving it to be noticed.

        Split by whether the user can act on it. A configured country with no
        district heating rows at all is information -- most of a run's countries
        have none. A node someone wrote a demand row for and that could not be
        built is a warning: that row is a request, and the node will otherwise
        sit in the model with a balance penalty and nothing to serve.
        """
        self.logger.log_status(
            f"District heating demand built for {len(built_nodes)} node(s).",
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
                f"No district heating demand for {len(entries)} node(s) -- {detail}. "
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
        # report no spread while every individual node swung by a fifth.
        totals = df_demands[usable].groupby(df_demands.index.year).sum()
        if len(totals) < 2:
            return
        means = totals.mean()
        usable = [n for n in usable if means[n] > 0]
        if not usable:
            return
        ratios = totals[usable] / means[usable]

        self.logger.log_status(
            f"Annual district heating demand matches the workbook's normal-year twh/year as "
            f"a {len(totals)}-year mean; individual climate years range "
            f"{ratios.min().min():.0%} to {ratios.max().max():.0%} of it. "
            f"A single-year build is not meant to reproduce the table figure.",
            level="info",
        )


    # ------------------------------------------------------------------

    def process(self) -> pd.DataFrame:
        """
        Run the full district heating demand pipeline.

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

        self.logger.log_status(
            f"Constructing heat demand profiles from '{self.temperature_file}'..."
        )
        df_temperature = self._read_temperature_grid()
        out_df = self.get_temperature_profile(processed_countries, df_temperature)

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
