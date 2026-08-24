import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.timeseries.processors.base_processor import BaseProcessor, SourceDataError
from src.timeseries.timeseries_helpers import summarise


#: The 22 underscore-separated fields of a PECD file name, in order.
#:
#: Several of these names are read off position and context rather than off a
#: published schema -- `statistic`, `correction`, `origin` and `scenario` in
#: particular are inferences. That costs less than it looks like it should,
#: because nothing below depends on knowing what a field *means*: the checks ask
#: only whether two files agree on it. A name that turns out to be wrong
#: misleads a reader of the build log; a field left out of the comparison would
#: let a genuine difference through unseen, which is worse.
FILENAME_FIELDS = (
    "resolution_class", "dataset", "provider", "source_grid",
    "technology", "technology_detail", "product", "spatial_level",
    "window_start", "window_end", "variable", "statistic",
    "temporal_resolution", "technology_variant", "correction", "origin",
    "hub_height", "scenario", "regridding", "physical_model",
    "pecd_version", "file_version",
)

#: The fields that describe *which download* a file came from, as opposed to
#: which period it covers. Two files may differ on the window and nothing else.
SELECTION_FIELDS = tuple(
    field for field in FILENAME_FIELDS
    if field not in ("window_start", "window_end")
)

#: The subset worth naming in a one-line log entry: the four fields that differ
#: between the shipped 4.1 files and a 4.2 download, plus hub height. The other
#: seventeen are constant across every PECD file this project has seen, and
#: printing them buries the ones that vary.
REPORTED_FIELDS = (
    "pecd_version", "physical_model", "technology_variant", "regridding",
    "hub_height",
)

#: PECD technology code -> the `flow` this project writes it as.
TECHNOLOGY_FLOWS = {"WON": "onshore", "WOF": "offshore", "SPV": "PV"}

#: The 12 digits after `S` and `E` in a file name: YYYYMMDDHHMM.
FILENAME_STAMP_FORMAT = "%Y%m%d%H%M"

#: The `Date` column of every PECD CSV read so far. Pinned rather than inferred:
#: `pd.to_datetime` without a format guesses per file, and a user's file that
#: happened to be day-first would be silently reordered.
DATE_COLUMN_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class PecdFile:
    """One PECD CSV, described by everything about it that is not its data.

    Built once per file before any of it is read, so that the folder can be
    checked as a whole -- which download it came from, which hours it claims --
    before a single value is loaded.

    Kept as a record rather than collapsed to a "folder is consistent" boolean
    on purpose. The selection a file came from logically belongs to the scenario
    being modelled (old-fleet profiles for a historical year, future turbines
    for 2040), and making it follow the scenario year is intended work. That
    will need this object; a boolean would have to be written twice.
    """

    path: str
    name: str
    header_row: int
    fields: dict
    metadata: dict
    columns: tuple
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    @property
    def selection(self) -> tuple:
        """The download this file came from, as comparable values."""
        return tuple(self.fields[field] for field in SELECTION_FIELDS)

    def describe_selection(self) -> str:
        """The fields worth naming in a log line, shortest form that stays clear."""
        return (
            f"{self.fields['pecd_version']}, {self.fields['physical_model']}, "
            f"variant {self.fields['technology_variant']}, "
            f"regrid {self.fields['regridding']}, "
            f"hub {self.fields['hub_height']}"
        )


class VRE_PECD(BaseProcessor):
    """
    Capacity factors for wind and solar, from a folder of PECD CSV files.

    One instance serves all three specs -- PV, onshore, offshore -- which differ
    only in the folder they are pointed at and the `flow` they write. See
    `docs/vre-timeseries.md`.

    What this processor cannot know
    -------------------------------
    A PECD download is a set of roughly ten choices -- technology variant,
    turbine class, regridding, physical model -- and **none of them is written
    into the CSV body**. The file name is the only record. Two files from
    different downloads have the same columns on the same hourly index and
    differ only in their values, so a folder holding both produces one series
    silently concatenated from two.

    Hence `_describe_file`, which reads the name and the comment block, and
    `_warn_blended_selection`, which says so when a folder holds more than one.
    It warns rather than refuses: the files are well-formed, and nothing here
    knows which download the scenario wanted -- only that today's arrangement,
    one configured folder per spec, cannot have meant both at once.

    Whether a capacity factor is *right* is not asked anywhere in this file.
    That is a question about the scenario, and it belongs to input data
    validation.

    Which zone a node gets
    ----------------------
    PECD's wind zones are finer than this model's nodes, so a code like `FR00`
    matches no column and has to be resolved by prefix -- and where several
    zones match, the one with the highest total is taken. That is a modelling
    decision, not a lookup: the chosen zone's capacity factor sits well above
    the mean of the zones it beat. It is kept because capacity is built at good
    sites rather than at average ones, and it is *reported* every run so that it
    is a decision someone can see rather than one buried in a prefix.

    Zeros
    -----
    A zero capacity factor is an ordinary statement -- a calm hour, or midnight
    for PV -- so unlike a heat demand node there is no per-hour zero alarm here;
    it would fire on hundreds of correct hours and teach people to ignore
    warnings. Two narrower checks replace it: a *series* that could not be built
    at all (`_report_coverage`), and a single flat hour wedged between two
    ordinary ones (`_report_isolated_dropouts`), which is what a dropped value
    would look like.

    Parameters
    ----------
    input_folder : str
        Folder holding the PECD CSV files, pre-joined by ProcessorRunner.
    country_codes : list of str
        Countries the run includes. The authority on spelling.
    start_year, end_year : int
        First and last climate year.
    attached_grid : str
        Suffix appended to each country code to form the node, e.g. `elec`.
    scaling_factor : float, optional
        Logit-normal shift of the mean, e.g. 0.8 to reduce it by a fifth.
        Default 1, no scaling.
    custom_column_value : dict, optional
        Written into the output columns it names; `flow` is the one used here.
    """

    # A capacity factor is a fraction of installed capacity, so it cannot leave
    # [0, 1] whatever the weather did. Declared once for all three specs (PV,
    # onshore, offshore) that share this class. See BaseProcessor.
    value_range = (0.0, 1.0)
    value_sign = "non_negative"

    #: How far above the written floor both neighbours must sit before a flat
    #: hour between them is called a dropout. A modelling choice: the source
    #: rounds to five decimals, so a genuine calm spell produces long runs of
    #: values a hair above zero, and any rule that ignores magnitude fires on
    #: hundreds of them. The *floor* is not a constant -- it comes from the
    #: user's `cutoff_below`; see `_written_floor`.
    ISOLATED_DROPOUT_MULTIPLIER = 5.0

    def __init__(self, **kwargs):
        # Initialize base class
        super().__init__(**kwargs)

        # List of required parameters
        required_params = [
            'input_folder',
            'country_codes',
            'start_year',
            'end_year',
            'attached_grid'
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack required parameters
        for param in required_params:
            setattr(self, param, kwargs.get(param))

        # Optional parameters. The scaling default is 1 here and not only in
        # config_reader: a processor constructed directly -- by a test, or by
        # any caller not going through the config -- used to arrive with None,
        # and `None != 1` sent it into the scaling branch to multiply by it.
        scaling_factor = kwargs.get('scaling_factor')
        self.scaling_factor = 1 if scaling_factor is None else scaling_factor
        self.custom_column_value = kwargs.get('custom_column_value') or {}

        # ProcessorRunner applies both of these to the `value` column after this
        # processor returns, so together they decide whether a small capacity
        # factor survives as a number or reaches GAMS as a zero.
        # _report_isolated_dropouts needs them to test what will actually be
        # written rather than what this processor happens to hold.
        try:
            self.rounding_precision = int(kwargs.get('rounding_precision') or 0)
        except (TypeError, ValueError):
            self.rounding_precision = 0
        try:
            cutoff = kwargs.get('cutoff_below')
            self.cutoff_below = None if cutoff is None else float(cutoff)
        except (TypeError, ValueError):
            self.cutoff_below = None

        # Derive full-year date boundaries from integer year values
        self.start_date = pd.Timestamp(f"{self.start_year}-01-01")
        self.end_date   = pd.Timestamp(f"{self.end_year}-12-31 23:00")

        #: (code, reason) for every configured country that got no series.
        self.unbuilt_codes = []

    # ------------------------------------------------------------------
    # Describing a folder before reading any of it
    # ------------------------------------------------------------------

    def _read_header_block(self, path):
        """Read the leading `#` comment block and the column line beneath it.

        PECD writes about fifty comment lines of metadata as `## Key` followed
        by `### Value`, then the real header row. The processor used to count
        past this block without looking at it, which is where the version,
        the unit and the declared date range were being thrown away.

        `utf-8-sig` rather than the platform default: these files are written
        elsewhere and a byte-order mark would otherwise turn the first `#` into
        an ordinary character, making the whole block look like data.

        Returns
        -------
        tuple[int, dict, tuple]
            The index of the real header row, the `## Key` -> `### Value` pairs,
            and the column names as split from the header line. The columns are
            used only to resolve candidates before reading; `read_input_csv`
            remains the authority and is checked against them.
        """
        metadata = {}
        key = None
        header_row = 0
        columns = ()

        with open(path, "r", encoding="utf-8-sig") as handle:
            for i, line in enumerate(handle):
                if not line.startswith("#"):
                    header_row = i
                    columns = tuple(part.strip() for part in line.rstrip("\n").split(","))
                    break
                stripped = line.rstrip("\n")
                depth = len(stripped) - len(stripped.lstrip("#"))
                text = stripped[depth:].strip()
                # Only the first value under a key is taken. Deeper levels
                # belong to nested blocks (a point of contact has a name and a
                # mail address under it) and are not metadata about the data.
                if depth == 2:
                    key = text
                elif depth == 3 and key is not None and key not in metadata:
                    metadata[key] = text

        return header_row, metadata, columns

    def _parse_filename(self, path):
        """Split a PECD file name into its named fields.

        Returns (fields, None) or (None, reason). The reason is phrased for a
        log line, because a name that does not parse is the one case where this
        processor cannot say anything at all about what it is being handed.
        """
        stem = os.path.basename(path)
        if stem.lower().endswith(".csv"):
            stem = stem[:-4]

        parts = stem.split("_")
        if len(parts) != len(FILENAME_FIELDS):
            return None, (
                f"has {len(parts)} underscore-separated field(s) where a PECD name "
                f"has {len(FILENAME_FIELDS)}"
            )

        fields = dict(zip(FILENAME_FIELDS, parts))
        start_text, end_text = fields["window_start"], fields["window_end"]
        if not (re.fullmatch(r"S\d{12}", start_text)
                and re.fullmatch(r"E\d{12}", end_text)):
            return None, (
                f"has '{start_text}_{end_text}' in field 9 and 10 where a PECD name "
                f"carries S<YYYYMMDDHHMM>_E<YYYYMMDDHHMM>"
            )

        return fields, None

    def _describe_file(self, path):
        """Everything about one file that is not its data, or None.

        A file that cannot be described is skipped and said so, rather than
        included on the grounds that it might be fine. That fallback is what let
        a file from another dataset into the folder unnoticed -- and since the
        name is the only record of which download a file came from, a name this
        cannot read is a file whose provenance is unknown.
        """
        fields, reason = self._parse_filename(path)
        if fields is None:
            self.logger.log_status(
                f"Skipping '{os.path.basename(path)}': it {reason}. The file name is the "
                f"only record of which PECD download a file came from, so a name that "
                f"cannot be read is a file whose provenance is unknown. Rename it to the "
                f"PECD convention, or move it out of {self.csv_folder}.",
                level="warn",
            )
            return None

        try:
            header_row, metadata, columns = self._read_header_block(path)
        except (OSError, UnicodeDecodeError) as e:
            self.logger.log_status(
                f"Skipping '{os.path.basename(path)}': unable to read its header ({e}).",
                level="warn",
            )
            return None

        try:
            window_start = pd.to_datetime(
                fields["window_start"][1:], format=FILENAME_STAMP_FORMAT
            )
            window_end = pd.to_datetime(
                fields["window_end"][1:], format=FILENAME_STAMP_FORMAT
            )
        except ValueError as e:
            self.logger.log_status(
                f"Skipping '{os.path.basename(path)}': its S/E fields are not dates ({e}).",
                level="warn",
            )
            return None

        described = PecdFile(
            path=path,
            name=os.path.basename(path),
            header_row=header_row,
            fields=fields,
            metadata=metadata,
            columns=tuple(c for c in columns if c),
            window_start=window_start,
            window_end=window_end,
        )
        self._check_header_agrees_with_name(described)
        return described

    def _check_header_agrees_with_name(self, described):
        """Compare the comment block against the file name.

        Both describe the same file, so a disagreement means one of them has
        been edited -- most often a file renamed by hand to fit a folder it does
        not belong in. Reported, not acted on: which of the two to believe is
        not something this can decide.
        """
        unit = described.metadata.get("Unit")
        if unit is not None and unit.replace(" ", "") != "MW/MW":
            self.logger.log_status(
                f"'{described.name}' declares Unit '{unit}' where a PECD capacity factor "
                f"file says MW/MW. Values are used as-is, so if this file is not a ratio "
                f"the capacity factors will be wrong by whatever its unit is.",
                level="warn",
            )

        for label, field, stamp in (
            ("Begin date", "window_start", described.window_start),
            ("End date", "window_end", described.window_end),
        ):
            declared = described.metadata.get(label)
            if declared is None:
                continue
            # PECD writes these as 2013-01-01-0000, which is neither of the two
            # formats already pinned in this module.
            parsed = pd.to_datetime(declared, format="%Y-%m-%d-%H%M", errors="coerce")
            if pd.isna(parsed) or parsed != stamp:
                self.logger.log_status(
                    f"'{described.name}' says {label} {declared!r} in its header but "
                    f"{stamp} in its name. The name is what decides which climate years "
                    f"this file is read for.",
                    level="warn",
                )

    # ------------------------------------------------------------------
    # Checking the folder as a whole
    # ------------------------------------------------------------------

    def _report_selection(self, described):
        """Say once which download the folder holds.

        A build is only reproducible from its log if the log says what went into
        it, and for this processor that is not the file names -- it is the ten
        or so choices those names encode. One line, at info level, every run.
        """
        selections = {}
        for pecd in described:
            selections.setdefault(pecd.selection, []).append(pecd)

        for selection, members in selections.items():
            first = members[0]
            years = sorted({p.window_start.year for p in members})
            span = f"{years[0]}-{years[-1]}" if len(years) > 1 else f"{years[0]}"
            self.logger.log_status(
                f"PECD {self.flow or self.attached_grid}: {first.describe_selection()}"
                f" -- {len(members)} file(s), {span}",
                level="info",
            )

    def _warn_blended_selection(self, described):
        """Say when one folder holds files from more than one download.

        Not a refusal. The files are well-formed, and nothing here knows which
        download the scenario wanted -- the choice belongs to the scenario and
        is currently made by pointing at a folder, so a folder holding two is
        under-specified rather than wrong. Expect this to become conditional
        once the selection follows the scenario year, which is why it must not
        harden into a refusal now.

        What it costs to miss is the point: two downloads carry the same
        columns on the same hourly index, so the series is concatenated with a
        step at the seam and no other symptom.
        """
        selections = {}
        for pecd in described:
            selections.setdefault(pecd.selection, []).append(pecd)
        if len(selections) < 2:
            return

        groups = list(selections.values())
        differing = [
            field for i, field in enumerate(SELECTION_FIELDS)
            if len({group[0].selection[i] for group in groups}) > 1
        ]
        # Named in REPORTED_FIELDS order rather than file-name order, so that
        # `pecd_version` leads. It sits fourth of the four fields that differ
        # between a 4.1 and a 4.2 download, and it is the one a reader needs;
        # left in positional order it is the one the list truncation drops.
        rank = {field: i for i, field in enumerate(REPORTED_FIELDS)}
        differing.sort(key=lambda field: (rank.get(field, len(rank)), field))

        detail = "; ".join(
            f"{'/'.join(g[0].fields[f] for f in differing)} ({len(g)} file(s))"
            for g in groups
        )
        self.logger.log_status(
            f"{len(groups)} different PECD downloads in one folder, differing on "
            f"{summarise(differing)}: {detail}. The series will step where they meet. "
            f"Use one download per folder -- see docs/vre-timeseries.md.",
            level="warn",
        )

    def _warn_technology_mismatch(self, described):
        """Say when the folder's technology is not the one the spec writes.

        The spec's `flow` and the folder are configured separately, so a PV
        download in the onshore folder is a two-line edit away at all times. The
        values would be written out as onshore wind and look entirely ordinary.
        """
        if not self.flow:
            return
        wrong = {}
        for pecd in described:
            code = pecd.fields["technology"]
            expected = TECHNOLOGY_FLOWS.get(code)
            if expected is not None and expected != self.flow:
                wrong.setdefault((code, expected), []).append(pecd.name)

        for (code, expected), names in wrong.items():
            self.logger.log_status(
                f"{len(names)} file(s) are technology '{code}' ({expected}) but this spec "
                f"writes flow '{self.flow}', and will be written as '{self.flow}'. "
                f"Check input_sub_folder and custom_column_value.",
                level="warn",
            )

    def _reject_overlapping_windows(self, selected):
        """Refuse a folder in which two files cover the same hour.

        The one refusal in this processor, and it is a different kind of thing
        from a blended selection. There is no reading under which a user meant
        two files to describe the same hour: the compile loop below writes them
        one after the other, so whichever the file system happened to list last
        would win, silently and unrepeatably. The usual cause is a download
        unpacked into the wrong folder.
        """
        ordered = sorted(selected, key=lambda p: (p.window_start, p.name))
        clashes = []
        for earlier, later in zip(ordered, ordered[1:]):
            if later.window_start <= earlier.window_end:
                clashes.append((earlier, later))

        if not clashes:
            return

        detail = summarise(
            f"'{a.name}' and '{b.name}' "
            f"({max(a.window_start, b.window_start):%Y-%m-%d} to "
            f"{min(a.window_end, b.window_end):%Y-%m-%d})"
            for a, b in clashes
        )
        self._reject(
            f"[{self.csv_folder}] {len(clashes)} pair(s) of files cover the same hours: "
            f"{detail}. Which one wins would be decided by the order the folder is listed "
            f"in. The usual cause is a download unpacked into the wrong folder. "
            f"No GDX output will be written."
        )

    # ------------------------------------------------------------------
    # Choosing a column for each country code
    # ------------------------------------------------------------------

    def _resolve_candidates(self, columns):
        """Which columns could serve each configured country code.

        Exact match first, then the first of the 4-, 3- and 2-letter prefixes
        that matches anything. Resolution depends on column *names* only, so it
        is settled before any values are read and does not depend on which file
        was listed first.

        Returns
        -------
        dict[str, tuple[str, list[str]]]
            code -> (tier, candidate column names). Codes matching nothing are
            absent, and `_report_coverage` names them.
        """
        resolved = {}
        for code in self.country_codes:
            if code in columns:
                resolved[code] = ("exact", [code])
                continue
            for length, tier in ((4, "prefix4"), (3, "prefix3"), (2, "prefix2")):
                matched = [c for c in columns if c.startswith(code[:length])]
                if matched:
                    resolved[code] = (tier, matched)
                    break
        return resolved

    def _choose_columns(self, df, resolved, all_columns):
        """Pick one column per country code: the one with the largest total.

        The policy, stated plainly because the code cannot: where PECD splits a
        country into zones finer than this model's nodes, the node gets the
        *best* of them rather than an average of them. Capacity is built at good
        sites, so the best zone is the closer approximation -- but it is an
        approximation, and the report below says by how much.

        Two things are decided here rather than earlier. The totals are taken
        over the whole configured climate window, so the winner does not change
        when `bb_timeseries_start` moves. And a candidate with no values at all
        is dropped before the comparison, because the sum of an all-NaN column
        is 0.0 and it would otherwise merely lose rather than be excluded --
        indistinguishable from a zone that is genuinely calm.
        """
        mapping = {}
        reports = []
        ignored_empty = []

        for code in self.country_codes:
            if code not in resolved:
                self.unbuilt_codes.append((code, "no matching PECD column"))
                continue

            tier, candidates = resolved[code]
            present = [c for c in candidates if c in df.columns]
            usable = [c for c in present if df[c].notna().any()]

            if not usable:
                self.unbuilt_codes.append((code, (
                    f"its candidate column(s) hold no values" if present else
                    f"none of its candidate columns is in the data"
                )))
                continue

            empty = [c for c in present if c not in usable]
            if empty:
                ignored_empty.append(f"{code} ({summarise(empty)})")

            means = {c: float(df[c].mean()) for c in usable}
            chosen = max(usable, key=lambda c: float(df[c].sum()))
            mapping[code] = chosen
            reports.append((code, tier, chosen, means, candidates, usable))

        if ignored_empty:
            self.logger.log_status(
                f"Empty candidate column(s) ignored, not compared: "
                f"{summarise(ignored_empty)}.",
                level="info",
            )
        self._report_zone_choices(reports, all_columns, resolved)
        return mapping

    def _report_zone_choices(self, reports, all_columns, resolved):
        """Say which zone each node was given, and what that choice was worth.

        The lift over the candidate mean is the number that matters. A code
        resolved to a single column has no choice to report and is left out; a
        code that picked one of eleven is stating a modelling assumption, and
        the size of it belongs in the log rather than in someone's memory.

        Only the largest few are named. The whole table is in
        `docs/vre-timeseries.md`, and a log line carrying ten of them is one
        nobody reads -- the biggest lifts are the ones worth arguing with.
        """
        chosen_from_several = [r for r in reports if len(r[5]) > 1]
        if chosen_from_several:
            def lift(entry):
                _, _, chosen, means, _, _ = entry
                average = sum(means.values()) / len(means)
                return means[chosen] / average - 1 if average > 0 else 0.0

            ranked = sorted(chosen_from_several, key=lift, reverse=True)
            detail = summarise(
                f"{code}->{chosen} {lift(entry):+.0%}"
                for entry in ranked
                for code, _, chosen, _, _, _ in [entry]
            )
            self.logger.log_status(
                f"{len(chosen_from_several)} node(s) take the best of several PECD zones "
                f"(lift over the candidate mean): {detail}.",
                level="info",
            )

        # A prefix is arithmetic rather than a choice: 'FR0' cannot see FR10 to
        # FR15, so six of France's fifteen onshore zones are not candidates for
        # anything. Nothing else in the pipeline can notice that, because the
        # column is simply never looked at.
        #
        # Zones claimed by another configured code are not reported. NOM1 and
        # NON1 both start with 'NO', but they are their own nodes rather than
        # zones NOS0 is missing out on, and listing them would bury the handful
        # of genuinely orphaned columns in noise.
        spoken_for = {c for _, candidates in resolved.values() for c in candidates}
        orphaned = sorted(
            c for c in all_columns
            if c not in spoken_for
            and any(c.startswith(code[:2]) for code in resolved)
        )
        if orphaned:
            # Grouped by country rather than listed: ten column names is a line
            # nobody reads, while "ES 3, FR 6, UK 1" is the same information in
            # a form that fits, and says which countries to go and look at.
            by_country = {}
            for column in orphaned:
                by_country[column[:2]] = by_country.get(column[:2], 0) + 1
            detail = ", ".join(
                f"{country} {count}" for country, count in sorted(by_country.items())
            )
            self.logger.log_status(
                f"{len(orphaned)} PECD zone(s) unreachable by prefix matching, never used: "
                f"{detail}.",
                level="info",
            )

        borrowed = [r for r in reports if r[1] == "prefix2"]
        if borrowed:
            detail = summarise(f"{code}->{chosen}" for code, _, chosen, _, _, _ in borrowed)
            self.logger.log_status(
                f"{len(borrowed)} node(s) matched a PECD column on their first two letters "
                f"only: {detail}. That can be a different zone of the same country.",
                level="warn",
            )

    # ------------------------------------------------------------------
    # Saying what happened
    # ------------------------------------------------------------------

    def _written_floor(self):
        """The smallest magnitude that survives to GAMS as a number.

        `cutoff_below` is the user's to set, and 0.05 is as legitimate as the
        0.01 shipped today, so this is read from the spec rather than written
        into the code -- a literal would make the dropout check quietly wrong
        for exactly the person who tuned the parameter. With no cutoff set, the
        floor is whatever `rounding_precision` leaves behind.
        """
        if self.cutoff_below is not None and self.cutoff_below > 0:
            return float(self.cutoff_below)
        return 0.5 * (10.0 ** -self.rounding_precision)

    def _as_written(self, df):
        """The frame as ProcessorRunner will write it: rounded, then cut off.

        Both steps happen after this processor returns, and both can turn a
        small number into a zero. A check run against the unrounded values would
        be answering a question nobody asked.
        """
        written = df.round(self.rounding_precision)
        cutoff = self.cutoff_below
        if cutoff is not None:
            written = written.where(written.isna() | (written.abs() >= cutoff), 0.0)
        return written

    def _report_isolated_dropouts(self, df):
        """Report a single flat hour sitting between two ordinary ones.

        There is no per-hour zero alarm in this processor, because a zero
        capacity factor is an ordinary thing to say -- midnight for PV, a calm
        hour for wind. What is not ordinary is one empty hour with real
        generation on both sides of it, which is what a dropped value looks
        like.

        The magnitude test is what makes this quiet enough to be worth reading.
        The source rounds to five decimals, so a real calm spell produces long
        runs of values a hair above zero and any rule that ignores magnitude
        fires on hundreds of them. Requiring both neighbours to sit well clear
        of the floor leaves only the cases where something really was lost.

        Reports and changes nothing. What the surviving cases actually are is
        not yet known well enough to write a repair rule, and a wrong repair
        here is invisible afterwards.
        """
        if df.empty:
            return

        floor = self._written_floor()
        threshold = self.ISOLATED_DROPOUT_MULTIPLIER * floor
        written = self._as_written(df)

        findings = []
        for node in written.columns:
            series = written[node]
            empty = series.isna() | (series.abs() < floor)
            neighbours_high = (
                (series.shift(1) > threshold) & (series.shift(-1) > threshold)
            )
            # `shift` leaves NaN at each end, and NaN > threshold is False, so
            # the first and last hour are never flagged -- correct, since
            # neither has two neighbours to be wedged between.
            flagged = empty & neighbours_high
            if flagged.any():
                findings.append((node, int(flagged.sum()), series.index[flagged][0]))

        if not findings:
            return

        # Earliest across every node, not the earliest of whichever node has the
        # most: the list below is ordered by count so the worst survives the
        # truncation, and reading "first" off it would name a later hour.
        earliest = min(first for _, _, first in findings)
        findings.sort(key=lambda finding: finding[1], reverse=True)
        detail = summarise(f"{node} {count}" for node, count, _ in findings)
        total = sum(count for _, count, _ in findings)
        self.logger.log_status(
            f"{total} isolated empty hour(s) in {len(findings)} node(s), each between two "
            f"hours above {threshold:g}: {detail}. Earliest {earliest}. Values unchanged.",
            level="warn",
        )

    def _report_coverage(self, built_codes):
        """Say once what was built and what was not.

        The zero that matters for a capacity factor is not an hour, it is a
        whole series: a code that finds no column produces no `ts_cf` rows at
        all, and the unit attached to that node can never generate for the
        entire run. That is indistinguishable, downstream, from a unit nobody
        asked for.
        """
        self.logger.log_status(
            f"Capacity factors built for {len(built_codes)} of "
            f"{len(self.country_codes)} country code(s).",
            level="info",
        )
        if not self.unbuilt_codes:
            return

        # Grouped by reason, because the reason is what a reader acts on and
        # every code usually shares one. The codes are named in full here even
        # past the list limit: this is the actionable line, and a node with no
        # profile at all is exactly what someone has to go and look up.
        by_reason = {}
        for code, reason in sorted(self.unbuilt_codes):
            by_reason.setdefault(reason, []).append(code)
        detail = "; ".join(
            f"{', '.join(codes)}: {reason}" for reason, codes in sorted(by_reason.items())
        )
        self.logger.log_status(
            f"No '{self.flow or self.attached_grid}' capacity factors for "
            f"{len(self.unbuilt_codes)} code(s) -- {detail}. Units on those nodes "
            f"cannot generate.",
            level="warn",
        )

    # ------------------------------------------------------------------

    @property
    def flow(self) -> str:
        """The `flow` this spec writes, e.g. 'onshore'."""
        return self.custom_column_value.get('flow', '')

    def process(self) -> pd.DataFrame:
        """
        Read a folder of PECD CSV files into one hourly capacity factor series
        per configured country code.

        Returns
        -------
        pd.DataFrame
            Long format ['flow', 'node', 'time', 'value'], hourly.
        """
        self.csv_folder = self.input_folder
        empty = pd.DataFrame(columns=['flow', 'node', 'time', 'value'])

        if not os.path.isdir(self.csv_folder):
            self.logger.log_status(
                f"The folder {self.csv_folder} does not exist.", level="warn"
            )
            return empty

        csv_files = sorted(glob.glob(os.path.join(self.csv_folder, "*.csv")))
        if not csv_files:
            self.logger.log_status(
                f"No CSV files found in {self.csv_folder}.", level="warn"
            )
            return empty

        self.logger.log_status(f"Processing input data in {self.csv_folder}...")

        described = [d for d in (self._describe_file(p) for p in csv_files) if d]
        if not described:
            self.logger.log_status(
                f"None of the {len(csv_files)} file(s) in {self.csv_folder} could be "
                f"identified as PECD data.",
                level="warn",
            )
            return empty

        self._report_selection(described)
        self._warn_technology_mismatch(described)
        self._warn_blended_selection(described)

        selected = [
            d for d in described
            if not (d.window_end < self.start_date or d.window_start > self.end_date)
        ]
        self.logger.log_status(
            f"Using {len(selected)} files within date range from the found "
            f"{len(csv_files)} files..."
        )
        if not selected:
            self.logger.log_status(
                f"No PECD file in '{self.csv_folder}' covers {self.start_date:%Y-%m-%d} "
                f"to {self.end_date:%Y-%m-%d}.",
                level="warn",
            )
            return empty

        # Raises SourceDataError, which ProcessorRunner turns into "no GDX for
        # this processor". Deliberately before anything is read.
        self._reject_overlapping_windows(selected)

        df_candidates, resolved, columns = self._read_and_compile_input_CSVs(selected)
        if df_candidates is None:
            return empty

        mapping = self._choose_columns(df_candidates, resolved, columns)
        self._report_coverage(mapping)

        summary_df = pd.DataFrame(index=df_candidates.index)
        for code, column in mapping.items():
            summary_df[code] = df_candidates[column]

        # Apply logit-normal scaling if scaling_factor differs from 1
        if self.scaling_factor != 1:
            self.logger.log_status(f"Applying logit scaling with factor {self.scaling_factor}...")
            for col in summary_df.columns:
                summary_df[col] = self._apply_logit_scaling(summary_df[col], self.scaling_factor)

        self._report_isolated_dropouts(summary_df)

        # Rename country columns to indicate the attached grid
        summary_df.rename(
            columns={code: f"{code}_{self.attached_grid}" for code in mapping},
            inplace=True,
        )

        # Secondary result is None for this processor
        self.secondary_result = None

        self.logger.log_status("Time series built.", level="info")

        # Convert to long format: [flow, node, time, value]
        result = summary_df.reset_index(names='time')
        result = result.melt(id_vars=['time'], var_name='node', value_name='value')
        result['flow'] = self.flow
        return result[['flow', 'node', 'time', 'value']]

    def _apply_logit_scaling(self, series, target_scaling, epsilon=1e-6):
        """
        Adjusts capacity factor using Logit-Normal transformation.
        Shifts the mean of a [0,1] bounded series by target_scaling multiplier
        while preserving the overall shape and bounds.
        """
        original_mean = series.mean()
        target_mean = original_mean * target_scaling

        if original_mean == 0:
            return series

        # Map to latent space (logit)
        clipped = np.clip(series.values, epsilon, 1 - epsilon)
        y = np.log(clipped / (1 - clipped))

        # Binary search for the offset
        low, high = -15.0, 15.0
        for _ in range(20):
            mid = (low + high) / 2
            transformed = 1 / (1 + np.exp(-(y + mid)))
            if transformed.mean() < target_mean:
                low = mid
            else:
                high = mid

        final_values = 1 / (1 + np.exp(-(y + mid)))

        # Preserve hard 0s and 1s
        final_values[series == 0] = 0
        final_values[series == 1] = 1

        return pd.Series(final_values, index=series.index)

    def _read_and_process_csv(self, pecd, keep, master_index):
        """
        Read one PECD CSV and return its candidate columns on the master index.

        Parameters
        ----------
        pecd : PecdFile
            The described file; its `header_row` is already known.
        keep : list of str
            The candidate columns to extract.
        master_index : pd.Index
            The complete date range index.

        Returns
        -------
        pd.DataFrame or None
        """
        # A single unreadable file is skipped with a warning, but malformed
        # numbers are not: these files are machine-generated, so one bad value
        # means the producer changed format and every other file in the folder is
        # suspect too. SourceDataError therefore propagates instead of skipping.
        try:
            df_csv = self.read_input_csv(
                pecd.path, skiprows=pecd.header_row, encoding="utf-8-sig"
            )
        except SourceDataError:
            raise
        except Exception as e:
            self.logger.log_status(f"Error reading file {pecd.path}: {e}", level="warn")
            return None

        if 'Date' not in df_csv.columns:
            self.logger.log_status(
                f"File {pecd.path} does not have a 'Date' column. Skipping the file.",
                level="warn",
            )
            return None

        stamps = pd.to_datetime(
            df_csv['Date'], format=DATE_COLUMN_FORMAT, errors='coerce'
        )
        unparsed = stamps.isna()
        if unparsed.all():
            self.logger.log_status(
                f"File {pecd.path} has no 'Date' value in the format "
                f"{DATE_COLUMN_FORMAT} (first is {df_csv['Date'].iloc[0]!r}). "
                f"Skipping the file.",
                level="warn",
            )
            return None
        if unparsed.any():
            self.logger.log_status(
                f"{pecd.name}: {int(unparsed.sum())} row(s) carry a 'Date' that is not "
                f"{DATE_COLUMN_FORMAT}, for example "
                f"{df_csv.loc[unparsed, 'Date'].iloc[0]!r}. Those rows are dropped and "
                f"the hours they should have carried count as missing data.",
                level="warn",
            )
            df_csv = df_csv.loc[~unparsed]
            stamps = stamps.loc[~unparsed]

        df_csv = df_csv.drop(columns=['Date'])
        df_csv.index = pd.DatetimeIndex(stamps.to_numpy())

        present = [c for c in keep if c in df_csv.columns]
        df_temp = df_csv[present]

        # Filter rows to only include indices from the master date range
        return df_temp[df_temp.index.isin(master_index)]

    def _read_and_compile_input_CSVs(self, selected):
        """
        Compile the selected PECD files into one frame of candidate columns.

        Every column that could serve a configured country code is carried until
        the end, and the winner is chosen from the assembled whole. That is the
        difference from reading the mapping off the first file: the first file
        is one climate year, and a year is not a statement about which zone a
        country's capacity sits in. The old behaviour made the choice move when
        the configured window moved, with nothing said about it.

        Returns
        -------
        tuple[pd.DataFrame | None, dict, list]
            The candidate columns on the full hourly index, the resolution that
            produced them, and every data column the files carry -- the last so
            that the report can name the zones no node can reach.
        """
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='60min')

        # Columns come from the header line read while describing each file, so
        # resolution costs no extra I/O and does not depend on file order.
        column_sets = {pecd.columns for pecd in selected}
        if len(column_sets) > 1:
            counts = sorted({len(cols) for cols in column_sets})
            self.logger.log_status(
                f"The {len(selected)} PECD file(s) in {self.csv_folder} do not all carry "
                f"the same columns ({counts} of them). Only columns common to every file "
                f"are used; a zone present in some years and not others would otherwise "
                f"be built from part of the window.",
                level="warn",
            )

        common = set.intersection(*(set(cols) for cols in column_sets))
        columns = [c for c in selected[0].columns if c in common and c != 'Date']

        resolved = self._resolve_candidates(columns)
        keep = sorted({c for _, candidates in resolved.values() for c in candidates})
        if not keep:
            self.logger.log_status(
                f"No column in '{self.csv_folder}' matches any configured country code. "
                f"Check the CSV column headers against country_codes.",
                level="warn",
            )
            return None, resolved, columns

        df_csv_summary = pd.DataFrame(np.nan, index=date_range, columns=keep)

        for pecd in selected:
            df_temp = self._read_and_process_csv(pecd, keep, date_range)
            if df_temp is None or df_temp.empty:
                continue
            df_csv_summary.loc[df_temp.index, df_temp.columns] = df_temp

        return df_csv_summary, resolved, columns
