import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.timeseries.processors.base_processor import BaseProcessor, SourceDataError
from src.timeseries.timeseries_helpers import nodes_needing_flow
from src.utils import summarise


#: The 22 underscore-separated fields of a PECD file name, in order.
#:
#: Several names are read off position and context rather than a published
#: schema -- `statistic`, `correction`, `origin` and `scenario` are inferences.
#: That costs little, because nothing below depends on what a field *means*: the
#: checks ask only whether two files agree on it. A wrong name misleads a reader
#: of the build log; a field left out of the comparison would let a genuine
#: difference through unseen, which is worse.
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

#: The subset worth naming in a one-line log entry: the four that differ between
#: a 4.1 and a 4.2 download, plus hub height. The other seventeen are constant
#: across every PECD file this project has seen, and printing them buries the
#: ones that vary.
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

    A record rather than a "folder is consistent" boolean, because making the
    selection follow the scenario year is intended work and will need this
    object. See the open items in docs/vre-timeseries.md.
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
    only in the folder they are pointed at and the `flow` they write.

    A PECD download is roughly ten choices, and **none of them is written into
    the CSV body**: the file name is the only record, and two files from
    different downloads carry the same columns on the same hourly index. That is
    what most of this file is about -- `_describe_file` reads the name and the
    comment block, `_warn_blended_selection` says when a folder holds more than
    one. Whether a capacity factor is *right* is not asked anywhere here; that
    is a question about the scenario, and belongs to input data validation.

    docs/vre-timeseries.md is the documentation and carries the reasoning: what
    a download decides, why the best of several zones wins, and why a zero hour
    is ordinary here where it is an alarm for heat demand.

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

    #: A capacity factor is a fraction of installed capacity, so it cannot leave
    #: [0, 1] whatever the weather did.
    value_range = (0.0, 1.0)
    value_sign = "non_negative"

    #: unitdata answers which nodes have a unit of this flow, and Backbone reads a
    #: capacity factor only through such a unit. Without it this processor built a
    #: series for every configured country and then warned about the ones PECD
    #: could not fill -- AT00 and CH00 offshore on the shipped data, where the
    #: workbook removes the unit and nobody ordered anything. See _needed_nodes.
    requires_source_data = ('unitdata',)

    #: How hard both neighbours must be blowing before a flat hour between them
    #: is called a dropout, as a capacity factor. A modelling choice: the source
    #: rounds to five decimals, so any rule that ignores magnitude fires on
    #: hundreds of ordinary calm hours.
    #:
    #: This used to be a multiple of the written floor, which made it follow the
    #: user's `cutoff_below` -- and at five times a 0.01 cutoff it still reported
    #: a dozen hours every build that nobody ever acted on. Half of nameplate on
    #: both sides is what a dropped value looks like, and unlike a multiple it
    #: stays inside the [0, 1] a capacity factor lives in whatever the cutoff is.
    #: Which hours count as *empty* still follows `cutoff_below`, in
    #: `_written_floor`, because that is what will actually reach GAMS as zero.
    ISOLATED_DROPOUT_NEIGHBOUR = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        required_params = [
            'input_folder',
            'country_codes',
            'start_year',
            'end_year',
            'attached_grid'
        ]

        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        for param in required_params:
            setattr(self, param, kwargs.get(param))

        # The scaling default is 1 here and not only in config_reader: a caller
        # not going through the config arrives with None, and `None != 1` sends
        # it into the scaling branch to multiply by it.
        scaling_factor = kwargs.get('scaling_factor')
        self.scaling_factor = 1 if scaling_factor is None else scaling_factor
        self.custom_column_value = kwargs.get('custom_column_value') or {}

        # ProcessorRunner applies both to the `value` column after this processor
        # returns, so together they decide whether a small capacity factor
        # survives as a number or reaches GAMS as a zero. _report_isolated_dropouts
        # needs them to test what will actually be written.
        try:
            self.rounding_precision = int(kwargs.get('rounding_precision') or 0)
        except (TypeError, ValueError):
            self.rounding_precision = 0
        try:
            cutoff = kwargs.get('cutoff_below')
            self.cutoff_below = None if cutoff is None else float(cutoff)
        except (TypeError, ValueError):
            self.cutoff_below = None

        self.start_date = pd.Timestamp(f"{self.start_year}-01-01")
        self.end_date   = pd.Timestamp(f"{self.end_year}-12-31 23:00")

        #: (code, reason) for every country that needs this flow and got no series.
        self.unbuilt_codes = []

        self.df_unitdata = kwargs.get('df_unitdata')
        self.needed_codes = self._needed_codes()

    def _needed_codes(self):
        """The configured country codes that have a unit of this flow.

        Backbone reads a capacity factor only through a unit, so a series for a
        node with no such unit is inert: building it costs a column in the GDX
        and, worse, makes "PECD has nothing for this code" look like a problem
        when nobody asked for anything. Austria has no offshore wind, and a build
        that says so every run teaches its reader to skip the line that matters.

        ``None`` from `nodes_needing_flow` is "cannot tell" -- no unitdata, no
        `flow` column, or a spec that writes no flow at all -- and every
        configured code is kept, which is what this processor did before. A spec
        with no `attached_grid` cannot name the node it would compare against, so
        it counts as the same answer. An empty set is a different one: the model
        has no unit of this flow, and nothing is built.
        """
        needed = nodes_needing_flow(self.df_unitdata, self.flow)
        if needed is None or not self.attached_grid:
            return list(self.country_codes)
        return [
            code for code in self.country_codes
            if f"{code}_{self.attached_grid}" in needed
        ]

    # ------------------------------------------------------------------
    # Describing a folder before reading any of it
    # ------------------------------------------------------------------

    def _read_header_block(self, path):
        """Read the leading `#` comment block and the column line beneath it.

        PECD writes about fifty comment lines of metadata as `## Key` followed
        by `### Value`, then the real header row. It carries the version, the
        unit and the declared date range.

        `utf-8-sig` rather than the platform default: these files are written
        elsewhere and a byte-order mark would otherwise turn the first `#` into
        an ordinary character, making the whole block look like data.

        Returns
        -------
        tuple[int, dict, tuple]
            The index of the real header row, the `## Key` -> `### Value` pairs,
            and the column names split from the header line. The columns only
            resolve candidates before reading; `read_input_csv` is the authority.
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
                # Only the first value under a key. Deeper levels belong to
                # nested blocks -- a point of contact has a name and a mail
                # address under it -- and are not metadata about the data.
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
        included on the grounds that it might be fine. The name is the only
        record of which download a file came from, so a name this cannot read is
        a file whose provenance is unknown.
        """
        fields, reason = self._parse_filename(path)
        if fields is None:
            self.logger.log_status(
                f"Skipping '{os.path.basename(path)}': it {reason}. The name is the only "
                f"record of which PECD download a file came from, so one that cannot be read "
                f"is a file whose provenance is unknown. Rename it to the PECD convention, "
                f"or move it out of {self.csv_folder}.",
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

        Both describe the same file, so a disagreement means one has been edited
        -- most often a file renamed by hand to fit a folder it does not belong
        in. Reported, not acted on: which to believe is not decidable here.
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
        it, and here that is not the file names but the ten or so choices they
        encode. One line, at info level, every run.
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

        Not a refusal: the files are well-formed, and a folder holding two is
        under-specified rather than wrong. Expect this to become conditional
        once the selection follows the scenario year, which is why it must not
        harden into a refusal now. What it costs to miss is a series concatenated
        from two downloads, with a step at the seam and no other symptom.
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
        # `pecd_version` leads. It is the field a reader needs, and it sits last
        # of the four in positional order -- exactly where truncation drops it.
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
        download in the onshore folder is always a two-line edit away, and its
        values would be written as onshore wind and look entirely ordinary.
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

        The one refusal in this processor, unlike a blended selection: there is
        no reading under which a user meant two files to describe the same hour,
        and the compile loop writes them one after the other, so whichever the
        file system listed last would win, silently and unrepeatably.
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
        """Which columns could serve each country code that needs this flow.

        Exact match first, then the first of the 4-, 3- and 2-letter prefixes
        that matches anything. Depends on column *names* only, so it is settled
        before any values are read and cannot depend on file order.

        Returns
        -------
        dict[str, tuple[str, list[str]]]
            code -> (tier, candidate column names). Codes matching nothing are
            absent, and `_report_coverage` names them.
        """
        resolved = {}
        for code in self.needed_codes:
            if code in columns:
                resolved[code] = ("exact", [code])
                continue
            for length, tier in ((4, "prefix4"), (3, "prefix3"), (2, "prefix2")):
                matched = [c for c in columns if c.startswith(code[:length])]
                if matched:
                    resolved[code] = (tier, matched)
                    break
        return resolved

    def _choose_columns(self, df, resolved):
        """Pick one column per country code: the one with the largest total.

        Where PECD splits a country into zones finer than this model's nodes, the
        node gets the *best* of them rather than an average -- a modelling
        decision, argued in "Which zone a node gets" in docs/vre-timeseries.md, and
        reported below so its size is visible.

        Two things are decided here rather than earlier. The totals are taken
        over the whole configured climate window, so the winner does not change
        when `bb_timeseries_start` moves. And a candidate with no values at all
        is dropped before the comparison: the sum of an all-NaN column is 0.0,
        which would merely lose rather than be excluded, indistinguishable from
        a zone that is genuinely calm.
        """
        mapping = {}
        reports = []

        for code in self.needed_codes:
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

            # An empty candidate is dropped without a word. It changes nothing --
            # the code still gets a column, and which zones PECD ships empty is
            # not the reader's to fix.
            means = {c: float(df[c].mean()) for c in usable}
            chosen = max(usable, key=lambda c: float(df[c].sum()))
            mapping[code] = chosen
            reports.append((code, tier, chosen, means, candidates, usable))

        self._report_zone_choices(reports)
        return mapping

    def _report_zone_choices(self, reports):
        """Say how many nodes were given the best of several zones, and warn about
        a borrowed one.

        Taking the best rather than the average is a modelling assumption, so the
        count is worth a line -- it moves when the download or the country set
        does. Which node took which zone, and by how much, is the table in
        "Which zone a node gets" in docs/vre-timeseries.md, where it stays put
        instead of being retyped into every build log.

        Zones no code can reach by prefix are not reported at all: prefix
        matching is arithmetic, the same zones are unreachable every run, and
        nothing about them asks the reader to do anything.
        """
        chosen_from_several = [r for r in reports if len(r[5]) > 1]
        if chosen_from_several:
            self.logger.log_status(
                f"{len(chosen_from_several)} node(s) take the best of several PECD zones.",
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

        `cutoff_below` is the user's to set, so it is read from the spec rather
        than written into the code: a literal would make the dropout check
        quietly wrong for exactly the person who tuned the parameter. With no
        cutoff set, the floor is whatever `rounding_precision` leaves behind.
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
        capacity factor is an ordinary thing to say. What is not ordinary is one
        empty hour with real generation on both sides of it, which is what a
        dropped value looks like. The magnitude test is what makes this quiet
        enough to be worth reading -- see the zeros section of
        docs/vre-timeseries.md.

        Reports and changes nothing. What the surviving cases actually are is not
        yet known well enough to write a repair rule, and a wrong repair here is
        invisible afterwards.
        """
        if df.empty:
            return

        floor = self._written_floor()
        # max, so that a cutoff above the neighbour level cannot make an hour
        # count as both empty and a neighbour worth reporting.
        threshold = max(self.ISOLATED_DROPOUT_NEIGHBOUR, floor)
        written = self._as_written(df)

        findings = []
        for node in written.columns:
            series = written[node]
            empty = series.isna() | (series.abs() < floor)
            neighbours_high = (
                (series.shift(1) > threshold) & (series.shift(-1) > threshold)
            )
            # `shift` leaves NaN at each end and NaN > threshold is False, so the
            # first and last hour are never flagged -- correct, since neither has
            # two neighbours to be wedged between.
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

        Counted against the codes that *need* this flow rather than every
        configured country: a country with no unit of this technology is not
        missing a series, and never was.

        The zero that matters for a capacity factor is not an hour but a whole
        series: a code that needs the flow and finds no column produces no
        `ts_cf` rows at all, and the unit on that node can never generate for the
        entire run -- downstream, indistinguishable from a unit nobody asked for.
        That is the case this warns about.
        """
        self.logger.log_status(
            f"Capacity factors built for {len(built_codes)} of "
            f"{len(self.needed_codes)} country code(s).",
            level="info",
        )
        if not self.unbuilt_codes:
            return

        # Grouped by reason, which is what a reader acts on and what the codes
        # usually share. Names stay in full even past the list limit: a node with
        # no profile at all is exactly what someone has to go and look up.
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
        per country code that has a unit of this flow.

        Returns
        -------
        pd.DataFrame
            Long format ['flow', 'node', 'time', 'value'], hourly.
        """
        self.csv_folder = self.input_folder
        empty = pd.DataFrame(columns=['flow', 'node', 'time', 'value'])

        nothing_built = (
            f"No '{self.flow or self.attached_grid}' capacity factors will be built, so "
            f"units on those nodes cannot generate."
        )

        # Before the folder is even listed: a run whose units never use this flow
        # has nothing to read the series, and reading a PECD download to build one
        # is minutes spent on a column Backbone will not open.
        if not self.needed_codes:
            self.logger.log_status(
                f"No unit uses the '{self.flow or self.attached_grid}' flow, so nothing "
                f"is built.",
                level="info",
            )
            return empty

        if not os.path.isdir(self.csv_folder):
            self.logger.log_status(
                f"The folder {self.csv_folder} does not exist. {nothing_built} "
                f"Check input_sub_folder, and see the README for where to download the files.",
                level="warn",
            )
            return empty

        csv_files = sorted(glob.glob(os.path.join(self.csv_folder, "*.csv")))
        if not csv_files:
            self.logger.log_status(
                f"No CSV files found in {self.csv_folder}. {nothing_built} "
                f"See the README for where to download them.",
                level="warn",
            )
            return empty

        self.logger.log_status(f"Processing input data in {self.csv_folder}...")

        described = [d for d in (self._describe_file(p) for p in csv_files) if d]
        if not described:
            self.logger.log_status(
                f"None of the {len(csv_files)} file(s) in {self.csv_folder} could be "
                f"identified as PECD data. {nothing_built}",
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
            f"Using {len(selected)} of {len(csv_files)} file(s); the rest fall outside "
            f"{self.start_date:%Y-%m-%d} to {self.end_date:%Y-%m-%d}."
        )
        if not selected:
            self.logger.log_status(
                f"No PECD file in '{self.csv_folder}' covers {self.start_date:%Y-%m-%d} "
                f"to {self.end_date:%Y-%m-%d}. {nothing_built} Check climate_data against "
                f"the years the folder holds.",
                level="warn",
            )
            return empty

        # Raises SourceDataError, which ProcessorRunner turns into "no GDX for
        # this processor". Deliberately before anything is read.
        self._reject_overlapping_windows(selected)

        df_candidates, resolved, _columns = self._read_and_compile_input_CSVs(selected)
        if df_candidates is None:
            return empty

        mapping = self._choose_columns(df_candidates, resolved)
        self._report_coverage(mapping)

        summary_df = pd.DataFrame(index=df_candidates.index)
        for code, column in mapping.items():
            summary_df[code] = df_candidates[column]

        if self.scaling_factor != 1:
            self.logger.log_status(f"Applying logit scaling with factor {self.scaling_factor}...")
            for col in summary_df.columns:
                summary_df[col] = self._apply_logit_scaling(summary_df[col], self.scaling_factor)

        self._report_isolated_dropouts(summary_df)

        # A country code becomes a node only here, by taking the spec's grid.
        summary_df.rename(
            columns={code: f"{code}_{self.attached_grid}" for code in mapping},
            inplace=True,
        )

        # Long format, with the spec's flow written into every row.
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
        Read one PECD CSV and return its `keep` columns on the master index.

        Returns None for a file that cannot be used, having said why.
        """
        # A single unreadable file is skipped with a warning, but malformed
        # numbers are not: these files are machine-generated, so one bad value
        # means the producer changed format and every other file in the folder
        # is suspect too. SourceDataError propagates instead of skipping.
        try:
            df_csv = self.read_input_csv(
                pecd.path, skiprows=pecd.header_row, encoding="utf-8-sig"
            )
        except SourceDataError:
            raise
        except Exception as e:
            self.logger.log_status(
                f"Unable to read '{pecd.name}': {e}. Skipping the file, so the climate "
                f"year(s) it covers have no capacity factors.",
                level="warn",
            )
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
        return df_temp[df_temp.index.isin(master_index)]

    def _read_and_compile_input_CSVs(self, selected):
        """
        Compile the selected PECD files into one frame of candidate columns.

        Every column that could serve a configured country code is carried until
        the end, and the winner is chosen from the assembled whole. Reading the
        mapping off the first file instead would decide it from one climate year,
        and move it whenever the configured window moved.

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
