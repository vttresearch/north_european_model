"""
Variants: several versions of the same source data, side by side
=================================================================

A source workbook can hold several versions of the same data -- Finnish district
heating as one national node, as grouped cities, or as every city -- and a config
build picks one version per country. This module is everything the source data
phase needs to know about that, in the order SourceDataPipeline asks:

0. **Declarations.** Each workbook's ``variantdata`` sheet names its variants, their
   options, the default and the check. Declarations merge across workbooks by
   variant name: options are pooled, and the first workbook to declare a default or
   a check wins where two disagree.
1. **Tags.** A row's ``variant`` cell is checked as it is read, before anything
   filters it: ``bio-price:high``, ``dh-level:grouped,cities`` or ``bio-price:all``.
   A malformed tag, or one its own workbook does not declare, drops the row.
2. **Choices.** Every country gets one option per variant from the build: an exact
   country code, then a two-letter key (already expanded by the config reader),
   then ``all``, which reaches only the countries that have rows for that option,
   then the declared default. A country with none of these is *dormant* for the
   variant, and none of the variant's rows are read there. Rows in any table are
   evidence, so every table follows the same choice.
3. **Checks.** ``totals`` and ``coverage`` look at every option of every variant,
   whatever the build chose, so a broken option shows in every build.
4. **Selection.** A table's rows split into the base (untagged rows, and ``:all``
   rows where the country has an option) and the chosen options' rows. The two
   merge in that order, so an option overrides the base whatever the file order.

Mistakes in the data warn and the build survives; what it fell back on is recorded
in ``partial_causes``, and build_input_data.py puts a PARTIAL banner on top of the
build's log. The full rules, and why, are in docs/variants.md.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

import src.source_data.source_data_loader as data_loader
import src.utils as utils

#: The column a data sheet tags its rows in.
VARIANT_COLUMN = "variant"

#: The sheet a workbook declares its variants in.
DECLARATION_SHEET = "variantdata"

#: In a tag, every option of the variant. Reserved: nothing may be named it.
EVERY_OPTION = "all"

#: What a declaration's ``check`` may say.
CHECKS = ("totals", "coverage", "none")

#: Marks the rows of the second merge pass, so merge_row_by_row can tell an option
#: overriding the base from a sheet repeating itself. One of MERGE_META_COLUMNS.
PASS_COLUMN = data_loader.VARIANT_PASS_COLUMN

#: Tables whose rows have no country: they take the build's ``all`` choice, or
#: the default.
COUNTRYLESS_TABLES = ("unittypedata", "emissiondata")

#: How far two options' totals may differ before the check says so.
TOTALS_TOLERANCE = 0.001

_NAME = re.compile(r"[a-z0-9-]+")

_CONNECTIONS = tuple(f"{put}{i}" for put in ("output", "input") for i in range(1, 6))


def parse_tag(text) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """``('bio-price', ('high',))`` from ``'Bio-Price : HIGH'``; None when malformed.

    Case and the spaces around ``:`` and ``,`` do not matter, as everywhere else
    in a source workbook. A tag names one variant and at least one option;
    ``all`` stands alone. Names are letters, digits and '-': no '_', which the
    source tables read as a node-name separator.
    """
    parts = str(text).split(":")
    if len(parts) != 2:
        return None
    variant = parts[0].strip().lower()
    options = [option.strip().lower() for option in parts[1].split(",")]
    if not _NAME.fullmatch(variant) or not all(_NAME.fullmatch(o) for o in options):
        return None
    if EVERY_OPTION in options and len(options) > 1:
        return None
    return variant, tuple(dict.fromkeys(options))


def _canonical(tag: Tuple[str, Tuple[str, ...]]) -> str:
    return f"{tag[0]}:{','.join(tag[1])}"


def _parsed(text: str) -> Tuple[str, Tuple[str, ...]]:
    """A tag already made canonical by check_tags, split back into its parts."""
    variant, options = text.split(":")
    return variant, tuple(options.split(","))


def _where(df: pd.DataFrame, fallback: str) -> str:
    return data_loader._frame_source(df, fallback)


def _workbook_has_sheet(path: Path, prefix: str) -> bool:
    """Whether an .xlsx has a sheet starting with `prefix`, without opening it in pandas.

    A declaration sheet is rare and every listed workbook is asked, so this reads
    only the zip's sheet list. Anything it cannot read is answered "maybe", and
    read_input_excels then decides and reports.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            workbook = archive.read("xl/workbook.xml").decode("utf-8", errors="replace")
    except (OSError, KeyError, zipfile.BadZipFile):
        return True
    names = re.findall(r'<sheet\b[^>]*\bname="([^"]*)"', workbook)
    return any(name.lower().startswith(prefix) for name in names)


@dataclass
class Declaration:
    """One variant, merged from every workbook that declares it."""

    variant: str
    options: List[str] = field(default_factory=list)
    default: Optional[str] = None
    check: str = "none"
    #: Where each setting was first stated, for the disagreement message.
    check_where: str = ""
    default_where: str = ""
    #: option -> (is the default, where first said), to catch a later workbook
    #: calling an option the default that an earlier one did not, or the reverse.
    flags: Dict[str, Tuple[bool, str]] = field(default_factory=dict)


def _all_workbooks(config: Mapping) -> List[str]:
    """Every workbook the config lists, once each, in the config's order."""
    files: List[str] = []
    for key, value in config.items():
        if key.endswith("_files") and isinstance(value, list):
            for name in value:
                if name not in files:
                    files.append(name)
    return files


class Variants:
    """One build's variants: declarations, choices, checks and row selection.

    Created once per SourceDataPipeline.run(). The pipeline hands it every table
    it prepares and asks it which rows to merge; what it decided is left in
    ``choices`` (variant -> country -> option, None where dormant) and
    ``partial_causes``.
    """

    def __init__(self, declarations: Dict[str, Declaration],
                 declared_by_file: Dict[str, Dict[str, set]],
                 build: str, build_choices: Mapping[str, Mapping[str, str]],
                 country_codes: Sequence[str], logger,
                 partial_causes: Optional[List[str]] = None):
        self.declarations = declarations
        self.declared_by_file = declared_by_file
        self.build = build
        self.raw_choices = dict(build_choices or {})
        self.country_codes = list(country_codes)
        self.logger = logger
        self.partial_causes: List[str] = list(partial_causes or [])

        self._code = {str(code).lower(): code for code in self.country_codes}
        #: (variant, option) -> rows naming that option, read anywhere and before
        #: anything was filtered. For "declared but no rows in any workbook".
        self.option_rows: Dict[Tuple[str, str], int] = {}
        #: variant -> option -> countries with rows of that option in this run.
        self.evidence: Dict[str, Dict[str, set]] = {}
        #: Variants with rows in a table that has no country: variant -> tables.
        self.countryless_rows: Dict[str, List[str]] = {}
        #: variant -> the options with rows of their own in those tables.
        self.countryless_options: Dict[str, set] = {}

        self.choices: Dict[str, Dict[str, Optional[str]]] = {}
        self.countryless: Dict[str, Optional[str]] = {}
        self._effective: Optional[Dict[str, Dict[str, str]]] = None
        self._disagreeing_links: Dict[str, Dict[Tuple[str, str], Tuple]] = {}

    # ------------------------------------------------------------------
    # 0. Declarations
    # ------------------------------------------------------------------

    @classmethod
    def from_workbooks(cls, input_folder: Path, config: Mapping, build: str,
                       country_codes: Sequence[str], logger) -> "Variants":
        """Read every listed workbook's variantdata sheet and merge the declarations."""
        build_choices = (config.get("variants") or {}).get(build, {})
        files = [name for name in _all_workbooks(config)
                 if _workbook_has_sheet(Path(input_folder) / name, DECLARATION_SHEET)]
        frames = (data_loader.read_input_excels(input_folder, files, DECLARATION_SHEET, logger)
                  if files else [])

        declarations: Dict[str, Declaration] = {}
        declared_by_file: Dict[str, Dict[str, set]] = {}
        conflicts: Dict[str, List[str]] = {}
        conflict_files: Dict[str, List[str]] = {}

        for df in frames:
            source = _where(df, DECLARATION_SHEET)
            file_name = str(df["_source_file"].iloc[0]) if "_source_file" in df.columns else source
            columns = {str(c).strip().lower(): c for c in df.columns}
            missing = [c for c in ("variant", "option", "check") if c not in columns]
            if missing:
                logger.log_status(
                    f"[{source}] A variantdata sheet needs the columns variant, option, "
                    f"default and check; this one lacks {', '.join(missing)}, so nothing in it "
                    f"is declared.",
                    level="warn",
                )
                continue

            def cell(row, name):
                column = columns.get(name)
                if column is None:
                    return ""
                value = row[column]
                return "" if pd.isna(value) else str(value).strip()

            for index, row in df.iterrows():
                where = f"{source} row {index + 2}"
                variant, option = cell(row, "variant").lower(), cell(row, "option").lower()
                check, default = cell(row, "check").lower(), cell(row, "default").lower()

                if not variant and not option:
                    continue
                if EVERY_OPTION in (variant, option):
                    logger.log_status(
                        f"[{where}] '{EVERY_OPTION}' is reserved: in a tag it means every "
                        f"option, so no variant or option can be named it. The declaration "
                        f"is dropped, and rows tagged with it are undeclared.",
                        level="warn",
                    )
                    continue
                if not _NAME.fullmatch(variant) or not _NAME.fullmatch(option):
                    logger.log_status(
                        f"[{where}] The declaration '{variant}' / '{option}' is malformed: a "
                        f"variant and an option are each one name of letters, digits and "
                        f"'-'. It is dropped, and rows tagged with it are undeclared.",
                        level="warn",
                    )
                    continue
                if check not in CHECKS:
                    logger.log_status(
                        f"[{where}] '{variant}:{option}' has an unknown check '{check}'; write "
                        f"totals, coverage or none. The declaration is dropped, and rows "
                        f"tagged with it are undeclared.",
                        level="warn",
                    )
                    continue
                if default not in ("", "yes", "no"):
                    logger.log_status(
                        f"[{where}] The default of '{variant}:{option}' says '{default}'; "
                        f"write 'yes' for the default option and leave the others empty. It "
                        f"is read as not the default.",
                        level="warn",
                    )
                is_default = default == "yes"

                declared_by_file.setdefault(file_name, {}).setdefault(variant, set()).add(option)
                declaration = declarations.get(variant)
                if declaration is None:
                    declaration = declarations[variant] = Declaration(
                        variant, check=check, check_where=file_name)

                said = conflicts.setdefault(variant, [])
                files_named = conflict_files.setdefault(variant, [])

                def disagree(text, *names):
                    said.append(text)
                    for name in names:
                        if name not in files_named:
                            files_named.append(name)

                if check != declaration.check:
                    disagree(f"check '{declaration.check}' in {declaration.check_where} but "
                             f"'{check}' in {file_name}", declaration.check_where, file_name)
                if option in declaration.flags and declaration.flags[option][0] != is_default:
                    first_default, first_where = declaration.flags[option]
                    disagree(f"'{option}' is {'' if first_default else 'not '}the default in "
                             f"{first_where} but {'' if is_default else 'not '}in {file_name}",
                             first_where, file_name)
                elif is_default and declaration.default not in (None, option):
                    disagree(f"the default is '{declaration.default}' in "
                             f"{declaration.default_where} but '{option}' in {file_name}",
                             declaration.default_where, file_name)
                declaration.flags.setdefault(option, (is_default, file_name))
                if is_default and declaration.default is None:
                    declaration.default, declaration.default_where = option, file_name
                if option not in declaration.options:
                    declaration.options.append(option)

        causes = []
        for variant, said in conflicts.items():
            if not said:
                continue
            files_named = conflict_files[variant]
            logger.log_status(
                f"Variant '{variant}' is declared differently in {' and '.join(files_named)}: "
                f"{'; '.join(dict.fromkeys(said))}. The first declaration is kept. Make the "
                f"variantdata sheets agree.",
                level="warn",
            )
            causes.append(f"{variant}: declared differently in {' and '.join(files_named)}; "
                          f"the first declaration is used")

        return cls(declarations, declared_by_file, build, build_choices,
                   country_codes, logger, partial_causes=causes)

    # ------------------------------------------------------------------
    # 1. Tags, as each sheet is read
    # ------------------------------------------------------------------

    def check_tags(self, dfs: Iterable[pd.DataFrame], table: str) -> List[pd.DataFrame]:
        """Check every frame's variant cells and drop the rows that cannot be read.

        Runs on frames straight from read_input_excels, before anything filters
        them: the form of a row is judged before whether this run needs it. A
        tag left is canonical (``bio-price:high``) in a column named ``variant``.
        """
        return [self._check_frame(df, table) for df in dfs]

    def _check_frame(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        columns = [c for c in df.columns if str(c).strip().lower() == VARIANT_COLUMN]
        if not columns:
            return df
        df = df.rename(columns={columns[0]: VARIANT_COLUMN})
        source = _where(df, table)
        file_name = str(df["_source_file"].iloc[0]) if "_source_file" in df.columns else ""
        declared = self.declared_by_file.get(file_name, {})

        text = df[VARIANT_COLUMN].map(
            lambda v: pd.NA if pd.isna(v) or not str(v).strip() else str(v).strip())
        malformed, undeclared, canonical = [], [], {}
        for index, value in text.items():
            if pd.isna(value):
                continue
            tag = parse_tag(value)
            if tag is None:
                malformed.append((index, value))
                continue
            variant, options = tag
            known = declared.get(variant)
            if known is None or (options != (EVERY_OPTION,) and not set(options) <= known):
                undeclared.append((index, value))
                continue
            canonical[index] = _canonical(tag)
            if options != (EVERY_OPTION,):
                for option in options:
                    key = (variant, option)
                    self.option_rows[key] = self.option_rows.get(key, 0) + 1

        if malformed:
            self.logger.log_status(
                f"[{source}] {len(malformed)} row(s) have a malformed variant tag and are not "
                f"read: {utils.summarise(f'row {i + 2} ({v!r})' for i, v in malformed)}. A tag "
                f"is 'variant:option', 'variant:option1,option2' or 'variant:all', with names "
                f"of letters, digits and '-', and one variant per cell.",
                level="warn",
            )
        if undeclared:
            where = ("its variantdata sheet" if declared
                     else "a variantdata sheet, which this workbook does not have")
            self.logger.log_status(
                f"[{source}] {len(undeclared)} row(s) carry an undeclared variant tag and are "
                f"not read: {utils.summarise(f'row {i + 2} ({v!r})' for i, v in undeclared)}. A "
                f"workbook may use only the variants and options declared in {where}.",
                level="warn",
            )

        dropped = [i for i, _ in malformed] + [i for i, _ in undeclared]
        out = df.drop(index=dropped).copy()
        tags = pd.Series(pd.NA, index=out.index, dtype=object)
        for index, value in canonical.items():
            tags.at[index] = value
        out[VARIANT_COLUMN] = tags
        return out

    # ------------------------------------------------------------------
    # 2. Choices
    # ------------------------------------------------------------------

    def _cause(self, text: str) -> None:
        if text not in self.partial_causes:
            self.partial_causes.append(text)

    def _effective_choices(self) -> Dict[str, Dict[str, str]]:
        """The build's choices with undeclared variants and options taken out, said once."""
        if self._effective is not None:
            return self._effective
        self._effective = {}
        build = f"Build '{self.build}'" if self.build else "The build"
        for variant, choice in self.raw_choices.items():
            declaration = self.declarations.get(variant)
            if declaration is None:
                self.logger.log_status(
                    f"{build} chooses '{variant}', which is not a declared variant in any "
                    f"workbook, so the build ignores it. Check the spelling against the "
                    f"variantdata sheets.",
                    level="warn",
                )
                self._cause(f"{variant}: is not a declared variant; ignored")
                continue
            kept = {}
            unknown: Dict[str, List[str]] = {}
            for key, option in choice.items():
                if option in declaration.options:
                    kept[key] = option
                else:
                    unknown.setdefault(option, []).append(key)
            for option, keys in unknown.items():
                fallback = (f"the default '{declaration.default}'" if declaration.default
                            else "no option, so the variant stays dormant there")
                self.logger.log_status(
                    f"{build}: '{option}' is not a declared option of variant '{variant}' "
                    f"(declared: {', '.join(declaration.options)}), so "
                    f"{', '.join(keys)} take(s) {fallback}.",
                    level="warn",
                )
                self._cause(f"{variant}: '{option}' is not a declared option; "
                            f"{', '.join(keys)} fell back")
            self._effective[variant] = kept
        return self._effective

    def resolve_countryless(self, tables: Mapping[str, Sequence[pd.DataFrame]]) -> None:
        """The option for rows without a country: the build's ``all`` choice, or the default."""
        for table, dfs in tables.items():
            for df in dfs:
                for variant, options in self._tags_in(df):
                    names = self.countryless_rows.setdefault(variant, [])
                    if table not in names:
                        names.append(table)
                    if options != (EVERY_OPTION,):
                        self.countryless_options.setdefault(variant, set()).update(options)

        effective = self._effective_choices()
        build = f"build '{self.build}'" if self.build else "the build"
        for variant, declaration in self.declarations.items():
            choice = effective.get(variant, {})
            option = choice.get("all", declaration.default)
            self.countryless[variant] = option
            per_country = [key for key in choice if key != "all"]
            if per_country and variant in self.countryless_rows:
                taken = f"'{option}'" if option else "nothing (dormant)"
                self.logger.log_status(
                    f"Variant '{variant}' has rows in {', '.join(self.countryless_rows[variant])}, "
                    f"which has no country, but {build} chooses it per country "
                    f"({', '.join(per_country)}). Those rows take {taken}: the 'all' choice or "
                    f"the default. Data without a country belongs in a variant of its own.",
                    level="warn",
                )
                self._cause(f"{variant}: chosen per country, but its "
                            f"{', '.join(self.countryless_rows[variant])} rows have no country")

    def _tags_in(self, df) -> set:
        if df is None or df.empty or VARIANT_COLUMN not in df.columns:
            return set()
        return {_parsed(t) for t in df[VARIANT_COLUMN].dropna().unique()}

    def _country(self, value) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        return self._code.get(str(value).lower())

    def _collect_evidence(self, tables: Mapping[str, Sequence[pd.DataFrame]]) -> None:
        """Which countries have rows of which option, from every prepared country table.

        ``:all`` rows are never evidence: they are in every option, so they say
        nothing about whether a country has the one chosen. A transfer row counts
        for both its ends.
        """
        for table, dfs in tables.items():
            ends = ("from_country", "to_country") if table == "transferdata" else ("country",)
            for df in dfs:
                if df is None or df.empty or VARIANT_COLUMN not in df.columns:
                    continue
                columns = [c for c in ends if c in df.columns]
                if not columns:
                    continue
                pairs = df[[VARIANT_COLUMN, *columns]].dropna(subset=[VARIANT_COLUMN])
                for values in pairs.drop_duplicates().itertuples(index=False):
                    variant, options = _parsed(values[0])
                    if options == (EVERY_OPTION,):
                        continue
                    for country in (self._country(v) for v in values[1:]):
                        if country is None:
                            continue
                        for option in options:
                            (self.evidence.setdefault(variant, {})
                             .setdefault(option, set()).add(country))

    def resolve_countries(self, tables: Mapping[str, Sequence[pd.DataFrame]]) -> None:
        """Each country's option for every declared variant, and what was said about it."""
        self._collect_evidence(tables)
        effective = self._effective_choices()
        build = f"Build '{self.build}'" if self.build else "The build"

        for variant, declaration in self.declarations.items():
            choice = effective.get(variant, {})
            default = declaration.default
            evidence = self.evidence.get(variant, {})
            resolved: Dict[str, Optional[str]] = {}
            without_rows: Dict[str, List[str]] = {}
            every = choice.get("all")
            reached, unreached = [], []

            for country in self.country_codes:
                if country in choice:
                    option = choice[country]
                    if option != default and country not in evidence.get(option, set()):
                        without_rows.setdefault(option, []).append(country)
                        option = default
                elif every is not None:
                    if every == default or country in evidence.get(every, set()):
                        option = every
                        reached.append(country)
                    else:
                        option = default
                        unreached.append(country)
                else:
                    option = default
                resolved[country] = option
            self.choices[variant] = resolved

            for option, countries in without_rows.items():
                fallback = (f"take(s) the default '{default}'" if default
                            else "stay(s) dormant: the variant has no default")
                self.logger.log_status(
                    f"{build}: variant '{variant}' has no rows for option '{option}' in "
                    f"{', '.join(countries)} (':all' rows don't count), so "
                    f"{', '.join(countries)} {fallback}. Check the country, or add the rows.",
                    level="warn",
                )
                self._cause(f"{variant}: {', '.join(countries)} has no rows for option "
                            f"'{option}'")

            if every is not None:
                rest = (f"took the default '{default}'" if default else "stayed dormant")
                reach = (f"{', '.join(reached)} with '{every}'" if reached
                         else f"no country with '{every}'")
                tail = f"; {', '.join(unreached)} {rest}" if unreached else ""
                self.logger.log_status(
                    f"Variant '{variant}': 'all' reached {reach}{tail}. 'all' reaches the "
                    f"countries that have rows for the option.",
                    level="info",
                )
                has_rows = (evidence.get(every)
                            or every in self.countryless_options.get(variant, set()))
                if not reached and every != default and not has_rows:
                    self.logger.log_status(
                        f"{build} chooses '{every}' of variant '{variant}' for all countries, "
                        f"but '{every}' reaches no country: it has no rows in this run.",
                        level="warn",
                    )
                    self._cause(f"{variant}: 'all' chose '{every}', which reaches no country")

        for variant, declaration in self.declarations.items():
            for option in declaration.options:
                if option != declaration.default and not self.option_rows.get((variant, option)):
                    self.logger.log_status(
                        f"Variant '{variant}': option '{option}' is declared but has no rows "
                        f"in any workbook, so a build choosing it gets nothing from it.",
                        level="warn",
                    )

        self._report_choices()

    def _report_choices(self) -> None:
        """One line per variant saying who got what, and one line for the dormant."""
        dormant = []
        for variant, resolved in self.choices.items():
            by_option: Dict[Optional[str], List[str]] = {}
            for country, option in resolved.items():
                by_option.setdefault(option, []).append(country)
            # The largest group last, and by what it leaves out when that is
            # shorter: 'high in FI00; ref in every country but FI00'.
            largest = max(by_option, key=lambda o: len(by_option[o]))
            ordered = [o for o in by_option if o != largest] + [largest]
            parts = []
            for option in ordered:
                countries = by_option[option]
                others = [c for c in resolved if c not in countries]
                if not others:
                    where = "every country"
                elif option == largest and len(others) < len(countries) and len(others) <= 3:
                    where = f"every country but {', '.join(others)}"
                else:
                    where = ", ".join(countries)
                if option is None:
                    dormant.append(f"{variant} in {where}")
                else:
                    parts.append(f"{option} in {where}")
            if parts:
                self.logger.log_status(f"Variant '{variant}': {'; '.join(parts)}.", level="info")
        if dormant:
            self.logger.log_status(
                f"Dormant variants, none of whose rows are read (no default, not chosen): "
                f"{'; '.join(dormant)}.",
                level="info",
            )

    # ------------------------------------------------------------------
    # 3. Checks
    # ------------------------------------------------------------------

    def check(self, tables: Mapping[str, Sequence[pd.DataFrame]],
              key_columns: Mapping[str, Sequence[str]]) -> None:
        """Run every variant's check on every option, silently but for what fails.

        A check sees each option's **check view**: the variant's ``:all`` rows,
        then that option's rows, merged with their methods. Untagged rows never
        enter it, so they are neither judged nor counted as covering anything.
        """
        for variant, declaration in self.declarations.items():
            if not declaration.options:
                continue
            rows = {table: [self._rows_of(df, variant) for df in dfs]
                    for table, dfs in tables.items()}
            rows = {table: [r for r in parts if r is not None]
                    for table, parts in rows.items()}
            if not any(rows.values()):
                continue
            if declaration.check == "none":
                self._report_rows(declaration, rows)
                continue
            views = {
                table: {option: self._view(parts, option, key_columns[table])
                        for option in declaration.options}
                for table, parts in rows.items() if parts
            }
            if declaration.check == "totals":
                self._check_totals(declaration, views)
            else:
                self._check_coverage(declaration, rows, views, key_columns)

    def _report_rows(self, declaration: Declaration, rows) -> None:
        """For a variant nothing checks: how many rows each option has, and from where.

        The only statement the build can make about a `none` variant, so a reader
        can at least see that an option is there and which workbooks supply it.
        """
        counts: Dict[str, int] = {}
        files: Dict[str, List[str]] = {}
        for parts in rows.values():
            for df in parts:
                file_name = (str(df["_source_file"].iloc[0])
                             if "_source_file" in df.columns else "")
                for tag in df[VARIANT_COLUMN]:
                    for option in _parsed(tag)[1]:
                        label = f":{option}" if option == EVERY_OPTION else option
                        counts[label] = counts.get(label, 0) + 1
                        named = files.setdefault(label, [])
                        if file_name and file_name not in named:
                            named.append(file_name)
        order = [f":{EVERY_OPTION}"] + declaration.options
        shown = [f"{label} {counts[label]} ({', '.join(files[label])})"
                 for label in order if label in counts]
        self.logger.log_status(
            f"Variant '{declaration.variant}' (check none): rows per option in this run: "
            f"{'; '.join(shown)}.",
            level="info",
        )

    def _rows_of(self, df, variant):
        """(:all rows, {option: rows}) of `variant` in one frame, or None."""
        if df is None or df.empty or VARIANT_COLUMN not in df.columns:
            return None
        tags = df[VARIANT_COLUMN]
        parsed = {t: _parsed(t) for t in tags.dropna().unique()}
        mine = tags.map(lambda t: not pd.isna(t) and parsed[t][0] == variant)
        if not mine.any():
            return None
        return df[mine]

    def _view(self, parts: Sequence[pd.DataFrame], option: str, keys: Sequence[str]):
        """One option's check view of one table: ':all' rows, then the option's own."""
        every, own = [], []
        for df in parts:
            options = df[VARIANT_COLUMN].map(lambda t: _parsed(t)[1])
            every.append(df[options.map(lambda o: o == (EVERY_OPTION,))])
            own.append(df[options.map(lambda o: option in o)])
        silent = data_loader._SilentLogger()
        frames = [f.drop(columns=[VARIANT_COLUMN]) for f in every + own]
        return {
            "merged": data_loader.merge_row_by_row(frames, silent, key_columns=list(keys)),
            "removed": {_key(row, keys) for f in every + own
                        for _, row in f.iterrows() if row.get("method") == "remove"},
        }

    def _compared_options(self, declaration: Declaration, country: str) -> List[str]:
        """The options worth comparing in one country: the default and those with rows there."""
        evidence = self.evidence.get(declaration.variant, {})
        options = [o for o in declaration.options
                   if o == declaration.default or country in evidence.get(o, set())]
        return options

    def _check_totals(self, declaration: Declaration, views) -> None:
        """Per country and grid: demand, and capacity in and out, equal across options."""
        sums: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        for option in declaration.options:
            demand = views.get("demanddata", {}).get(option)
            if demand is not None:
                for (country, grid), value in _demand_totals(demand["merged"]).items():
                    sums.setdefault((country, grid, "demand"), {})[option] = value
            units = views.get("unitdata", {}).get(option)
            if units is not None:
                for (country, direction, grid), value in _capacity_totals(units["merged"]).items():
                    sums.setdefault((country, grid, f"{direction} capacity"), {})[option] = value

        offenders = []
        for (country, grid, measure), by_option in sums.items():
            code = self._country(country)
            if code is None:
                continue
            compared = self._compared_options(declaration, code)
            if len(compared) < 2:
                continue
            values = {o: by_option.get(o, 0.0) for o in compared}
            low, high = min(values.values()), max(values.values())
            if high - low > TOTALS_TOLERANCE * max(abs(low), abs(high)):
                shown = ", ".join(f"{o} {v:g}" for o, v in values.items())
                offenders.append(f"{code} {grid} {measure}: {shown}")

        if offenders:
            self.logger.log_status(
                f"Variant '{declaration.variant}' (check totals): totals differ between "
                f"options in {len(offenders)} place(s): {utils.summarise(offenders)}. Each "
                f"option should add up to the same demand and capacity per country and grid; "
                f"look for a missing or doubled row.",
                level="warn",
            )
            self._cause(f"{declaration.variant}: totals differ between options")

    def _check_coverage(self, declaration: Declaration, rows, views, key_columns) -> None:
        """Every key the variant touches must be resolved by every option."""
        offenders = []
        for table, parts in rows.items():
            if not parts:
                continue
            keys = list(key_columns[table])
            labels: Dict[tuple, str] = {}
            for df in parts:
                for _, row in df.iterrows():
                    labels.setdefault(_key(row, keys), _label(row, table, keys))
            resolved = {
                option: _keys_of(views[table][option]["merged"], keys)
                | views[table][option]["removed"]
                for option in declaration.options
            }
            for key, label in labels.items():
                missing = [o for o in declaration.options if key not in resolved[o]]
                if missing:
                    offenders.append(f"'{label}' missing from {', '.join(missing)}")

        if offenders:
            self.logger.log_status(
                f"Variant '{declaration.variant}' (check coverage): {len(offenders)} key(s) "
                f"are missing from some options: {utils.summarise(offenders)}. Every option "
                f"must resolve every key the variant touches, from its own rows or the "
                f"':all' rows; untagged rows don't count, so tag a base row the options "
                f"build on as '{declaration.variant}:all'.",
                level="warn",
            )
            self._cause(f"{declaration.variant}: coverage check failed")

    # ------------------------------------------------------------------
    # 4. Selection
    # ------------------------------------------------------------------

    def link_option(self, variant: str, a, b) -> Optional[str]:
        """A link's option: the one both ends chose, else the default, else none."""
        resolved = self.choices.get(variant, {})
        first, second = resolved.get(self._country(a)), resolved.get(self._country(b))
        if first == second:
            return first
        default = self.declarations[variant].default
        if default is None:
            pair = tuple(sorted(str(x) for x in (self._country(a) or a, self._country(b) or b)))
            self._disagreeing_links.setdefault(variant, {})[pair] = (first, second)
        return default

    def select(self, dfs: Sequence[pd.DataFrame], table: str
               ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Split each frame into the base pass and the chosen options' pass.

        Base: untagged rows, and ``:all`` rows wherever the row's country, link
        or table has an option. Options: rows naming the option chosen there.
        Everything else -- other options, dormant countries -- is not read. The
        variant column goes, and option rows are marked for merge_row_by_row.
        """
        base, chosen = [], []
        for df in dfs:
            first, second = self._split(df, table)
            base.append(first)
            chosen.append(second)
        if table == "transferdata":
            self._report_links()
        return base, chosen

    def _holder(self, row, variant: str, table: str) -> Optional[str]:
        if table in COUNTRYLESS_TABLES:
            return self.countryless.get(variant)
        if table == "transferdata":
            return self.link_option(variant, row.get("from_country"), row.get("to_country"))
        return self.choices.get(variant, {}).get(self._country(row.get("country")))

    def _split(self, df: pd.DataFrame, table: str):
        empty = pd.DataFrame()
        if df is None or df.empty or VARIANT_COLUMN not in df.columns:
            return df, empty
        tags = df[VARIANT_COLUMN]
        tagged = tags.notna()
        if not tagged.any():
            return df.drop(columns=[VARIANT_COLUMN]), empty

        in_base = ~tagged
        in_option = pd.Series(False, index=df.index)
        for index, row in df[tagged].iterrows():
            variant, options = _parsed(row[VARIANT_COLUMN])
            option = self._holder(row, variant, table)
            if option is None:
                continue
            if options == (EVERY_OPTION,):
                in_base.at[index] = True
            elif option in options:
                in_option.at[index] = True

        first = df[in_base].drop(columns=[VARIANT_COLUMN])
        second = df[in_option].drop(columns=[VARIANT_COLUMN])
        if not second.empty:
            second = second.assign(**{PASS_COLUMN: 2})
        return first, second

    def _report_links(self) -> None:
        for variant, links in self._disagreeing_links.items():
            shown = [f"{a}-{b} ({x or 'none'} and {y or 'none'})"
                     for (a, b), (x, y) in links.items()]
            self.logger.log_status(
                f"Variant '{variant}' has no default, and on {len(shown)} link(s) the two "
                f"ends choose different options, so those links' '{variant}' rows are not "
                f"read: {utils.summarise(shown)}. Give both ends the same option.",
                level="warn",
            )
            self._cause(f"{variant}: ends choose different options on "
                        f"{utils.summarise(f'{a}-{b}' for a, b in links)}")
        self._disagreeing_links = {}


# ----------------------------------------------------------------------
# Helpers for the checks
# ----------------------------------------------------------------------

def _key(row, keys: Sequence[str]) -> tuple:
    """A row's key as merge_row_by_row folds it: text lower-cased, a gap as None."""
    values = []
    for column in keys:
        value = row.get(column) if hasattr(row, "get") else None
        if value is None or pd.isna(value):
            values.append(None)
        else:
            values.append(value.lower() if isinstance(value, str) else value)
    return tuple(values)


def _keys_of(df: pd.DataFrame, keys: Sequence[str]) -> set:
    if df is None or df.empty:
        return set()
    return {_key(row, keys) for _, row in df.iterrows()}


def _label(row, table: str, keys: Sequence[str]) -> str:
    """What a reader calls the thing a row is about: a node, a unit, a link."""
    def text(column):
        value = row.get(column)
        return None if value is None or pd.isna(value) else str(value)

    if table in ("nodedata", "demanddata") and text("node"):
        return text("node")
    if table == "unitdata" and text("unit"):
        return text("unit")
    if table == "transferdata" and text("from_node") and text("to_node"):
        return f"{text('from_node')}-{text('to_node')}"
    return "/".join(v for v in (text(c) for c in keys) if v)


def _demand_totals(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Annual demand per (country, grid)."""
    if df is None or df.empty or not {"country", "grid", "twh/year"} <= set(df.columns):
        return {}
    values = pd.to_numeric(df["twh/year"], errors="coerce").fillna(0.0)
    grouped = values.groupby([df["country"].astype(str), df["grid"].astype(str).str.lower()]).sum()
    return {key: float(value) for key, value in grouped.items()}


def _capacity_totals(df: pd.DataFrame) -> Dict[Tuple[str, str, str], float]:
    """Capacity per (country, 'output' or 'input', grid), over every connection."""
    if df is None or df.empty or "country" not in df.columns:
        return {}
    totals: Dict[Tuple[str, str, str], float] = {}
    for put in _CONNECTIONS:
        grid_column = f"grid_{put}"
        capacity_column = f"capacity_{put}"
        # normalize_dataframe renames a numeric capacity_output1 to capacity.
        if put == "output1" and capacity_column not in df.columns:
            capacity_column = "capacity"
        if grid_column not in df.columns or capacity_column not in df.columns:
            continue
        direction = "output" if put.startswith("output") else "input"
        values = pd.to_numeric(df[capacity_column], errors="coerce").fillna(0.0)
        frame = pd.DataFrame({"country": df["country"].astype(str),
                              "grid": df[grid_column], "value": values}).dropna(subset=["grid"])
        for (country, grid), value in frame.groupby(["country", "grid"])["value"].sum().items():
            key = (country, direction, str(grid).lower())
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals
