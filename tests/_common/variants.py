"""Helpers for the variant tests.

The behaviour these tests pin is the table in
``tests/expectations/variants.md``; each test names its row (A1, E15 ...).

The source-stage interface they expect:

- ``config_reader.parse_variants(raw, country_codes)`` normalises the config's
  ``variants`` dict -- build name -> variant -> {``'all'`` or an exact country
  code: option} -- with two-letter keys expanded and names lower-cased. A config
  dict built here goes through it, as ``load_config`` would.
- ``SourceDataPipelineInputs(variant=<build name>)`` selects the build.
- After ``run()``, the pipeline exposes ``variant_choices`` (variant -> country
  -> option, ``None`` where the variant is dormant) and ``partial_causes`` (a
  list of strings, empty for a clean build).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from tests._common.fixtures import FakeLogger
from tests._common.routes import config_for_workbooks, run_source

#: The fixture countries of tests/expectations/variants.md. Two Swedish codes so that a
#: two-letter key has something to expand to.
COUNTRIES = ("FI00", "SE01", "SE02", "EE00")

# Stable message substrings (tests/README.md: error-message identity).
MALFORMED = "malformed variant tag"
UNDECLARED = "undeclared variant tag"
DECLARED_DIFFERENTLY = "declared differently"
UNKNOWN_CHECK = "unknown check"
NO_ROWS_ANYWHERE = "no rows in any workbook"
NO_COUNTRY = "has no country"
ENDS_DISAGREE = "ends choose different options"
NO_ROWS_FOR_OPTION = "has no rows for option"
UNDECLARED_OPTION = "is not a declared option"
UNDECLARED_VARIANT = "is not a declared variant"
DORMANT = "dormant"
RESERVED = "reserved"
ALL_REACHED = "'all' reached"
ALL_REACHES_NOTHING = "reaches no country"
TOTALS_DIFFER = "totals differ"
COVERAGE_MISSING = "missing from"
WRITTEN_TWICE = "written twice in the same sheet"
UNMATCHED_ARITHMETIC = "add to or multiply a key that no earlier row established"


def sheet(name: str, header: str, *rows: str) -> str:
    """One ``.wb.txt`` sheet from a header line and row lines."""
    return "\n".join([f"[{name}]", header, *rows]) + "\n"


#: Fuel prices as ObservedTrends.xlsx writes them, plus the variant column.
NODE_HEADER = "Country | Grid | Scenario | Year | price | method | variant"
#: Annual demand per node.
DEMAND_HEADER = "Country | Grid | Node_suffix | Scenario | Year | TWh/year | method | variant"


def declarations(*rows: Sequence[Any]) -> str:
    """A ``[variantdata]`` sheet: one ``(variant, option, default, check)`` per row."""
    lines = ["[variantdata]", "variant | option | default | check | description"]
    for variant, option, default, check in rows:
        lines.append(f"{variant} | {option} | {default} | {check} |")
    return "\n".join(lines) + "\n"


def run_variants(
    tmp_path: Path,
    workbooks: Mapping[str, str],
    *,
    variants: Mapping[str, Any] | None = None,
    build: str = "",
    countries: Sequence[str] = COUNTRIES,
    year: int = 2030,
    logger: FakeLogger | None = None,
    **config_overrides: Any,
):
    """Run the source data phase for one build; returns ``(pipeline, logger)``.

    `variants` is written as a config author would write it. It goes through the
    production normaliser, so two-letter keys and lower-casing behave as in a
    real build. Each call gets its own folder under `tmp_path`.
    """
    from src.infrastructure.config_reader import parse_variants

    raw = {"": {}} if variants is None else variants
    config = config_for_workbooks(
        workbooks,
        country_codes=list(countries),
        variants=parse_variants(raw, list(countries)),
        **config_overrides,
    )
    folder = tmp_path / f"build-{build or 'base'}"
    folder.mkdir(parents=True, exist_ok=True)
    return run_source(folder, workbooks=workbooks, config=config, year=year,
                      variant=build, logger=logger)


def rows_where(df: pd.DataFrame, **keys: Any) -> pd.DataFrame:
    """Rows matching every ``column=value``, compared as text, case-insensitively."""
    if df is None or df.empty:
        return pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    for column, value in keys.items():
        if column not in df.columns:
            return df.iloc[0:0]
        mask &= df[column].astype("string").str.lower() == str(value).lower()
    return df[mask]


def value_at(df: pd.DataFrame, column: str, **keys: Any):
    """The one value of `column` in the row matching `keys`, or None when no row does."""
    found = rows_where(df, **keys)
    assert len(found) <= 1, f"{len(found)} rows match {keys}:\n{found}"
    if found.empty or column not in found.columns:
        return None
    value = found[column].iloc[0]
    return None if pd.isna(value) else value


def price(pipeline, country: str, grid: str = "biomass"):
    """A fuel node's price in the merged nodedata."""
    return value_at(pipeline.df_nodedata, "price", node=f"{country}_{grid}")


def demand(pipeline, node: str):
    """A node's annual demand in the merged demanddata."""
    return value_at(pipeline.df_demanddata, "twh/year", node=node)
