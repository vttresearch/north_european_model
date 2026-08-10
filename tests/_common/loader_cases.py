"""The table of source-data loader functions that the contract sweeps run over.

Adding a function to the sweeps is **one entry here**.  That is the whole point:
nobody has to remember to write a dtype assertion for a new transform, so the
coverage does not decay as the pipeline grows.

Each case adapts a loader function to a uniform ``(df, logger) -> DataFrame``
shape.  The frame handed in is deliberately wide -- real source frames are --
and functions simply ignore the columns they do not use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

import src.source_data.source_data_loader as loader

#: A wide frame covering the columns the loader functions look for, so that one
#: input shape can drive every case.  Mirrors the real source schemas
#: (unitdata / nodedata / transferdata) rather than inventing new names.
SWEEP_COLUMNS: tuple[str, ...] = (
    "country",
    "grid",
    "node_suffix",
    "from_country",
    "to_country",
    "from_suffix",
    "to_suffix",
    "scenario",
    "year",
    "generator_id",
    "unit_name_prefix",
    "capacity",
    "method",
)

#: Clean reference unittypedata for the two-frame functions.  The nasty value
#: under test always goes into the *primary* frame; nastying both at once would
#: make a failure ambiguous about which side caused it.
UNITTYPEDATA = pd.DataFrame(
    {
        "generator_id": pd.Series(["gen1"], dtype="object"),
        "unittype": pd.Series(["coalplant"], dtype="object"),
        "grid_input1": pd.Series(["coal"], dtype="object"),
        "grid_output1": pd.Series(["elec"], dtype="object"),
        "eff00": pd.Series([0.4], dtype="Float64"),
        "method": pd.Series(["replace"], dtype="object"),
    }
)


@dataclass(frozen=True)
class LoaderCase:
    """One loader function, adapted to ``(df, logger) -> DataFrame``."""

    name: str
    call: Callable[[pd.DataFrame, object], pd.DataFrame]
    #: Set where the function's own docstring states it needs normalized input.
    needs_normalized_input: bool = True
    #: Reason string when this function is *known* to breach the contract.
    #: The sweeps downgrade its failures to xfail so the rest of the signal
    #: stays readable.  Every entry here must also have a precise, strict-xfail
    #: tripwire in tests/unit/source_data/test_known_contract_violations.py,
    #: which fails loudly once the underlying issue is fixed -- otherwise a
    #: fixed bug would silently keep its exemption.
    known_contract_violation: str | None = None

    def __str__(self) -> str:  # pytest id
        return self.name


LOADER_CASES: list[LoaderCase] = [
    LoaderCase(
        "normalize_dataframe",
        lambda df, log: loader.normalize_dataframe(df, "sweep", log),
        needs_normalized_input=False,
    ),
    LoaderCase(
        "drop_underscore_values",
        lambda df, log: loader.drop_underscore_values(df, "sweep", log),
    ),
    LoaderCase(
        "build_node_column",
        lambda df, log: loader.build_node_column(df, log),
    ),
    LoaderCase(
        "build_from_to_columns",
        lambda df, log: loader.build_from_to_columns(df, log),
    ),
    LoaderCase(
        "build_unittype_unit_column",
        lambda df, log: loader.build_unittype_unit_column(df, UNITTYPEDATA, log),
    ),
    LoaderCase(
        "build_unit_grid_and_node_columns",
        lambda df, log: loader.build_unit_grid_and_node_columns(df, UNITTYPEDATA, log),
    ),
    LoaderCase(
        "merge_unittypedata_into_unitdata",
        lambda df, log: loader.merge_unittypedata_into_unitdata(df, UNITTYPEDATA, log),
    ),
    LoaderCase(
        "expand_all_country",
        lambda df, log: loader.expand_all_country(df, ["FI", "SE"]),
    ),
    LoaderCase(
        "apply_whitelist",
        lambda df, log: loader.apply_whitelist(
            df,
            {"scenario": ["test"], "year": [2030], "country": ["FI", "SE"]},
            log,
            "sweep",
        ),
    ),
    LoaderCase(
        "apply_blacklist",
        lambda df, log: loader.apply_blacklist(df, "sweep", {"grid": ["excluded"]}, log),
    ),
    LoaderCase(
        "apply_unit_grids_blacklist",
        lambda df, log: loader.apply_unit_grids_blacklist(df, ["excluded"], "sweep", log),
    ),
    LoaderCase(
        "apply_unit_nodes_blacklist",
        lambda df, log: loader.apply_unit_nodes_blacklist(df, ["excluded"], "sweep", log),
    ),
    LoaderCase(
        # Two frames, so the merge machinery actually runs rather than
        # short-circuiting on a single input.
        "merge_row_by_row",
        lambda df, log: loader.merge_row_by_row(
            [df, df], log, key_columns=["country", "grid"]
        ),
    ),
    LoaderCase(
        "filter_nonzero_numeric_rows",
        lambda df, log: loader.filter_nonzero_numeric_rows(df),
    ),
]
