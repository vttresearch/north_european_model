"""Constructing a ``BBExcelPipeline`` for method-level tests.

``BBExcelPipeline.__init__`` only reads seven DataFrames off ``source_data`` and
touches no disk until ``run()`` writes the workbook. So its ``create_*`` /
``fill_*`` methods can be exercised directly on hand-made frames, without
building an input folder or an Excel file.

``cache_manager`` is a field of ``BBExcelInputs`` that the pipeline never reads
and does not even store, so it stays None -- which also avoids CacheManager's
mkdir-on-construction and its CWD-relative source hashing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.bb_excel.bb_excel_inputs import BBExcelInputs
from src.bb_excel.bb_excel_pipeline import BBExcelPipeline
from tests._common.fixtures import FakeLogger, make_config

#: The frames BBExcelPipeline.__init__ reads off source_data.
SOURCE_FRAMES = (
    "df_emissiondata",
    "df_nodedata",
    "df_transferdata",
    "df_unitdata",
    "df_demanddata",
    "df_boundarydata",
    "df_userconstraintdata",
)


def make_pipeline(
    *,
    logger: FakeLogger | None = None,
    config: dict | None = None,
    **source_frames: pd.DataFrame,
) -> BBExcelPipeline:
    """A pipeline whose source frames are whatever the test supplies.

    Unnamed frames default to empty, so a test names only what it is about.
    """
    unknown = set(source_frames) - set(SOURCE_FRAMES)
    if unknown:
        raise KeyError(f"unknown source frame(s) {sorted(unknown)}; known: {SOURCE_FRAMES}")

    source = SimpleNamespace(
        **{name: source_frames.get(name, pd.DataFrame()) for name in SOURCE_FRAMES}
    )

    return BBExcelPipeline(
        BBExcelInputs(
            input_folder=Path("."),
            output_folder=Path("."),
            scen_tags=["test", "2030"],
            config=config or make_config(),
            cache_manager=None,
            logger=logger or FakeLogger(),
            source_data=source,
        )
    )


def gnu_frame(*rows: dict) -> pd.DataFrame:
    """A p_gnu_io frame, as the builders pass one around.

    No wrapping needed: the fake MultiIndex is applied in bb_excel_writer on the
    way to the workbook, so every method here takes and returns a plain frame.
    """
    defaults = {"grid": "elec", "node": "FI_elec", "unit": "u1", "input_output": "output"}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def unit_frame(*rows: dict) -> pd.DataFrame:
    """A flat p_unit frame."""
    return pd.DataFrame([{"unit": "u1", **row} for row in rows])
