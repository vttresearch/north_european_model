"""Driving the full route: source workbooks -> ``inputData.xlsx``.

Tier 1 -- the default. ``SourceDataPipeline`` then ``BBExcelPipeline``, with a
dict config and an absolute output folder in ``tmp_path``. No GAMS, no
``CacheManager``, no ``.ini``, and no dependence on the current working
directory.

Three things make that possible:

- ``load_config`` returns a plain dict and every consumer treats it as one, so
  configparser is bypassed entirely and tested separately.
- ``BBExcelInputs`` is a bare dataclass with no runtime validation, and
  ``BBExcelPipeline`` stores ``cache_manager`` without ever reading it. Passing
  None avoids CacheManager's mkdir-on-construction and its CWD-relative source
  hashing in one step.
- Nothing on this path imports ``GDX_exchange``, so no GDX is written and the
  GAMS API is never touched.

Roughly a second per run, dominated by writing the workbook and re-reading the
real ``indexSheet.xlsx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.bb_excel.bb_excel_inputs import BBExcelInputs
from src.bb_excel.bb_excel_pipeline import BBExcelPipeline
from src.source_data.source_data_inputs import SourceDataPipelineInputs
from src.source_data.source_data_pipeline import SourceDataPipeline
from src.timeseries.timeseries_results import TimeseriesPipelineOutput
from tests._common.excel_read import read_output_workbook, read_output_workbook_raw
from tests._common.fixtures import FakeLogger, make_config
from tests._common.workbook_text import sheet_names, write_workbook_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_FILES = REPO_ROOT / "src_files"

#: Config key per source-data category, so a fixture's declared sheets can drive
#: the config instead of every test restating them.
CATEGORY_FILES_KEY = {
    "unittypedata": "unittypedata_files",
    "nodedata": "nodedata_files",
    "emissiondata": "emissiondata_files",
    "demanddata": "demanddata_files",
    "transferdata": "transferdata_files",
    "unitdata": "unitdata_files",
    "userconstraintdata": "userconstraintdata_files",
}


def empty_ts_results() -> TimeseriesPipelineOutput:
    """A valid, empty timeseries result.

    ``BBExcelPipeline.__init__`` filters ``secondary_results`` by key prefix and
    every ``ts_domains`` read is guarded, so empty dicts are safe and mean "no
    timeseries contributed anything".
    """
    return TimeseriesPipelineOutput(secondary_results={}, ts_domains={}, ts_domain_pairs={})


def _category_of(sheet_name: str) -> str | None:
    """Map a declared sheet name to its category by longest matching prefix.

    Longest-first because ``unittypedata`` also starts with ``unit``; the
    pipeline itself matches sheets by prefix, so this mirrors it.
    """
    lowered = sheet_name.lower()
    for category in sorted(CATEGORY_FILES_KEY, key=len, reverse=True):
        if lowered.startswith(category):
            return category
    return None


def config_for_workbooks(workbooks: Mapping[str, str], **overrides: Any) -> dict:
    """Build a config whose ``*_files`` keys are derived from the fixtures' sheets.

    A new data category is then picked up by every route test as soon as one
    fixture declares a sheet for it, with no test edits.
    """
    files: dict[str, list[str]] = {key: [] for key in CATEGORY_FILES_KEY.values()}
    for filename, text in workbooks.items():
        for name in sheet_names(text):
            category = _category_of(name)
            if category is None:
                continue
            key = CATEGORY_FILES_KEY[category]
            if filename not in files[key]:
                files[key].append(filename)
    files.update(overrides)
    return make_config(**files)


def build_input_folder(
    tmp_path: Path,
    *,
    workbooks: Mapping[str, str],
    index_sheet: bool = True,
    gams_files: bool = False,
) -> Path:
    """Create an input folder holding the fixture workbooks as real .xlsx files.

    The genuine ``src_files/indexSheet.xlsx`` is copied rather than faked:
    ``add_index_sheet`` filters it by sheet name, so reusing it exercises the
    real filter instead of a stand-in that cannot disagree with it.
    """
    import shutil

    folder = tmp_path / "input"
    (folder / "data_files").mkdir(parents=True, exist_ok=True)
    for filename, text in workbooks.items():
        write_workbook_text(text, folder / "data_files" / filename, source=filename)

    if index_sheet:
        shutil.copy(SRC_FILES / "indexSheet.xlsx", folder / "indexSheet.xlsx")
    if gams_files:
        shutil.copytree(SRC_FILES / "GAMS_files", folder / "GAMS_files", dirs_exist_ok=True)
    return folder


@dataclass
class RouteResult:
    """Everything a route test needs to assert on."""

    input_folder: Path
    output_folder: Path
    output_file: Path
    config: dict
    source: SourceDataPipeline
    logger: FakeLogger
    sheets: dict[str, pd.DataFrame] = field(default_factory=dict)
    raw_sheets: dict[str, pd.DataFrame] = field(default_factory=dict)


def run_source(
    tmp_path: Path,
    *,
    workbooks: Mapping[str, str],
    config: dict | None = None,
    scenario: str = "test",
    year: int = 2030,
    alternatives: Sequence[str] = (),
    logger: FakeLogger | None = None,
) -> tuple[SourceDataPipeline, FakeLogger]:
    """Read fixture workbooks through ``SourceDataPipeline`` only.

    For assertions about the *source* stage, where ``pd.NA`` and ``0`` are still
    distinct -- the convention changes only once BBExcelPipeline runs.
    """
    logger = logger or FakeLogger()
    config = config or config_for_workbooks(workbooks)
    input_folder = build_input_folder(tmp_path, workbooks=workbooks, index_sheet=False)

    alts = list(alternatives) + [""] * (4 - len(alternatives))
    pipeline = SourceDataPipeline(
        SourceDataPipelineInputs(
            config=config,
            input_folder=input_folder,
            scenario=scenario,
            scenario_year=year,
            country_codes=config["country_codes"],
            logger=logger,
            scenario_alternative=alts[0],
            scenario_alternative2=alts[1],
            scenario_alternative3=alts[2],
            scenario_alternative4=alts[3],
        )
    )
    pipeline.run()
    return pipeline, logger


def run_route(
    tmp_path: Path,
    *,
    workbooks: Mapping[str, str],
    config: dict | None = None,
    scenario: str = "test",
    year: int = 2030,
    alternatives: Sequence[str] = (),
    ts_results: TimeseriesPipelineOutput | None = None,
    logger: FakeLogger | None = None,
) -> RouteResult:
    """Source workbooks -> ``inputData.xlsx``, read back and ready to assert on."""
    logger = logger or FakeLogger()
    config = config or config_for_workbooks(workbooks)

    source, _ = run_source(
        tmp_path,
        workbooks=workbooks,
        config=config,
        scenario=scenario,
        year=year,
        alternatives=alternatives,
        logger=logger,
    )

    input_folder = build_input_folder(tmp_path, workbooks=workbooks)
    output_folder = tmp_path / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    builder = BBExcelPipeline(
        BBExcelInputs(
            input_folder=input_folder,
            output_folder=output_folder,
            scen_tags=[scenario, str(year), *alternatives],
            config=config,
            cache_manager=None,   # stored, never read
            logger=logger,
            source_data=source,
            ts_results=ts_results or empty_ts_results(),
        )
    )
    builder.run()

    output_file = output_folder / "inputData.xlsx"
    result = RouteResult(
        input_folder=input_folder,
        output_folder=output_folder,
        output_file=output_file,
        config=config,
        source=source,
        logger=logger,
    )
    if output_file.is_file():
        result.sheets = read_output_workbook(output_file)
        result.raw_sheets = read_output_workbook_raw(output_file)
    return result
