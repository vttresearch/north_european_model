"""Which source files the cache watches for changes.

``CacheManager`` decides what to re-run by hashing a fixed list of source files.
A module missing from that list is invisible: edit it, and the cache reports
"nothing changed" and serves stale output with no error anywhere. That is the
worst failure mode the cache has, because the build still succeeds.

``timeseries_helpers.py`` was missing for exactly this reason -- it holds the
climate-window slicing and the climatological forecast maths, so editing it
changed the model's timeseries while the cache insisted nothing had.

The coverage test below is written to catch the *next* forgotten module rather
than just that one: any new file under ``src/`` must either be watched or be
listed as a deliberate exclusion with a reason.
"""

from pathlib import Path

import pytest

from src.infrastructure.cache_manager import CacheManager

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Modules deliberately not hashed, and why. Adding an entry here is a decision:
#: it asserts that changing this file cannot change the built output.
DELIBERATELY_UNWATCHED = {
    "src/__init__.py": "empty package marker",
    "src/infrastructure/__init__.py": "empty package marker",
    "src/source_data/__init__.py": "empty package marker",
    "src/timeseries/__init__.py": "empty package marker",
    "src/bb_excel/__init__.py": "empty package marker",
    "src/timeseries/processors/__init__.py": "empty package marker",
    "src/infrastructure/logger.py": (
        "presentation only -- changes message formatting and the run's own "
        "error counters, never the data written to inputData.xlsx or GDX"
    ),
}

#: Processors are hashed individually, per processor, by ProcessorRunner --
#: not through the pipeline-level groups.
PROCESSOR_DIR = "src/timeseries/processors/"


def _watched() -> set[str]:
    groups = (
        CacheManager._OVERALL_CODE_FILES,
        CacheManager._SOURCE_PIPELINE_FILES,
        CacheManager._TS_PIPELINE_FILES,
        CacheManager._BB_PIPELINE_FILES,
    )
    return {Path(p).as_posix().lstrip("./") for group in groups for p in group}


class TestTimeseriesHelpersIsWatched:
    def test_timeseries_helpers_is_in_the_timeseries_group(self):
        """Regression for the module that was missing.

        It owns split_timeseries_to_climate_windows and
        calculate_climatological_forecasts -- editing either changes every
        timeseries GDX the build writes.
        """
        assert "src/timeseries/timeseries_helpers.py" in _watched()

    @pytest.mark.parametrize(
        "module",
        [
            "src/timeseries/timeseries_pipeline.py",
            "src/timeseries/timeseries_processor.py",
            "src/timeseries/timeseries_helpers.py",
            "src/GDX_exchange.py",
        ],
    )
    def test_the_timeseries_group_covers_the_whole_timeseries_route(self, module):
        watched = {Path(p).as_posix().lstrip("./") for p in CacheManager._TS_PIPELINE_FILES}
        assert module in watched


class TestNoModuleEscapesUnnoticed:
    def test_every_source_module_is_watched_or_deliberately_excluded(self):
        """The durable version: fails when a *new* module is forgotten.

        Without this, the next helper added under src/ repeats the same silent
        staleness, and nothing says so until someone notices the output is wrong.
        """
        watched = _watched()
        unaccounted = []

        for path in (REPO_ROOT / "src").rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel.startswith(PROCESSOR_DIR):
                continue  # hashed per processor by ProcessorRunner
            if rel in watched or rel in DELIBERATELY_UNWATCHED:
                continue
            unaccounted.append(rel)

        assert not unaccounted, (
            f"source module(s) not hashed by CacheManager: {sorted(unaccounted)}.\n"
            f"Editing these changes the build while the cache reports 'no change', "
            f"so add them to the appropriate _*_FILES group in cache_manager.py -- "
            f"or to DELIBERATELY_UNWATCHED in this test with a reason why changing "
            f"the file cannot change the output."
        )

    def test_every_watched_path_actually_exists(self):
        # compute_file_hash opens the path directly and raises FileNotFoundError,
        # so a renamed module would crash every build rather than degrade.
        missing = [p for p in _watched() if not (REPO_ROOT / p).is_file()]
        assert not missing, f"watched but nonexistent: {sorted(missing)}"

    def test_the_exclusion_list_has_no_stale_entries(self):
        # Keeps the reasons honest: a deleted or now-watched module should not
        # linger here claiming an exemption it no longer needs.
        watched = _watched()
        stale = [
            rel
            for rel in DELIBERATELY_UNWATCHED
            if not (REPO_ROOT / rel).is_file() or rel in watched
        ]
        assert not stale, f"stale DELIBERATELY_UNWATCHED entries: {sorted(stale)}"
