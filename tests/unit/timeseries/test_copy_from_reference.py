"""Reusing a processor's output instead of re-running it.

A source that does not depend on the scenario -- weather does not care which
scenario year is modelled -- is copied from the first output folder rather than
rebuilt for every one. The copy is only correct if **everything** the processor
would have produced comes across. Its GDX files are the obvious part; its
contributions to the source data tables are not, and they decide what the input
Excel says about a node.

A half-copied processor looks to the next build exactly like one that half
failed, so each piece is asserted separately rather than through one "did it
work" check.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.infrastructure.cache_manager import CacheManager
from src.timeseries.timeseries_inputs import TimeseriesPipelineInputs
from src.timeseries.timeseries_pipeline import TimeseriesPipeline
from tests._common.fixtures import FakeLogger, make_config

HUMAN_NAME = "PV"
PROCESSOR = "VRE_PECD"


def contribution(node: str = "FI00_elec") -> pd.DataFrame:
    return pd.DataFrame([{"grid": "elec", "node": node}])


#: What a finished reference build leaves behind for this spec: one climate
#: window and the forecast branches. Both have to be there before the copy can
#: register them for GAMS.
REFERENCE_GDX = ("ts_cf_PV_1982.gdx", "ts_cf_PV_forecasts.gdx")


def make_reference(tmp_path: Path, *, frames=None, hashes=None, gdx=REFERENCE_GDX) -> Path:
    """A folder shaped like a finished build's output."""
    reference = tmp_path / "reference"
    (reference / "cache").mkdir(parents=True, exist_ok=True)
    for name in gdx:
        (reference / name).write_bytes(b"not really a gdx")
    if frames is not None:
        with open(reference / "cache" / "processor_frames.pkl", "wb") as f:
            pickle.dump(frames, f)
    if hashes is not None:
        (reference / "cache" / "processor_hashes.json").write_text(
            "{" + ", ".join(f'"{k}": "{v}"' for k, v in hashes.items()) + "}"
        )
    return reference


def make_pipeline(
    tmp_path: Path, reference: Path | None, logger: FakeLogger, **config_overrides
) -> TimeseriesPipeline:
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    (tmp_path / "input" / "data_files").mkdir(parents=True, exist_ok=True)
    config = make_config(**config_overrides)

    class Source:
        df_demanddata = pd.DataFrame()

    return TimeseriesPipeline(TimeseriesPipelineInputs(
        config=config,
        input_folder=tmp_path / "input",
        output_folder=output,
        cache_manager=CacheManager(
            input_folder=tmp_path / "input", output_folder=output,
            config=config, logger=logger,
        ),
        source_data_pipeline=Source(),
        logger=logger,
        reference_ts_folder=reference,
    ))


SPEC = {
    "name": PROCESSOR,
    "human_name": HUMAN_NAME,
    "file": "unused",
    "spec": {
        "bb_parameter": "ts_cf",
        "gdx_name_suffix": "PV",
        "bb_parameter_dimensions": ["flow", "node", "f", "t"],
    },
}


@pytest.fixture
def logger():
    return FakeLogger()


class TestACompleteCopy:
    @pytest.fixture
    def copied(self, tmp_path, logger):
        reference = make_reference(
            tmp_path,
            frames={HUMAN_NAME: {"nodedata": contribution()}},
            hashes={PROCESSOR: "the-hash-from-the-reference-run"},
        )
        pipeline = make_pipeline(tmp_path, reference, logger)
        return pipeline, pipeline._copy_processor_from_reference(SPEC)

    def test_the_contributions_come_back(self, copied):
        _, frames = copied
        assert list(frames["nodedata"]["node"]) == ["FI00_elec"]

    def test_and_are_written_into_this_run_s_cache(self, copied):
        """Otherwise the *next* build, which copies nothing, would lose them.

        The reference folder is only consulted for a processor being copied; a
        later partial rerun reads this cache instead.
        """
        pipeline, _ = copied
        stored = pipeline.cache_manager.load_processor_frames()
        assert list(stored[HUMAN_NAME]["nodedata"]["node"]) == ["FI00_elec"]

    def test_the_gdx_files_come_across(self, copied):
        pipeline, _ = copied
        assert (Path(pipeline.output_folder) / "ts_cf_PV_1982.gdx").is_file()

    def test_and_are_registered_for_gams(self, copied):
        pipeline, _ = copied
        inc = (Path(pipeline.output_folder) / "import_timeseries.inc").read_text()
        assert "ts_cf_PV" in inc

    def test_the_hash_says_the_copy_is_current(self, copied):
        """Without it the next build sees a changed processor and re-runs it,
        undoing the saving the copy exists for."""
        pipeline, _ = copied
        assert pipeline.cache_manager.load_processor_hashes()[PROCESSOR] == (
            "the-hash-from-the-reference-run"
        )


class TestSpecsSharingOneProcessor:
    """PV, wind_onshore and wind_offshore are all ``VRE_PECD``.

    The reference cache is keyed by the spec, so copying one must not fetch
    another's answer -- which is what a cache keyed by the processor's name
    would have done.
    """

    def test_each_spec_gets_its_own(self, tmp_path, logger):
        reference = make_reference(tmp_path, frames={
            "PV": {"nodedata": contribution("FI00_elec")},
            "wind_onshore": {"nodedata": contribution("SE00_elec")},
        })
        pipeline = make_pipeline(tmp_path, reference, logger)

        frames = pipeline._copy_processor_from_reference(
            {**SPEC, "human_name": "wind_onshore"}
        )
        assert list(frames["nodedata"]["node"]) == ["SE00_elec"]


class TestWhatIsMissing:
    def test_no_reference_folder_is_a_warning_not_a_crash(self, tmp_path, logger):
        pipeline = make_pipeline(tmp_path, None, logger)

        assert pipeline._copy_processor_from_reference(SPEC) == {}
        logger.assert_logged("Cannot copy", level="warn")

    def test_a_reference_folder_that_does_not_exist_is_too(self, tmp_path, logger):
        pipeline = make_pipeline(tmp_path, tmp_path / "nowhere", logger)

        assert pipeline._copy_processor_from_reference(SPEC) == {}
        logger.assert_logged("does not exist", level="warn")

    def test_no_gdx_to_copy_is_reported_and_actually_stops(self, tmp_path, logger):
        """Regression: the message said "produces no output" and carried on.

        The next thing it did was ask ``update_import_timeseries_inc`` to
        register a file nobody had copied, which raises -- so a reference folder
        where this processor had failed killed the whole build, one line after
        promising the run would continue.
        """
        reference = make_reference(tmp_path, frames={}, gdx=())
        pipeline = make_pipeline(tmp_path, reference, logger)

        assert pipeline._copy_processor_from_reference(SPEC) == {}
        logger.assert_logged("nothing to copy", level="warn")
        assert not (Path(pipeline.output_folder) / "import_timeseries.inc").exists()

    def test_a_processor_that_contributed_nothing_copies_nothing(self, tmp_path, logger):
        """The ordinary case: most processors contribute no frames at all.

        It must not read as a failure, and must not stop the GDX copy.
        """
        reference = make_reference(tmp_path, frames={HUMAN_NAME: {}}, hashes={PROCESSOR: "h"})
        pipeline = make_pipeline(tmp_path, reference, logger)

        assert pipeline._copy_processor_from_reference(SPEC) == {}
        assert (Path(pipeline.output_folder) / "ts_cf_PV_1982.gdx").is_file()
        logger.assert_no_errors()

    def test_a_reference_with_no_cache_still_copies_the_gdx(self, tmp_path, logger):
        # Half a copy is better than none, and the run carries on either way.
        reference = make_reference(tmp_path)
        pipeline = make_pipeline(tmp_path, reference, logger)

        assert pipeline._copy_processor_from_reference(SPEC) == {}
        assert (Path(pipeline.output_folder) / "ts_cf_PV_1982.gdx").is_file()


class TestADeterministicRun:
    """No forecast branches, so there is no _forecasts.gdx to register.

    Regression: the copy decided whether to register one from the spec's
    dimensions alone, while the runner that *writes* it also checks
    ``forecast_quantiles``. An empty one meant the copy asked
    ``update_import_timeseries_inc`` for a file nobody had written, and it raises
    -- so a deterministic multi-scenario build died on its second scenario.
    """

    def test_the_copy_survives_an_empty_forecast_quantiles(self, tmp_path, logger):
        reference = make_reference(tmp_path, gdx=("ts_cf_PV_1982.gdx",))
        pipeline = make_pipeline(tmp_path, reference, logger, forecast_quantiles={})

        pipeline._copy_processor_from_reference(SPEC)

        inc = (Path(pipeline.output_folder) / "import_timeseries.inc").read_text()
        assert "ts_cf_PV_1982" in inc or "climateYear" in inc
        assert "forecasts" not in inc

    def test_forecasts_are_still_registered_when_there_are_branches(self, tmp_path, logger):
        # Negative control: the guard must not disable the ordinary path.
        reference = make_reference(tmp_path)
        pipeline = make_pipeline(tmp_path, reference, logger)

        pipeline._copy_processor_from_reference(SPEC)

        inc = (Path(pipeline.output_folder) / "import_timeseries.inc").read_text()
        assert "forecasts" in inc
