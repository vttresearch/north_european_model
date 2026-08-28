"""What survives a partial rerun, and what the cache is allowed to hold.

On a partial rerun only some specs execute, and the workbook still has to
describe the whole model. Every contribution the timeseries phase makes has to
come back -- from the specs that ran this time and from the ones that did not.

This replaces a regression test about merging sets of tuples into a shared JSON
file. That merge silently lost every non-running processor's pairs, because it
matched a pair collection by its container and the JSON round trip did not
preserve the container. The bug class is designed out rather than re-tested: one
entry per spec, replaced whole, cannot drop a key nothing wrote to.

The other half is a rule about *what* goes in. The cache holds what a processor
returned and never a merged or melted table, so that the tables the builder reads
are recomputed from workbooks plus this file on every build. A cache of
half-merged frames could not survive a workbook edit, and there would be no way
to tell that it had not.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.infrastructure.cache_manager import CacheManager
from src.source_data.source_data_contributions import combine_contributions
from src.timeseries.timeseries_inputs import TimeseriesPipelineInputs
from src.timeseries.timeseries_pipeline import TimeseriesPipeline
from tests._common.fixtures import FakeLogger, make_config


def make_manager(tmp_path: Path, specs: dict | None = None) -> CacheManager:
    """A manager over `tmp_path`.

    `specs` names the entries of ``timeseries_specs``; the values are never read
    by anything under test. The default is the fixture's empty dict, which
    `load_processor_frames` reads as "no config to prune against" and so leaves
    every cached spec alone -- which is what lets the tests below save under
    whatever name reads clearly.
    """
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    (tmp_path / "input" / "data_files").mkdir(parents=True, exist_ok=True)
    return CacheManager(
        input_folder=tmp_path / "input",
        output_folder=output,
        config=make_config(**({"timeseries_specs": specs} if specs else {})),
        logger=FakeLogger(),
    )


def boundary(node: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "grid": "reservoir",
        "node": node,
        "param_gnboundarytypes": "upwardLimit",
        "usetimeseries": 1,
    }])


class TestEverySpecSurvivesAPartialRerun:
    def test_a_second_spec_does_not_displace_the_first(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.save_processor_frames("hydro storage limits", {"boundarydata": boundary("FI")})
        manager.save_processor_frames("other source", {"boundarydata": boundary("SE")})

        assert set(manager.load_processor_frames()) == {"hydro storage limits", "other source"}

    def test_one_spec_rerunning_alone_keeps_every_other(self, tmp_path):
        """The build-shaped statement of the same thing.

        Three specs contribute on a full run; one of them re-runs on its own
        afterwards. What the phase returns must still describe the whole model.
        """
        manager = make_manager(tmp_path)
        for name, node in (("a", "FI"), ("b", "SE"), ("c", "NO")):
            manager.save_processor_frames(name, {"boundarydata": boundary(node)})

        # A later build in which only 'b' re-runs.
        manager.save_processor_frames("b", {"boundarydata": boundary("SE")})

        combined = combine_contributions(list(manager.load_processor_frames().values()))
        assert set(combined["boundarydata"]["node"]) == {"FI", "SE", "NO"}

    def test_a_rerun_replaces_that_spec_rather_than_adding_to_it(self, tmp_path):
        """A spec that now says less must not keep saying the old thing.

        Accumulating would make a node impossible to retire: dropping it from the
        source data would leave the cached claim standing.
        """
        manager = make_manager(tmp_path)
        manager.save_processor_frames("a", {"boundarydata": boundary("FI")})
        manager.save_processor_frames("a", {"boundarydata": boundary("SE")})

        combined = combine_contributions(list(manager.load_processor_frames().values()))
        assert set(combined["boundarydata"]["node"]) == {"SE"}

    def test_a_spec_that_now_contributes_nothing_leaves_nothing_behind(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.save_processor_frames("a", {"boundarydata": boundary("FI")})
        manager.save_processor_frames("a", {})

        assert combine_contributions(list(manager.load_processor_frames().values())) == {}

    def test_an_empty_cache_reads_as_nothing_rather_than_failing(self, tmp_path):
        assert make_manager(tmp_path).load_processor_frames() == {}


class TestThreeSpecsSharingOneProcessor:
    """The naming bug the previous scheme had.

    ``VRE_PECD`` is named by three ``timeseries_specs`` entries -- PV, onshore
    and offshore -- which are three different runs producing three different
    outputs. The old per-processor cache files were named after the processor, so
    the three overwrote each other and only the last one's answer survived. The
    spec's own key is unique by construction, which is why it is the key here.
    """

    def test_specs_sharing_a_processor_keep_separate_entries(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.save_processor_frames("PV", {"boundarydata": boundary("FI")})
        manager.save_processor_frames("wind_onshore", {"boundarydata": boundary("SE")})
        manager.save_processor_frames("wind_offshore", {"boundarydata": boundary("NO")})

        combined = combine_contributions(list(manager.load_processor_frames().values()))
        assert set(combined["boundarydata"]["node"]) == {"FI", "SE", "NO"}


class TestASpecThatDidNotRunStillContributes:
    """The whole point of the store, exercised through the pipeline.

    On a partial rerun most specs do not execute -- and in the shipped configs
    only one of them contributes anything at all, so if the pipeline returned
    just what ran, the workbook would lose every seasonal hydro boundary the
    moment anything *else* changed.

    Confirmed against a real build as well: forgetting `elec_demand_TYNDP2024`'s
    hash and rebuilding OT2030 re-runs that processor alone, and all 24
    `useTimeseries` rows are still in `inputData.xlsx` afterwards.
    """

    def test_the_pipeline_returns_what_the_cache_holds(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.save_processor_frames("hydro storage limits", {"boundarydata": boundary("FI")})

        pipeline = TimeseriesPipeline(TimeseriesPipelineInputs(
            config=make_config(),          # no timeseries_specs: nothing runs
            input_folder=tmp_path / "input",
            output_folder=tmp_path / "output",
            cache_manager=manager,
            source_data_pipeline=SimpleNamespace(df_demanddata=pd.DataFrame()),
            logger=FakeLogger(),
        ))

        contributed = pipeline.run()
        assert list(contributed["boundarydata"]["node"]) == ["FI"]


class TestOnlyRawContributionsAreStored:
    def test_what_goes_in_comes_back_unchanged(self, tmp_path):
        """No normalisation, no merging, no melting on the way through.

        Everything derived is recomputed from workbooks plus this file, so what
        it holds has to be the processor's own answer and nothing else.
        """
        manager = make_manager(tmp_path)
        contributed = boundary("FI")
        manager.save_processor_frames("a", {"boundarydata": contributed})

        restored = manager.load_processor_frames()["a"]["boundarydata"]
        pd.testing.assert_frame_equal(restored, contributed)

    def test_the_cache_holds_no_source_data_tables(self, tmp_path):
        """A merged df_nodedata in here would be unreadable after a workbook edit.

        Only names a processor may contribute to appear as keys, and the frames
        under them are per-spec, so nothing in the file can be mistaken for the
        merged table the builder reads.
        """
        manager = make_manager(tmp_path)
        manager.save_processor_frames("a", {"nodedata": pd.DataFrame([
            {"grid": "elec", "node": "FI_elec", "influx": -5.0}
        ])})

        stored = manager.load_processor_frames()
        assert list(stored) == ["a"]
        assert list(stored["a"]["nodedata"].columns) == ["grid", "node", "influx"]


class TestASpecThatLeftTheConfig:
    """Nothing else prunes this file.

    Removing or renaming a `timeseries_specs` entry is not one of the full-rerun
    causes in ``CacheManager.run`` -- it checks topology, climate, forecasts, code
    hashes and the previous run's failures, not the spec set -- so an entry that
    is not dropped here is an entry that reaches the input excel forever.
    """

    def test_its_contribution_is_dropped(self, tmp_path):
        manager = make_manager(tmp_path, specs={"kept": {}, "removed": {}})
        manager.save_processor_frames("kept", {"boundarydata": boundary("FI")})
        manager.save_processor_frames("removed", {"boundarydata": boundary("SE")})

        after = make_manager(tmp_path, specs={"kept": {}}).load_processor_frames()

        assert set(after) == {"kept"}

    def test_and_the_drop_is_recorded(self, tmp_path):
        manager = make_manager(tmp_path, specs={"removed": {}})
        manager.save_processor_frames("removed", {"boundarydata": boundary("SE")})

        shorter = make_manager(tmp_path, specs={"kept": {}})
        shorter.load_processor_frames()

        shorter.logger.assert_logged("removed")

    def test_the_shorter_dict_is_what_gets_written_back(self, tmp_path):
        manager = make_manager(tmp_path, specs={"kept": {}, "removed": {}})
        manager.save_processor_frames("kept", {"boundarydata": boundary("FI")})
        manager.save_processor_frames("removed", {"boundarydata": boundary("SE")})

        shorter = make_manager(tmp_path, specs={"kept": {}})
        shorter.save_processor_frames("kept", {"boundarydata": boundary("FI")})

        assert set(make_manager(tmp_path).load_processor_frames()) == {"kept"}

    def test_a_config_naming_no_specs_prunes_nothing(self, tmp_path):
        """An empty ``timeseries_specs`` is "cannot tell", not "none of them"."""
        manager = make_manager(tmp_path)
        manager.save_processor_frames("hydro storage limits", {"boundarydata": boundary("FI")})

        assert set(make_manager(tmp_path).load_processor_frames()) == {"hydro storage limits"}


class TestACacheThatCannotBeRead:
    """It is read once per processor, from inside ProcessorRunner.run, so raising
    would take the whole build down rather than cost it its warm start."""

    def test_it_reads_as_empty_rather_than_raising(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.save_processor_frames("hydro storage limits", {"boundarydata": boundary("FI")})
        manager.processor_frames_file.write_bytes(b"not a pickle")

        assert make_manager(tmp_path).load_processor_frames() == {}

    def test_and_says_so(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.processor_frames_file.write_bytes(b"not a pickle")
        manager.load_processor_frames()

        manager.logger.assert_logged("processor_frames.pkl", level="warn")

    def test_a_later_save_starts_the_file_over(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.processor_frames_file.write_bytes(b"not a pickle")
        manager.save_processor_frames("hydro storage limits", {"boundarydata": boundary("FI")})

        assert set(make_manager(tmp_path).load_processor_frames()) == {"hydro storage limits"}

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.save_processor_frames("hydro storage limits", {"boundarydata": boundary("FI")})

        assert not list(manager.cache_folder.glob("*.tmp"))
