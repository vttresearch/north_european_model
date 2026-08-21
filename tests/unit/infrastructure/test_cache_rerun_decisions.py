"""What the cache decides to re-run, and why.

Two decisions here used to be wrong in ways a build could not show you.

A processor's output does not stop at the GDX -- a secondary result feeds
``create_p_gn``, ``create_p_gnBoundaryPropertiesForStates`` and
``add_storage_starts``. Processor files are hashed individually by
``ProcessorRunner`` and deliberately kept out of ``_TS_PIPELINE_FILES``, so
editing one used to re-run the timeseries phase while leaving
``inputData.xlsx`` describing the previous run. The workbook went stale against
its own GDX, silently.

And a processor that declares ``requires_source_data`` reads a source workbook,
so editing that workbook has to re-run it. The link used to exist only for
``demand_grid``; anything else served stale GDX.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.cache_manager import CacheManager
from tests._common.fixtures import FakeLogger, make_config

SPEC = {
    "processor_name": "some_processor",
    "bb_parameter": "ts_influx",
    "bb_parameter_dimensions": ["grid", "node", "f", "t"],
    "demand_grid": "",
    "secondary_output_name": None,
}


def make_manager(tmp_path: Path, **config_overrides) -> CacheManager:
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    (tmp_path / "input" / "data_files").mkdir(parents=True, exist_ok=True)
    return CacheManager(
        input_folder=tmp_path / "input",
        output_folder=output,
        config=make_config(**config_overrides),
        logger=FakeLogger(),
    )


class TestRebuildBBExcelFollowsTheTimeseries:
    """A changed processor must rebuild the workbook, not only the GDX."""

    def test_a_changed_processor_rebuilds_the_workbook(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.timeseries_changed = {"hydro storage limits": True}
        assert manager.any_timeseries_changed

        # The expression under test, evaluated the way run() evaluates it.
        rebuild = (
            manager.full_rerun
            or manager.demand_files_changed
            or manager.other_input_files_changed
            or manager.bb_excel_pipeline_code_updated
            or manager.any_timeseries_changed
        )
        assert rebuild, (
            "a processor's secondary result feeds p_gn and "
            "p_gnBoundaryPropertiesForStates; leaving the workbook alone makes it "
            "stale against the GDX written in the same run"
        )

    def test_nothing_changed_does_not_rebuild(self, tmp_path):
        """The positive control -- otherwise the assertion above proves nothing."""
        manager = make_manager(tmp_path)
        manager.timeseries_changed = {"hydro storage limits": False}
        assert not manager.any_timeseries_changed

    def test_processors_are_not_in_the_pipeline_file_groups(self):
        """Why the flag is needed at all.

        If processor modules were watched as a group, a processor edit would set
        timeseries_pipeline_code_updated and trigger a full rerun. They are not,
        by design -- they are hashed one at a time.
        """
        watched = {
            Path(p).as_posix().lstrip("./")
            for group in (CacheManager._TS_PIPELINE_FILES, CacheManager._OVERALL_CODE_FILES)
            for p in group
        }
        assert "src/timeseries/processors/hydro_inflow_MAF2019.py" not in watched
        assert "src/timeseries/processors/base_processor.py" in watched, (
            "the base class carries the declaration defaults, so it is watched"
        )


class TestSourceDataRequirementsDriveReruns:
    """`requires_source_data` has to reach the cache, or edits go unnoticed."""

    def _prev_config(self, spec=None):
        return {"timeseries_specs": {"hydro": json.loads(json.dumps(spec or SPEC))}}

    def test_a_declared_category_changing_reruns_the_processor(self, tmp_path):
        manager = make_manager(tmp_path, timeseries_specs={"hydro": dict(SPEC)})
        manager.save_processor_requirements("some_processor", ["nodedata"])

        changed = manager._detect_timeseries_spec_changes(
            manager.config, self._prev_config(), {"nodedata_files": True}
        )
        assert changed["hydro"], (
            "editing hydroUpd-v1.xlsx must re-run a processor that reads nodedata"
        )

    def test_an_undeclared_category_changing_does_not(self, tmp_path):
        """Only the frames a processor asked for should wake it."""
        manager = make_manager(tmp_path, timeseries_specs={"hydro": dict(SPEC)})
        manager.save_processor_requirements("some_processor", ["nodedata"])

        changed = manager._detect_timeseries_spec_changes(
            manager.config, self._prev_config(), {"transferdata_files": True}
        )
        assert not changed["hydro"]

    def test_a_processor_with_no_recorded_requirement_is_rerun_conservatively(self, tmp_path):
        """First run, cleared cache, or a processor that never finished.

        Nothing is recorded, so its requirements are unknown rather than empty.
        Re-running is the cheap mistake; serving stale GDX is not.
        """
        manager = make_manager(tmp_path, timeseries_specs={"hydro": dict(SPEC)})
        changed = manager._detect_timeseries_spec_changes(
            manager.config, self._prev_config(), {"nodedata_files": True}
        )
        assert changed["hydro"]

    def test_no_input_change_leaves_an_unrecorded_processor_alone(self, tmp_path):
        """Conservative, not paranoid: with nothing changed there is nothing to do."""
        manager = make_manager(tmp_path, timeseries_specs={"hydro": dict(SPEC)})
        changed = manager._detect_timeseries_spec_changes(
            manager.config, self._prev_config(), {"nodedata_files": False}
        )
        assert not changed["hydro"]

    def test_demand_grid_still_works(self, tmp_path):
        """The original special case has to survive being generalised."""
        spec = dict(SPEC, demand_grid="elec")
        manager = make_manager(tmp_path, timeseries_specs={"hydro": spec})
        manager.save_processor_requirements("some_processor", [])

        changed = manager._detect_timeseries_spec_changes(
            manager.config, self._prev_config(spec), {"demanddata_files": True}
        )
        assert changed["hydro"]

    def test_requirements_survive_a_round_trip(self, tmp_path):
        manager = make_manager(tmp_path)
        manager.save_processor_requirements("a", ["nodedata"])
        manager.save_processor_requirements("b", [])
        assert manager.load_processor_requirements() == {"a": ["nodedata"], "b": []}


class TestSourceExcelsAreLoadedForDeclaringProcessors:
    """A declaring processor must never meet an unloaded SourceDataPipeline.

    ``SourceDataPipeline.run()`` is conditional, and its frames start empty. If
    the cache lets a declaring processor run without forcing the import, the
    processor refuses and writes no GDX -- for a reason the user cannot see.
    """

    @pytest.mark.parametrize("requirement,expected", [(["nodedata"], True), ([], False)])
    def test_a_changed_declaring_processor_forces_the_import(
        self, tmp_path, requirement, expected
    ):
        manager = make_manager(tmp_path, timeseries_specs={"hydro": dict(SPEC)})
        manager.save_processor_requirements("some_processor", requirement)
        manager.timeseries_changed = {"hydro": True}

        recorded = manager.load_processor_requirements()
        forces_import = any(
            manager.timeseries_changed.get(name, False)
            and (spec.get("demand_grid") or recorded.get(spec.get("processor_name"), []))
            for name, spec in manager.config["timeseries_specs"].items()
        )
        assert bool(forces_import) is expected
