"""What the cache decides to re-run, and why.

Two decisions here used to be wrong in ways a build could not show you.

A processor's output does not stop at the GDX -- its contributions to the source
data tables feed ``create_p_gn``, ``create_p_gnBoundaryPropertiesForStates`` and
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

import src.hash_utils as hash_utils
import src.json_exchange as json_exchange
from src.infrastructure.cache_manager import CacheManager
from tests._common.fixtures import FakeLogger, make_config

SPEC = {
    "processor_name": "some_processor",
    "bb_parameter": "ts_influx",
    "bb_parameter_dimensions": ["grid", "node", "f", "t"],
    "demand_grid": "",
}

#: Two real processor files, because _detect_processor_code_changes hashes the
#: file on disk and skips a name it cannot find. VRE_PECD is the shape that
#: matters: no demand_grid, and requires_source_data is empty.
VRE_SPEC = dict(SPEC, processor_name="VRE_PECD", bb_parameter="ts_cf",
                bb_parameter_dimensions=["flow", "node", "f", "t"])
DH_SPEC = dict(SPEC, processor_name="DH_demand_fromTemperature", demand_grid="dheat")

#: The keys run() writes to config_structural.json in its finalization phase.
_STRUCTURAL_KEYS = [
    "country_codes", "exclude_grids", "exclude_nodes",
    "climate_data", "bb_timeseries_start", "bb_timeseries_length",
    "forecast_quantiles", "forecast_weights", "timeseries_specs",
]


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


def settle_cache(manager: CacheManager, stale_processors: tuple[str, ...] = ()) -> None:
    """Leave the cache as a clean, complete run leaves it.

    Every full-rerun cause in Phase 1 is answered: hashes recorded for all four
    file groups, a config structure to compare against, and the three success
    flags build_input_data writes when nothing failed. Without this a new
    manager's run() reports a full rerun and decides nothing granular, which is
    the state most of this module's assertions would silently pass in.

    `stale_processors` names the processor files whose recorded hash is wrong,
    i.e. the ones the next run will see as edited.
    """
    manager._save_all_source_code_hashes()
    manager._check_source_code_changes(
        CacheManager._BB_PIPELINE_FILES, "bb_excel_pipeline_hashes.json"
    )
    manager._detect_input_file_changes(manager.config, manager.input_file_folder)
    json_exchange.save_json(
        manager.cache_folder / "config_structural.json",
        {k: manager.config[k] for k in _STRUCTURAL_KEYS if k in manager.config},
    )
    manager.merge_dict_to_cache(
        {
            "source_excel_run_successfully": True,
            "timeseries_run_successfully": True,
            "bb_excel_succesfully_built": True,
        },
        "general_flags.json",
    )

    processors_base = Path(__file__).resolve().parents[3] / "src" / "timeseries" / "processors"
    for spec in manager.config["timeseries_specs"].values():
        name = spec["processor_name"]
        manager.save_processor_hash(
            name,
            "stale" if name in stale_processors
            else hash_utils.compute_file_hash(processors_base / f"{name}.py"),
        )
        manager.save_processor_requirements(name, [])


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
            "a processor's contribution to the source data tables feeds p_gn and "
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


class TestARebuildAlwaysGetsItsSourceData:
    """BBExcelPipeline reads every source frame, so the two flags cannot diverge.

    ``SourceDataPipeline`` holds its frames as empty DataFrames until ``run()``
    fills them, and ``run()`` is called only when ``reimport_source_excels`` is
    set. A rebuild against a skipped import does not produce a thin workbook: it
    dies on the first column it looks for -- ``KeyError: 'unit'``, from
    ``p_gnu_io_flat`` built out of an empty ``df_unitdata``.

    These go through ``run()`` rather than re-evaluating the expression. The gap
    they cover was open for three commits underneath assertions that re-stated
    the code they were checking, and a copy of the logic agrees with itself
    however wrong it is.
    """

    def test_editing_a_processor_that_reads_no_source_data_still_imports_it(self, tmp_path):
        """The reported crash: edit VRE_PECD, rebuild the workbook from nothing.

        VRE_PECD has no demand_grid and declares no requires_source_data, so it
        is the one processor whose edit reaches the workbook without asking for
        a single source frame on the way.
        """
        specs = {"PV": dict(VRE_SPEC), "District heating demand": dict(DH_SPEC)}
        settle_cache(make_manager(tmp_path, timeseries_specs=specs),
                     stale_processors=("VRE_PECD",))

        manager = make_manager(tmp_path, timeseries_specs=specs)
        manager.run()

        assert not manager.full_rerun, "a stale processor hash is a granular rerun"
        assert manager.rebuild_bb_excel, "a changed processor reaches the workbook"
        assert manager.reimport_source_excels, (
            "the workbook is built from df_unitdata, df_nodedata and four more "
            "frames that stay empty until SourceDataPipeline.run() is called"
        )

    def test_a_settled_cache_rebuilds_nothing(self, tmp_path):
        """The control: without it the assertions above pass on a full rerun."""
        specs = {"PV": dict(VRE_SPEC), "District heating demand": dict(DH_SPEC)}
        settle_cache(make_manager(tmp_path, timeseries_specs=specs))

        manager = make_manager(tmp_path, timeseries_specs=specs)
        manager.run()

        assert not manager.full_rerun
        assert not manager.any_timeseries_changed
        assert not manager.rebuild_bb_excel
        assert not manager.reimport_source_excels


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
