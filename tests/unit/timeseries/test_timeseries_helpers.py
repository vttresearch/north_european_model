"""The GAMS import include file, and the unitdata gate beside it.

``update_import_timeseries_inc`` appends a ``$gdxin`` block to the file Backbone
reads. It is called once per processor per run and must be idempotent -- a
duplicated block would load the same parameter twice.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.timeseries.timeseries_helpers import (
    nodes_needing_flow,
    update_import_timeseries_inc,
)


class TestUpdateImportTimeseriesInc:
    def _write_gdx(self, folder: Path, name: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"not a real gdx")

    def test_writes_a_gdxin_block_for_a_single_file(self, tmp_path):
        self._write_gdx(tmp_path, "ts_influx_elec.gdx")
        update_import_timeseries_inc(
            tmp_path, bb_parameter="ts_influx", gdx_name_suffix="elec"
        )

        content = (tmp_path / "import_timeseries.inc").read_text()
        assert "$$gdxin '%input_dir%/ts_influx_elec.gdx'" in content
        assert "$$loaddcm ts_influx" in content

    def test_per_year_files_are_imported_through_the_climate_year_macro(self, tmp_path):
        # One block covers every climate year; Backbone substitutes the year.
        for year in (2014, 2015, 2016):
            self._write_gdx(tmp_path, f"ts_influx_elec_{year}.gdx")
        update_import_timeseries_inc(
            tmp_path, bb_parameter="ts_influx", gdx_name_suffix="elec"
        )

        content = (tmp_path / "import_timeseries.inc").read_text()
        assert "ts_influx_elec_%climateYear%.gdx" in content

    def test_calling_it_twice_does_not_duplicate_the_block(self, tmp_path):
        """Idempotency, which the build depends on.

        The file is appended to, once per processor per run, and a repeated
        block would load the same parameter twice.
        """
        self._write_gdx(tmp_path, "ts_influx_elec.gdx")
        for _ in range(3):
            update_import_timeseries_inc(
                tmp_path, bb_parameter="ts_influx", gdx_name_suffix="elec"
            )

        content = (tmp_path / "import_timeseries.inc").read_text()
        assert content.count("$$loaddcm ts_influx") == 1

    def test_different_parameters_each_get_their_own_block(self, tmp_path):
        self._write_gdx(tmp_path, "ts_influx_elec.gdx")
        self._write_gdx(tmp_path, "ts_cf_wind.gdx")
        update_import_timeseries_inc(tmp_path, bb_parameter="ts_influx", gdx_name_suffix="elec")
        update_import_timeseries_inc(tmp_path, bb_parameter="ts_cf", gdx_name_suffix="wind")

        content = (tmp_path / "import_timeseries.inc").read_text()
        assert "$$loaddcm ts_influx" in content
        assert "$$loaddcm ts_cf" in content

    def test_the_block_is_guarded_so_a_missing_gdx_is_not_fatal_for_gams(self, tmp_path):
        # $ifthen exist: a model can be run with a subset of the GDX files
        # present without GAMS failing on the missing ones.
        self._write_gdx(tmp_path, "ts_influx_elec.gdx")
        update_import_timeseries_inc(tmp_path, bb_parameter="ts_influx", gdx_name_suffix="elec")

        content = (tmp_path / "import_timeseries.inc").read_text()
        assert content.strip().startswith("$ifthen exist")
        assert "$endIf" in content

    def test_a_named_suffix_file_is_imported_directly(self, tmp_path):
        self._write_gdx(tmp_path, "ts_influx_elec_forecasts.gdx")
        update_import_timeseries_inc(
            tmp_path, file_suffix="forecasts",
            bb_parameter="ts_influx", gdx_name_suffix="elec",
        )
        assert "ts_influx_elec_forecasts.gdx" in (tmp_path / "import_timeseries.inc").read_text()

    def test_a_missing_named_suffix_file_raises(self, tmp_path):
        # Asked for a specific file by name and it is not there: that is a
        # programming error rather than a data condition.
        with pytest.raises(FileNotFoundError, match="forecasts"):
            update_import_timeseries_inc(
                tmp_path, file_suffix="forecasts",
                bb_parameter="ts_influx", gdx_name_suffix="elec",
            )


class TestNodesNeedingFlow:
    """Which nodes the model attaches a unit of a given flow to.

    The gate that keeps a VRE processor from building -- and then complaining
    about -- a capacity factor nobody reads. `flow` reaches unitdata from
    unittypedata, and the merge has already applied this run's scenario, year and
    country filtering, so what is in the frame is what the model has.
    """

    @staticmethod
    def _frame(rows):
        return pd.DataFrame(rows, columns=["unit", "flow", "node_output1"])

    def test_a_node_with_a_unit_of_that_flow_is_named(self):
        frame = self._frame([("w1", "onshore", "FI00_elec")])
        assert nodes_needing_flow(frame, "onshore") == {"FI00_elec"}

    def test_another_flow_does_not_count(self):
        """An empty set, not None: the model was asked and has no such unit."""
        frame = self._frame([("w1", "offshore", "FI00_elec")])
        assert nodes_needing_flow(frame, "onshore") == set()

    def test_the_comparison_ignores_case_and_padding(self):
        frame = self._frame([("w1", " Onshore ", "FI00_elec")])
        assert nodes_needing_flow(frame, "onshore") == {"FI00_elec"}

    def test_input_nodes_do_not_count(self):
        """A flow says what the weather produces, so a fuel node is not it."""
        frame = pd.DataFrame(
            [("w1", "onshore", "FI00_elec", "FI00_gas")],
            columns=["unit", "flow", "node_output1", "node_input1"],
        )
        assert nodes_needing_flow(frame, "onshore") == {"FI00_elec"}

    @pytest.mark.parametrize("frame", [
        None,
        pd.DataFrame(),
        pd.DataFrame([("w1", "FI00_elec")], columns=["unit", "node_output1"]),
    ])
    def test_cannot_tell_is_none_so_the_caller_fails_open(self, frame):
        """Distinct from the empty set, which is a real answer meaning "none"."""
        assert nodes_needing_flow(frame, "onshore") is None

    def test_no_flow_asked_for_is_also_cannot_tell(self):
        frame = self._frame([("w1", "onshore", "FI00_elec")])
        assert nodes_needing_flow(frame, "") is None
