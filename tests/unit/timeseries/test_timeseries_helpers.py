"""The GAMS import include file.

``update_import_timeseries_inc`` appends a ``$gdxin`` block to the file Backbone
reads. It is called once per processor per run and must be idempotent -- a
duplicated block would load the same parameter twice.
"""

from pathlib import Path

import pytest

from src.timeseries.timeseries_helpers import update_import_timeseries_inc


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
