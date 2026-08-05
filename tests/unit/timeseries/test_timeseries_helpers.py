"""Domain caching and the GAMS import include file.

``collect_domains_for_cache`` / ``collect_domain_pairs_for_cache`` gather what a
processor produced so later stages know which (grid, node) combinations exist
without re-reading the GDX. The pairs matter as much as the members: knowing
``elec`` and ``FI_heat`` both exist says nothing about whether ``elec/FI_heat``
does, and generating parameters for combinations that do not exist is what the
pair cache prevents.

``update_import_timeseries_inc`` appends a ``$gdxin`` block to the file Backbone
reads. It is called once per processor per run and must be idempotent -- a
duplicated block would load the same parameter twice.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.timeseries.timeseries_helpers import (
    collect_domain_pairs_for_cache,
    collect_domains_for_cache,
    update_import_timeseries_inc,
)

DOMAINS = ["grid", "node", "flow", "group"]
PAIRS = [["grid", "node"], ["flow", "node"]]


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


class TestCollectDomains:
    def test_collects_the_distinct_values_of_each_present_domain(self):
        out = collect_domains_for_cache(
            _frame(grid=["elec", "elec", "heat"], node=["a", "b", "c"]), DOMAINS
        )
        assert sorted(out["grid"]) == ["elec", "heat"]
        assert sorted(out["node"]) == ["a", "b", "c"]

    def test_absent_domains_are_omitted_rather_than_empty(self):
        # The dict is merged across processors, so an empty entry would claim
        # the processor had opinions about a domain it never touched.
        out = collect_domains_for_cache(_frame(grid=["elec"]), DOMAINS)
        assert set(out) == {"grid"}

    def test_missing_values_are_not_collected(self):
        # pd.NA is not a GAMS set element.
        out = collect_domains_for_cache(_frame(grid=["elec", pd.NA]), DOMAINS)
        assert out["grid"] == ["elec"]

    def test_a_column_of_only_missing_values_is_omitted(self):
        out = collect_domains_for_cache(_frame(grid=[pd.NA, pd.NA]), DOMAINS)
        assert out == {}

    def test_the_result_is_json_serialisable(self):
        # It is written straight to a JSON cache file.
        import json

        out = collect_domains_for_cache(_frame(grid=["elec"], node=["FI_elec"]), DOMAINS)
        assert json.loads(json.dumps({k: list(v) for k, v in out.items()}))

    def test_case_is_preserved_rather_than_normalised(self):
        # Normalisation happens once, downstream in compile_domain_df, which
        # keeps the first-seen spelling. Folding here would pre-empt that.
        out = collect_domains_for_cache(_frame(node=["FI_heat_HKI"]), DOMAINS)
        assert out["node"] == ["FI_heat_HKI"]


class TestCollectDomainPairs:
    def test_collects_distinct_pairs(self):
        out = collect_domain_pairs_for_cache(
            _frame(grid=["elec", "elec", "heat"], node=["a", "a", "b"]), PAIRS
        )
        assert sorted(out["grid_node"]) == [("elec", "a"), ("heat", "b")]

    def test_a_pair_is_not_implied_by_its_members(self):
        """The reason the pair cache exists.

        Two grids and two nodes do not mean four combinations. Generating
        parameters for pairs that never occur is exactly what this prevents.
        """
        out = collect_domain_pairs_for_cache(
            _frame(grid=["elec", "heat"], node=["FI_elec", "FI_heat"]), PAIRS
        )
        assert set(out["grid_node"]) == {("elec", "FI_elec"), ("heat", "FI_heat")}
        assert ("elec", "FI_heat") not in out["grid_node"]

    def test_pairs_with_a_missing_column_are_skipped(self):
        out = collect_domain_pairs_for_cache(_frame(grid=["elec"], node=["a"]), PAIRS)
        assert set(out) == {"grid_node"}      # flow_node has no flow column

    def test_the_key_names_both_domains(self):
        out = collect_domain_pairs_for_cache(_frame(flow=["wind"], node=["a"]), PAIRS)
        assert set(out) == {"flow_node"}

    @pytest.mark.parametrize("bad", [["grid"], ["grid", "node", "unit"], "grid"])
    def test_a_pair_that_is_not_two_names_is_rejected(self, bad):
        # Raised rather than logged: a malformed pair list is a caller bug, and
        # silently skipping it would quietly disable the check it asks for.
        with pytest.raises(ValueError, match="exactly two"):
            collect_domain_pairs_for_cache(_frame(grid=["elec"], node=["a"]), [bad])


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
