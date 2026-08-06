"""The real entry point: ``build_input_data.main()``.

The thinnest tier, and the only one that exercises the scenario loop, the
``.ini`` parsing and ``CacheManager`` together. Everything below it is covered
faster elsewhere, so there are only a handful of tests here -- but they are the
ones that would catch a wiring mistake no unit test can see.

``timeseries_specs`` is empty throughout: the timeseries tier needs the ~1 GB of
real PECD/TYNDP inputs, and its logic is tested directly in ``tests/unit``.
``_check_dependencies`` is stubbed out because it requires the GAMS executable
on PATH, and requiring a GAMS install to test Excel assembly would make this
tier unrunnable almost everywhere.
"""

import configparser
from pathlib import Path

import pytest

import build_input_data
from tests._common.asserts import assert_workbook_consistent
from tests._common.excel_read import read_output_workbook
from tests._common.routes import build_input_folder, config_for_workbooks
from tests._common.workbook_text import load_workbook_fixture

pytestmark = pytest.mark.entrypoint

MINIMAL = load_workbook_fixture("minimal")


def _write_ini(path: Path, config: dict) -> Path:
    """Render a config dict as the ``[inputdata]`` .ini main() will parse."""
    parser = configparser.ConfigParser()
    parser["inputdata"] = {
        "output_folder_prefix": str(config["output_folder_prefix"]),
        "scenarios": repr(config["scenarios"]),
        "scenario_years": repr(config["scenario_years"]),
        "climate_data": str(config["climate_data"]),
        "country_codes": repr(config["country_codes"]),
        "bb_timeseries_length": str(config["bb_timeseries_length"]),
        "unitdata_files": repr(config["unitdata_files"]),
        "unittypedata_files": repr(config["unittypedata_files"]),
        "nodedata_files": repr(config["nodedata_files"]),
        "demanddata_files": repr(config["demanddata_files"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return path


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A complete input folder plus a matching .ini, ready for main()."""
    monkeypatch.setattr(build_input_data, "_check_dependencies", lambda: None)

    workbooks = {"data.xlsx": MINIMAL}
    input_folder = build_input_folder(tmp_path, workbooks=workbooks, gams_files=True)
    config = config_for_workbooks(workbooks)

    def make(**overrides):
        merged = {**config, **overrides}
        return input_folder, _write_ini(input_folder / "config_test.ini", merged)

    return make


class TestASingleScenario:
    def test_builds_a_consistent_workbook(self, project, tmp_path):
        input_folder, config_file = project()

        build_input_data.main(input_folder, config_file, output_root=tmp_path / "out")

        folders = list((tmp_path / "out").iterdir())
        assert len(folders) == 1
        workbook = folders[0] / "inputData.xlsx"
        assert workbook.is_file()
        assert_workbook_consistent(read_output_workbook(workbook))

    def test_the_folder_is_named_from_prefix_scenario_and_year(self, project, tmp_path):
        input_folder, config_file = project(
            output_folder_prefix="myrun", scenarios=["base"], scenario_years=[2040]
        )

        build_input_data.main(input_folder, config_file, output_root=tmp_path / "out")

        assert (tmp_path / "out" / "myrun_base_2040").is_dir()

    def test_a_summary_log_is_written(self, project, tmp_path):
        input_folder, config_file = project()
        build_input_data.main(input_folder, config_file, output_root=tmp_path / "out")

        summary = next((tmp_path / "out").iterdir()) / "summary.log"
        assert summary.is_file() and summary.read_text(encoding="utf-8").strip()

    def test_the_gams_templates_are_copied_and_patched(self, project, tmp_path):
        # The finalize step: templates are copied out of the input folder with
        # config-derived substitutions applied.
        input_folder, config_file = project(bb_timeseries_length=2)
        build_input_data.main(input_folder, config_file, output_root=tmp_path / "out")

        schedule = (next((tmp_path / "out").iterdir()) / "scheduleInit.gms").read_text(
            encoding="utf-8"
        )
        assert "'dataLength') =  48;" in schedule     # 2 days * 24 h


class TestTheScenarioLoop:
    def test_every_combination_gets_its_own_folder(self, project, tmp_path):
        """The cartesian product over scenarios and years.

        A wiring mistake here silently builds fewer folders than asked for, and
        the missing ones look like they were never requested.
        """
        input_folder, config_file = project(
            scenarios=["a", "b"], scenario_years=[2030, 2040]
        )

        build_input_data.main(input_folder, config_file, output_root=tmp_path / "out")

        names = sorted(p.name for p in (tmp_path / "out").iterdir())
        assert names == [
            "test_a_2030", "test_a_2040", "test_b_2030", "test_b_2040",
        ]

    def test_each_folder_gets_its_own_cache(self, project, tmp_path):
        # The cache lives inside the output folder, so scenarios must not share
        # one -- otherwise the second run reads the first one's state.
        input_folder, config_file = project(scenarios=["a", "b"])
        build_input_data.main(input_folder, config_file, output_root=tmp_path / "out")

        for folder in (tmp_path / "out").iterdir():
            assert (folder / "cache").is_dir()


class TestOutputLocation:
    def test_output_goes_where_it_is_told_regardless_of_the_working_directory(
        self, project, tmp_path, monkeypatch
    ):
        """The seam this tier exists to prove.

        main() used to create its output folder relative to the working
        directory, so the same command run from two places maintained two
        independent caches and two half-built outputs -- and since the cache
        lives inside the output folder, neither knew about the other.
        """
        input_folder, config_file = project()
        elsewhere = tmp_path / "somewhere_else"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        build_input_data.main(input_folder, config_file, output_root=tmp_path / "out")

        assert list((tmp_path / "out").iterdir())
        assert not list(elsewhere.iterdir())

    def test_the_cache_finds_its_source_files_from_another_directory(
        self, project, tmp_path, monkeypatch
    ):
        """CacheManager hashes the pipeline's own source files to detect changes.

        Those paths were resolved against the working directory, so running from
        anywhere else died on a bare FileNotFoundError naming
        ./build_input_data.py -- which reads like a broken installation rather
        than a wrong directory.
        """
        input_folder, config_file = project()
        monkeypatch.chdir(tmp_path)

        build_input_data.main(input_folder, config_file, output_root=tmp_path / "out")

        cache = next((tmp_path / "out").iterdir()) / "cache"
        assert (cache / "overall_code_files_hashes.json").is_file()


class TestRerunBehaviour:
    def test_a_second_run_reuses_the_cache_and_still_produces_the_workbook(
        self, project, tmp_path
    ):
        """The only end-to-end exercise of CacheManager.

        The second run must be a no-op that still leaves a valid workbook
        behind -- a cache that skips work but also skips the output is worse
        than no cache.
        """
        input_folder, config_file = project()
        out = tmp_path / "out"

        build_input_data.main(input_folder, config_file, output_root=out)
        first = (next(out.iterdir()) / "inputData.xlsx").read_bytes()

        build_input_data.main(input_folder, config_file, output_root=out)
        workbook = next(out.iterdir()) / "inputData.xlsx"

        assert workbook.is_file()
        assert_workbook_consistent(read_output_workbook(workbook))
        assert len(workbook.read_bytes()) == pytest.approx(len(first), rel=0.05)

    def test_editing_the_source_data_is_picked_up_on_the_next_run(self, project, tmp_path):
        # The cache keys on sheet values, so an edited workbook must rebuild.
        from tests._common.workbook_text import workbook_text_with, write_workbook_text

        input_folder, config_file = project()
        out = tmp_path / "out"
        build_input_data.main(input_folder, config_file, output_root=out)

        edited = workbook_text_with(
            MINIMAL, sheet="unitdata", header="capacity_output1", value=999,
            where={"Country": "FI", "Generator_ID": "windturbine"},
        )
        write_workbook_text(edited, input_folder / "data_files" / "data.xlsx")

        build_input_data.main(input_folder, config_file, output_root=out)

        sheets = read_output_workbook(next(out.iterdir()) / "inputData.xlsx")
        capacities = sheets["p_gnu_io"]["capacity"].astype(float).tolist()
        assert 999 in capacities
