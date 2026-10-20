"""run_model.py: which input folder a run gets, and what --list says.

Expectations: ``tests/expectations/variants.md``, G9. The argument is the built
input folder, so every build of a config -- every scenario, year and build of its
``variants`` -- is run the same way.

Everything here runs against folders in ``tmp_path``. Nothing imports the parent
checkout's runner or launches GAMS: the functions under test decide what a run is
pointed at, which is where a wrong answer would cost a run.
"""

import json
import re
import textwrap
from types import SimpleNamespace

import pytest

import run_model

CONFIG = """\
[inputdata]
output_folder_prefix = input
scenarios = ['Observed Trends']
scenario_years = [2030]
climate_data = 2014-2015
country_codes = ['FI00', 'SE01']
variants = {'': {}, 'sens1': {}}
"""


def built(root, name):
    """A folder that looks like build_input_data.py wrote it."""
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "inputData.xlsx").write_bytes(b"")
    return folder


def with_config(root):
    (root / "src_files").mkdir(exist_ok=True)
    (root / "src_files" / "config_OT2030.ini").write_text(textwrap.dedent(CONFIG),
                                                          encoding="utf-8")


class TestWhichFolderARunGets:
    def test_a_built_folder_by_its_name(self, tmp_path):
        folder = built(tmp_path, "input_ObservedTrends_2030_sens1")
        assert run_model.resolve_input_folder(folder.name, here=tmp_path) == folder.resolve()

    def test_a_full_path(self, tmp_path):
        folder = built(tmp_path / "elsewhere", "input_x")
        assert run_model.resolve_input_folder(str(folder), here=tmp_path) == folder.resolve()

    def test_a_folder_that_was_never_built_is_refused(self, tmp_path):
        (tmp_path / "input_half").mkdir()
        with pytest.raises(SystemExit, match="inputData.xlsx"):
            run_model.resolve_input_folder("input_half", here=tmp_path)

    def test_a_scenario_name_is_answered_with_the_folders_it_builds(self, tmp_path):
        # The form this script used to take. One config can build several
        # folders, so the message names them all rather than guessing one.
        with_config(tmp_path)
        built(tmp_path, "input_ObservedTrends_2030")
        with pytest.raises(SystemExit) as raised:
            run_model.resolve_input_folder("OT2030", here=tmp_path)
        message = str(raised.value)
        assert "input_ObservedTrends_2030\n" in message
        assert "input_ObservedTrends_2030_sens1  (not built)" in message

    def test_an_unknown_name_points_to_the_list(self, tmp_path):
        with pytest.raises(SystemExit, match="--list"):
            run_model.resolve_input_folder("nope", here=tmp_path)

    def test_a_folder_outside_the_backbone_checkout_is_refused(self, tmp_path):
        # Backbone takes --input_dir relative to its own root.
        bb_root = tmp_path / "backbone"
        bb_root.mkdir()
        outside = built(tmp_path, "input_x")
        with pytest.raises(SystemExit, match="outside the Backbone checkout"):
            run_model.check_inside(outside, bb_root)
        inside = built(bb_root / "model", "input_y")
        assert run_model.check_inside(inside, bb_root) == inside


class TestTheRun:
    def test_the_default_tag_comes_from_the_folder(self, tmp_path):
        folder = built(tmp_path, "input_ObservedTrends_2030_sens1")
        assert run_model.default_tag(folder, 2015, 7) == "input_ObservedTrends_2030_sens1-2015-7d"

    def test_backbone_is_given_the_folder_relative_to_its_root(self, tmp_path):
        bb_root = tmp_path / "backbone"
        folder = built(bb_root / "model", "input_ObservedTrends_2030_sens1")
        args = SimpleNamespace(gams=None, solver=None, cpus=None, show=False, year=2015,
                               days=7, debug="1", backbone_args=[])
        argv = run_model.build_argv(args, bb_root, folder, bb_root / "model" / "results" / "t")
        assert "--input_dir=model/input_ObservedTrends_2030_sens1" in argv
        assert "--output-dir" in argv and "model/results/t" in argv

    def test_the_input_dir_option_is_gone(self):
        # The positional argument is the folder; there is no second way to name it.
        with pytest.raises(SystemExit):
            run_model.main(["--input-dir", "x"])


class TestPinningALongRun:
    """--cpus is the parent runner's own flag; --threads is the layer that goes with it."""

    @staticmethod
    def argv(tmp_path, *, cpus=None, backbone_args=(), threads_layer=None):
        bb_root = tmp_path / "backbone"
        folder = built(bb_root / "model", "input_x")
        output_dir = bb_root / "model" / "results" / "t"
        args = SimpleNamespace(gams=None, solver=None, cpus=cpus, show=False, year=2015,
                               days=365, debug="1", backbone_args=list(backbone_args))
        layer = output_dir / threads_layer if threads_layer else None
        return run_model.build_argv(args, bb_root, folder, output_dir, layer)

    def test_cpus_goes_to_the_runner_not_to_gams(self, tmp_path):
        # The pin is the runner's process affinity, which GAMS inherits. Behind the
        # bare -- it would reach GAMS as a --cpus parameter that nothing reads.
        argv = self.argv(tmp_path, cpus="0-3")
        cut = argv.index("--")
        assert argv[argv.index("--cpus") + 1] == "0-3"
        assert argv.index("--cpus") < cut

    def test_unpinned_unless_asked(self, tmp_path):
        argv = self.argv(tmp_path)
        assert "--cpus" not in argv
        assert not any(re.match(r"--mod\d_dir=", a) for a in argv)

    def test_the_threads_layer_takes_the_last_slot(self, tmp_path):
        # Backbone includes the slots' 1_options.gms in order and the last wins.
        argv = self.argv(tmp_path, threads_layer=run_model.THREADS_LAYER)
        assert argv[-1] == f"--mod5_dir=model/results/t/{run_model.THREADS_LAYER}"

    def test_the_callers_own_mods_keep_their_slots(self, tmp_path):
        argv = self.argv(tmp_path, backbone_args=["--mod1_dir=mods/a", "--mod5_dir=mods/b"],
                         threads_layer=run_model.THREADS_LAYER)
        assert "--mod1_dir=mods/a" in argv and "--mod5_dir=mods/b" in argv
        assert argv[-1].startswith("--mod4_dir=")

    def test_no_free_slot_is_refused(self, tmp_path):
        # Dropping the cap quietly would leave a pinned run on the model's own count.
        taken = [f"--mod{n}_dir=mods/m{n}" for n in range(1, 6)]
        with pytest.raises(SystemExit, match="all 5 are taken"):
            self.argv(tmp_path, backbone_args=taken, threads_layer=run_model.THREADS_LAYER)

    def test_the_layer_sets_option_threads(self, tmp_path):
        layer = run_model.write_threads_layer(tmp_path / "results" / "t" / "layer", 1)
        assert (layer / "1_options.gms").read_text(encoding="utf-8") == "option threads = 1;\n"


class TestScheduleAndInvestmentRuns:
    """1_options.gms gives every run one thread, which wins in schedule runs only."""

    @staticmethod
    def note(*backbone_args, threads=None):
        return run_model.threads_note(
            SimpleNamespace(backbone_args=list(backbone_args), threads=threads))

    def test_a_schedule_run_is_silent(self):
        assert self.note() is None
        assert self.note("--init_file=scheduleInit.gms") is None

    def test_another_init_file_on_one_thread_is_told_how_to_raise_it(self):
        note = self.note("--debug=2", "--init_file=investInit.gms")
        assert "--init_file=investInit.gms" in note and "--threads -1" in note

    def test_an_init_file_with_threads_set_is_silent(self):
        # The caller has chosen; the note would only repeat what they did.
        assert self.note("--init_file=investInit.gms", threads=-1) is None
        assert self.note("--init_file=investInit.gms", threads=1) is None


class TestList:
    def test_built_folders_who_holds_them_and_where_they_come_from(self, tmp_path, capsys):
        with_config(tmp_path)
        built(tmp_path, "input_ObservedTrends_2030")
        busy = built(tmp_path, "input_ObservedTrends_2030_sens1")
        (busy / run_model.LOCK_NAME).write_text(json.dumps(
            {"tag": "sweep-1", "pid": 1, "started": "2026-10-05T00:00:00+00:00"}),
            encoding="utf-8")
        built(tmp_path, "input_handmade")
        (tmp_path / "results").mkdir()

        assert run_model.do_list(here=tmp_path) == 0
        lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()}

        assert set(lines) == {"input_ObservedTrends_2030", "input_ObservedTrends_2030_sens1",
                              "input_handmade"}
        assert "free" in lines["input_ObservedTrends_2030"]
        assert "config_OT2030.ini" in lines["input_ObservedTrends_2030"]
        assert "BUSY (sweep-1)" in lines["input_ObservedTrends_2030_sens1"]
        assert "build 'sens1'" in lines["input_ObservedTrends_2030_sens1"]
        assert "no config" in lines["input_handmade"]

    def test_nothing_built_says_how_to_build(self, tmp_path, capsys):
        assert run_model.do_list(here=tmp_path) == 0
        assert "build_input_data.py" in capsys.readouterr().out
