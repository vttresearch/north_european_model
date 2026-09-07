"""The shipped configs name files that exist, spelled the way the folder spells them.

``read_input_excels`` opens each listed file by name. On NTFS a wrong case opens
the file anyway, so a mismatch survives every local run and every local test, and
fails only on a case-sensitive filesystem -- where the file is simply not found,
the category comes back empty, and the build produces a model missing whatever
that workbook carried.

``transferdata_files`` spelled ``finland_dheat_and_industry.xlsx`` in all four
configs while every other list spelled it ``Finland_...``. That is the shape this
guards: one entry out of twenty-one, invisible on the machine it was written on.
"""

import ast
import configparser

import pytest

CATEGORIES = ("unitdata", "unittypedata", "nodedata", "transferdata",
              "demanddata", "emissiondata", "userconstraintdata")


def _configs(src_files_dir):
    return sorted(src_files_dir.glob("config_*.ini"))


def _listed_files(config_path):
    parser = configparser.ConfigParser(inline_comment_prefixes=None)
    parser.read(config_path, encoding="utf-8")
    for category in CATEGORIES:
        raw = parser["inputdata"].get(f"{category}_files", "[]")
        for filename in ast.literal_eval(raw):
            yield category, filename


class TestEveryListedWorkbookExists:
    def test_there_are_configs_to_check(self, src_files_dir):
        """Guards the sweep below from passing because it found nothing."""
        assert _configs(src_files_dir)

    def test_every_entry_names_a_file_that_is_there(self, src_files_dir):
        missing = [
            f"{config.name}: {category}_files names {filename!r}"
            for config in _configs(src_files_dir)
            for category, filename in _listed_files(config)
            if not (src_files_dir / "data_files" / filename).exists()
        ]
        assert not missing, "\n".join(missing)

    def test_every_entry_spells_it_the_way_the_folder_does(self, src_files_dir):
        """Exact case, which Path.exists() on Windows will not tell you."""
        data_files = src_files_dir / "data_files"
        on_disk = {path.name for path in data_files.glob("*.xlsx")}

        wrong = [
            f"{config.name}: {category}_files names {filename!r}, "
            f"but the folder spells it {next(n for n in on_disk if n.lower() == filename.lower())!r}"
            for config in _configs(src_files_dir)
            for category, filename in _listed_files(config)
            if filename not in on_disk
            and any(name.lower() == filename.lower() for name in on_disk)
        ]
        assert not wrong, "\n".join(wrong)
