"""Tests for ``src/hash_utils.py``.

Hashing drives the cache, so the property that matters is not *which* hash a
file gets but *when* two things hash alike.  Two rules carry real weight:

- ``compute_excel_sheets_hash`` hashes sheet **values**, not file bytes, so a
  workbook rewritten with identical contents is cache-stable.  Everything in the
  planned text-fixture route depends on that.
- ``compute_file_hash`` raises rather than degrading when a file is missing.
  That is worth pinning: the alternative -- returning a sentinel -- would make
  the cache silently stop detecting source-code changes.
"""

import pandas as pd
import pytest

from src.hash_utils import compute_excel_sheets_hash, compute_file_hash, compute_folder_hash


def _write_xlsx(path, sheets: dict):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return path


class TestComputeFileHash:
    def test_identical_content_hashes_alike(self, tmp_path):
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text("same", encoding="utf-8")
        b.write_text("same", encoding="utf-8")
        assert compute_file_hash(a) == compute_file_hash(b)

    def test_different_content_hashes_differently(self, tmp_path):
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text("one", encoding="utf-8")
        b.write_text("two", encoding="utf-8")
        assert compute_file_hash(a) != compute_file_hash(b)

    def test_raises_on_a_missing_file(self, tmp_path):
        """Pinned deliberately: this must NOT degrade to a sentinel.

        The cache hashes source files by relative path. If a missing file
        returned a placeholder instead of raising, a mistyped or moved path
        would make the cache quietly stop noticing code changes -- and stale
        outputs would be served indefinitely with no error anywhere.
        """
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_path / "absent.py")


class TestComputeExcelSheetsHash:
    def test_a_rewritten_workbook_with_identical_values_is_cache_stable(self, tmp_path):
        """The property the whole text-fixture approach rests on.

        Fixtures are stored as text and rebuilt into .xlsx at test time, so two
        builds produce byte-different files. Because the hash is taken over
        sheet *values*, that does not invalidate the cache.
        """
        frame = pd.DataFrame({"country": ["FI", "SE"], "capacity": [1.0, 2.0]})
        first = _write_xlsx(tmp_path / "first.xlsx", {"unitdata": frame})
        # Same unitdata values, but a genuinely different file: an extra sheet
        # that the prefix does not select. Rewriting the identical frame alone
        # would not do -- openpyxl is deterministic enough to produce identical
        # bytes, which would make the test prove nothing.
        second = _write_xlsx(
            tmp_path / "second.xlsx",
            {"unitdata": frame, "notes": pd.DataFrame({"comment": ["irrelevant"]})},
        )

        assert first.read_bytes() != second.read_bytes()  # different files...
        assert compute_excel_sheets_hash(first, "unitdata") == compute_excel_sheets_hash(
            second, "unitdata"
        )  # ...same hash for the sheets that matter

    def test_a_changed_value_changes_the_hash(self, tmp_path):
        original = _write_xlsx(
            tmp_path / "a.xlsx", {"unitdata": pd.DataFrame({"capacity": [1.0]})}
        )
        edited = _write_xlsx(
            tmp_path / "b.xlsx", {"unitdata": pd.DataFrame({"capacity": [2.0]})}
        )
        assert compute_excel_sheets_hash(original, "unitdata") != compute_excel_sheets_hash(
            edited, "unitdata"
        )

    def test_selects_sheets_by_prefix(self, tmp_path):
        path = _write_xlsx(
            tmp_path / "wb.xlsx",
            {
                "unitdata_FI": pd.DataFrame({"a": [1]}),
                "unitdata_SE": pd.DataFrame({"a": [2]}),
                "nodedata": pd.DataFrame({"a": [3]}),
            },
        )
        hashes = compute_excel_sheets_hash(path, "unitdata")
        assert set(hashes) == {"unitdata_FI", "unitdata_SE"}

    def test_returns_empty_for_a_prefix_that_matches_nothing(self, tmp_path):
        path = _write_xlsx(tmp_path / "wb.xlsx", {"nodedata": pd.DataFrame({"a": [1]})})
        assert compute_excel_sheets_hash(path, "unitdata") == {}

    def test_returns_empty_rather_than_raising_for_a_missing_file(self, tmp_path):
        # Unlike compute_file_hash. The asymmetry is deliberate in the source:
        # a missing *data* workbook is a user-config problem reported elsewhere,
        # while a missing *source-code* file is a broken installation.
        assert compute_excel_sheets_hash(tmp_path / "absent.xlsx", "unitdata") == {}


class TestComputeFolderHash:
    def test_is_stable_across_calls(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "b.py").write_text("y = 2", encoding="utf-8")
        assert compute_folder_hash(tmp_path) == compute_folder_hash(tmp_path)

    def test_changes_when_a_file_changes(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text("x = 1", encoding="utf-8")
        before = compute_folder_hash(tmp_path)
        target.write_text("x = 2", encoding="utf-8")
        assert compute_folder_hash(tmp_path) != before

    def test_changes_when_a_file_is_added(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        before = compute_folder_hash(tmp_path)
        (tmp_path / "b.py").write_text("y = 2", encoding="utf-8")
        assert compute_folder_hash(tmp_path) != before

    def test_extension_filter_ignores_other_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        before = compute_folder_hash(tmp_path, extensions=[".py"])
        (tmp_path / "notes.txt").write_text("irrelevant", encoding="utf-8")
        assert compute_folder_hash(tmp_path, extensions=[".py"]) == before
