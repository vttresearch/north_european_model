"""What a processor was given, and whether it is the same as last time.

The record decides which timeseries processors rerun, so two properties matter
more than any single case. It has to be **stable** -- an unchanged input must
produce a byte-identical record, or every build reruns everything and nobody
notices for weeks. And it has to be **honest about what changed**, because the
message it produces is what tells someone which edit woke a processor.
"""

import pandas as pd
import pytest

from src.infrastructure.processor_input_record import (
    MISSING,
    build_record,
    describe_difference,
    file_manifest,
    files_still_match,
    frame_record,
)


def record(frame: pd.DataFrame, scalars: dict | None = None, files=None) -> dict:
    return build_record({"unitdata": frame}, scalars or {}, files)


class TestStability:
    """The property whose failure looks like "it reruns every time"."""

    FRAME = pd.DataFrame({"flow": ["onshore", "PV"], "node": ["FI00_elec", "SE01_elec"]})

    def test_the_same_frame_records_the_same_thing(self):
        assert frame_record(self.FRAME) == frame_record(self.FRAME.copy())

    def test_row_order_does_not_matter(self):
        shuffled = self.FRAME.iloc[::-1].reset_index(drop=True)
        assert frame_record(self.FRAME) == frame_record(shuffled)

    def test_column_order_does_not_matter(self):
        swapped = self.FRAME[["node", "flow"]]
        assert frame_record(self.FRAME) == frame_record(swapped)

    def test_a_float_survives_the_round_trip(self):
        """Formatting instead of repr would make 0.1 compare unequal to itself."""
        frame = pd.DataFrame({"node": ["FI00"], "limit": [0.1 + 0.2]})
        once, twice = frame_record(frame), frame_record(frame.copy())
        assert once == twice
        assert once["rows"][0][0] == repr(0.1 + 0.2)

    def test_provenance_columns_are_left_out(self):
        """They carry the file and sheet a row came from, so keeping them would
        rerun every processor when a workbook is renamed and nothing else is."""
        frame = self.FRAME.assign(_source_file="ObservedTrends.xlsx")
        assert frame_record(frame) == frame_record(self.FRAME)


class TestMissingIsNotZero:
    """`0` and `pd.NA` are distinct in SourceDataPipeline frames, unlike at the
    GDX boundary, and `method=replace` depends on the difference."""

    @pytest.mark.parametrize("value", [0, 0.0, ""])
    def test_a_present_value_is_not_recorded_as_missing(self, value):
        frame = pd.DataFrame({"node": ["FI00"], "limit": [value]})
        assert frame_record(frame)["rows"][0][0] != MISSING

    def test_a_missing_value_is(self):
        frame = pd.DataFrame({"node": ["FI00"], "limit": [pd.NA]})
        assert MISSING in frame_record(frame)["rows"][0]

    def test_zero_and_missing_do_not_compare_equal(self):
        zero = record(pd.DataFrame({"node": ["FI00"], "limit": [0.0]}))
        absent = record(pd.DataFrame({"node": ["FI00"], "limit": [pd.NA]}))
        assert describe_difference(zero, absent) is not None


class TestWhatItSaysChanged:
    COLUMNS = ["country", "twh/year"]

    def test_an_unchanged_input_says_nothing(self):
        frame = pd.DataFrame({"country": ["FI"], "twh/year": [100.0]})
        assert describe_difference(record(frame), record(frame)) is None

    def test_an_edited_value_is_named_with_its_row(self):
        was = pd.DataFrame({"country": ["FI"], "twh/year": [100.0]})
        now = pd.DataFrame({"country": ["FI"], "twh/year": [101.0]})
        assert describe_difference(record(was), record(now)) == (
            "unitdata country=FI twh/year 100.0 -> 101.0"
        )

    def test_a_value_that_moves_the_row_is_still_one_change(self):
        """Rows are sorted, so a change early in the sort order moves the row.
        Compared by position, three rows shuffling along read as three edits."""
        was = pd.DataFrame({"country": ["AT", "BE", "CH"], "twh/year": [1.0, 2.0, 3.0]})
        now = pd.DataFrame({"country": ["ZZ", "BE", "CH"], "twh/year": [1.0, 2.0, 3.0]})
        difference = describe_difference(record(was), record(now))
        assert difference == "unitdata twh/year=1.0 country AT -> ZZ"

    def test_a_new_row_is_named(self):
        was = pd.DataFrame({"country": ["FI"], "twh/year": [100.0]})
        now = pd.DataFrame({"country": ["FI", "SE"], "twh/year": [100.0, 5.0]})
        assert "gained" in describe_difference(record(was), record(now))

    def test_nothing_recorded_yet_is_a_reason_of_its_own(self):
        frame = pd.DataFrame({"country": ["FI"], "twh/year": [100.0]})
        assert describe_difference(None, record(frame)) == "nothing was recorded for it yet"

    def test_a_changed_setting_is_named(self):
        frame = pd.DataFrame({"country": ["FI"], "twh/year": [100.0]})
        was = record(frame, {"rounding_precision": 4})
        now = record(frame, {"rounding_precision": 5})
        assert describe_difference(was, now) == "its settings changed: rounding_precision"


class TestInputFiles:
    def test_a_declared_file_is_recorded_and_matches_itself(self, tmp_path):
        (tmp_path / "a.csv").write_text("x", encoding="utf-8")
        manifest = file_manifest(tmp_path, ("*.csv",))
        assert files_still_match(manifest)

    def test_a_file_appearing_in_a_globbed_folder_is_noticed(self, tmp_path):
        """Topping a PECD folder up with later years is the case that matters:
        it overlaps no hour, so nothing downstream would ever say so."""
        (tmp_path / "a.csv").write_text("x", encoding="utf-8")
        manifest = file_manifest(tmp_path, ("*.csv",))
        (tmp_path / "b.csv").write_text("y", encoding="utf-8")
        assert not files_still_match(manifest)

    def test_an_edited_file_is_noticed(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("x", encoding="utf-8")
        manifest = file_manifest(tmp_path, ("*.csv",))
        path.write_text("xx", encoding="utf-8")
        assert not files_still_match(manifest)

    def test_declaring_nothing_is_cannot_tell_and_never_matches(self, tmp_path):
        """The stated cost of not declaring: rerunning every build. Silently
        assuming "no files" would serve the old GDX after a new download."""
        assert file_manifest(tmp_path, ()) is None
        assert not files_still_match(None)
