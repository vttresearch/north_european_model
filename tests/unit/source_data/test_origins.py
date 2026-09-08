"""Where a value was written down, and how a message says so.

A build's late checks speak about merged tables. ``merge_row_by_row`` drops the
provenance columns, so by then a message can say a node reaches nothing but not
which of forty-odd sheets to open -- which is the state a rename across several
workbooks puts its author in. ``collect_origins`` indexes it while the per-sheet
frames still have it; ``describe_origins`` renders it.

The case-variant behaviour is the point rather than a nicety. A half-finished
rename leaves the same name in two spellings, and naming the spelling that *is*
there is the whole diagnosis.
"""

import pandas as pd

from src.source_data.source_data_loader import collect_origins, describe_origins


def frame(source_file, source_sheet, **columns):
    rows = len(next(iter(columns.values())))
    return pd.DataFrame({
        **{name: pd.Series(values, dtype="object") for name, values in columns.items()},
        "_source_file": pd.Series([source_file] * rows, dtype="object"),
        "_source_sheet": pd.Series([source_sheet] * rows, dtype="object"),
    })


class TestCollectOrigins:
    def test_records_the_sheet_a_value_was_written_in(self):
        origins = collect_origins(
            [frame("a.xlsx", "nodedata", node=["FI00_elec"])], {"node"}
        )
        assert origins == {"fi00_elec": {("FI00_elec", "a.xlsx:nodedata")}}

    def test_connection_suffixes_count_as_the_base_column(self):
        """A unit declares its nodes as node_input1, node_output1 and so on."""
        origins = collect_origins(
            [frame("a.xlsx", "unitdata",
                   node_input1=["FI00_gas"], node_output1=["FI00_elec"])],
            {"node"},
        )
        assert set(origins) == {"fi00_gas", "fi00_elec"}

    def test_several_sheets_accumulate_into_one_index(self):
        origins = collect_origins([frame("a.xlsx", "nodedata", node=["FI00_elec"])],
                                  {"node"})
        collect_origins([frame("b.xlsx", "demanddata", node=["FI00_elec"])],
                        {"node"}, into=origins)
        assert origins["fi00_elec"] == {
            ("FI00_elec", "a.xlsx:nodedata"),
            ("FI00_elec", "b.xlsx:demanddata"),
        }

    def test_a_frame_with_no_provenance_is_skipped(self):
        """Rather than indexing every value under an empty source."""
        bare = pd.DataFrame({"node": pd.Series(["FI00_elec"], dtype="object")})
        assert collect_origins([bare], {"node"}) == {}

    def test_columns_the_caller_did_not_ask_for_are_left_alone(self):
        origins = collect_origins(
            [frame("a.xlsx", "unitdata", node=["FI00_elec"], unittype=["CHPbio"])],
            {"node"},
        )
        assert set(origins) == {"fi00_elec"}


class TestDescribeOrigins:
    ORIGINS = {
        "fi00_elec": {("FI00_elec", "a.xlsx:nodedata")},
        "at00_gas": {("AT00_gas", "a.xlsx:nodedata")},
        "spread": {("Spread", "a.xlsx:one"), ("Spread", "b.xlsx:two"),
                   ("Spread", "c.xlsx:three")},
    }

    def test_names_the_sheet(self):
        assert describe_origins("FI00_elec", self.ORIGINS) == " (a.xlsx:nodedata)"

    def test_says_nothing_when_nothing_recorded_the_value(self):
        # Empty, not "(unknown)": a message that admits it cannot say where
        # something came from has spent a line saying nothing.
        assert describe_origins("FI00_heat", self.ORIGINS) == ""

    def test_a_case_variant_is_reported_as_a_different_spelling(self):
        """The diagnosis of a half-finished rename.

        Nothing matched `AT00_Gas`, and the reason is that the sheet holding it
        says `AT00_gas`. Naming the sheet alone would look like a contradiction --
        'nothing declares this node' beside 'here is the sheet that declares it'.
        """
        said = describe_origins("AT00_Gas", self.ORIGINS)
        assert said == " (spelled 'AT00_gas' in a.xlsx:nodedata)"

    def test_an_exact_match_wins_over_a_case_variant(self):
        origins = {"n": {("N", "a.xlsx:one"), ("n", "b.xlsx:two")}}
        assert describe_origins("n", origins) == " (b.xlsx:two)"

    def test_more_sheets_than_the_limit_are_counted(self):
        said = describe_origins("Spread", self.ORIGINS, limit=2)
        assert said == " (a.xlsx:one, b.xlsx:two and 1 more)"
