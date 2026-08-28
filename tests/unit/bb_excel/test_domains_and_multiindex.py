"""Domain compilation and the GDXXRW fake MultiIndex.

``compile_domain_df`` is where the project's identity convention actually lives:
**case-insensitive identity, first-presented form preserved**. Production
suffixes are uppercase city codes (HKI, TKU, TRE, ESP, VAN, OUL, JKL), so
folding labels to lower case is not an option -- the first spelling seen has to
win instead.

``create_fake_MultiIndex`` is the format contract with GDXXRW: a second header
row, blank in the dimension columns and repeating the parameter name elsewhere.
That is pinned exactly, being one of the five cases where pinning a value is
correct. It runs once per sheet in write_workbook and nothing undoes it, so what
these tests pin is a one-way transform.
"""

import pandas as pd

from src.bb_excel.bb_excel_tables import compile_domain_df
from src.bb_excel.bb_excel_writer import create_fake_MultiIndex



class TestCompileDomainDf:
    def test_sorts_case_insensitively(self):
        out = compile_domain_df(["beta", "Alpha", "gamma"], "node")
        assert out["node"].tolist() == ["Alpha", "beta", "gamma"]

    def test_deduplicates_exact_repeats(self):
        out = compile_domain_df(["FI_elec", "FI_elec"], "node")
        assert out["node"].tolist() == ["FI_elec"]

    def test_labels_differing_only_in_case_collapse_to_one(self):
        """GAMS treats these as one label, so the domain must too.

        Writing both would give GDXXRW two members for a single set element.
        """
        out = compile_domain_df(["FI_heat_dh", "FI_heat_DH"], "node")
        assert len(out) == 1

    def test_the_first_spelling_seen_is_the_one_kept(self):
        """The convention, stated directly.

        Not lower case, not upper case -- whichever form appeared first. That is
        what lets HKI stay HKI while still being the same node as hki.
        """
        assert compile_domain_df(["HKI", "hki"], "node")["node"].tolist() == ["HKI"]
        assert compile_domain_df(["hki", "HKI"], "node")["node"].tolist() == ["hki"]

    def test_uppercase_labels_survive_untouched(self):
        # The reason the rule is first-form-wins rather than casefold: these are
        # real node suffixes and renaming them would break every reference.
        codes = ["FI00_dheat_HKI", "FI00_dheat_TKU", "FI00_dheat_TRE"]
        assert compile_domain_df(codes, "node")["node"].tolist() == sorted(codes)

    def test_an_empty_list_gives_an_empty_frame(self):
        assert compile_domain_df([], "node").empty

    def test_non_string_values_are_ignored(self):
        # Domain members are GAMS set elements; a stray number is not one.
        out = compile_domain_df([None, 42, "FI_elec"], "node")
        assert out["node"].tolist() == ["FI_elec"]

    def test_a_list_with_nothing_usable_gives_an_empty_frame(self):
        assert compile_domain_df([None, 42], "node").empty

    def test_the_column_is_named_after_the_domain(self):
        assert compile_domain_df(["a"], "unittype").columns.tolist() == ["unittype"]


class TestFakeMultiIndex:
    DIMENSIONS = ["grid", "node"]

    def _flat(self):
        return pd.DataFrame(
            {
                "grid": ["elec", "heat"],
                "node": ["FI_elec", "FI_heat"],
                "capacity": [100.0, 200.0],
                "vomCosts": [1.0, 2.0],
            }
        )

    @staticmethod
    def _is_blank(value) -> bool:
        """Blank in the marker row means empty string in memory, NaN once read back.

        create_fake_MultiIndex writes '', and Excel returns NaN for an empty
        cell -- so the same contract has two spellings depending on which side
        of the workbook you are looking from. Tests on the written file (see
        assert_fake_multiindex) see NaN; this one sees the frame directly.
        """
        return pd.isna(value) or str(value).strip() == ""

    def test_the_marker_row_is_blank_in_dimension_columns(self):
        out = create_fake_MultiIndex(self._flat(), self.DIMENSIONS)
        assert self._is_blank(out.iloc[0]["grid"])
        assert self._is_blank(out.iloc[0]["node"])

    def test_and_repeats_the_name_in_parameter_columns(self):
        # This is what GDXXRW reads as the parameter label, so it is the
        # contract rather than decoration.
        out = create_fake_MultiIndex(self._flat(), self.DIMENSIONS)
        assert out.iloc[0]["capacity"] == "capacity"
        assert out.iloc[0]["vomCosts"] == "vomCosts"

    def test_the_data_is_pushed_down_by_exactly_one_row(self):
        flat = self._flat()
        out = create_fake_MultiIndex(flat, self.DIMENSIONS)
        assert len(out) == len(flat) + 1
        assert out.iloc[1]["node"] == "FI_elec"

    def test_the_values_below_the_marker_row_are_untouched(self):
        # The transform reformats; it must not edit. Everything below row 0 is
        # what the builder produced, in the order it produced it.
        flat = self._flat()
        out = create_fake_MultiIndex(flat, self.DIMENSIONS)
        pd.testing.assert_frame_equal(
            out.iloc[1:].reset_index(drop=True),
            flat.reset_index(drop=True),
            check_dtype=False,
        )

    def test_the_numbers_survive_exactly(self):
        out = create_fake_MultiIndex(self._flat(), self.DIMENSIONS)
        assert out["capacity"].tolist()[1:] == [100.0, 200.0]

    def test_a_frame_with_columns_but_no_rows_becomes_the_marker_row_alone(self):
        # A sheet the model has nothing to say about still needs its parameter
        # label, or GDXXRW has no column dimension to read.
        empty = pd.DataFrame(columns=["grid", "node", "capacity"])
        out = create_fake_MultiIndex(empty, self.DIMENSIONS)
        assert len(out) == 1
        assert out.iloc[0]["capacity"] == "capacity"
