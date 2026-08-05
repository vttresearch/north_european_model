"""Domain compilation and the GDXXRW fake MultiIndex.

``compile_domain_df`` is where the project's identity convention actually lives:
**case-insensitive identity, first-presented form preserved**. Production
suffixes are uppercase city codes (HKI, TKU, TRE, ESP, VAN, OUL, JKL), so
folding labels to lower case is not an option -- the first spelling seen has to
win instead.

``create_fake_MultiIndex`` / ``drop_fake_MultiIndex`` are the format contract
with GDXXRW: a second header row, blank in the dimension columns and repeating
the parameter name elsewhere. That is pinned exactly, being one of the five
cases where pinning a value is correct.
"""

import pandas as pd
import pytest

from tests._common.bb_excel import make_pipeline


@pytest.fixture(scope="module")
def pipeline():
    return make_pipeline()


class TestCompileDomainDf:
    def test_sorts_case_insensitively(self):
        out = make_pipeline().compile_domain_df(["beta", "Alpha", "gamma"], "node")
        assert out["node"].tolist() == ["Alpha", "beta", "gamma"]

    def test_deduplicates_exact_repeats(self, pipeline):
        out = pipeline.compile_domain_df(["FI_elec", "FI_elec"], "node")
        assert out["node"].tolist() == ["FI_elec"]

    def test_labels_differing_only_in_case_collapse_to_one(self, pipeline):
        """GAMS treats these as one label, so the domain must too.

        Writing both would give GDXXRW two members for a single set element.
        """
        out = pipeline.compile_domain_df(["FI_heat_dh", "FI_heat_DH"], "node")
        assert len(out) == 1

    def test_the_first_spelling_seen_is_the_one_kept(self, pipeline):
        """The convention, stated directly.

        Not lower case, not upper case -- whichever form appeared first. That is
        what lets HKI stay HKI while still being the same node as hki.
        """
        assert pipeline.compile_domain_df(["HKI", "hki"], "node")["node"].tolist() == ["HKI"]
        assert pipeline.compile_domain_df(["hki", "HKI"], "node")["node"].tolist() == ["hki"]

    def test_uppercase_labels_survive_untouched(self, pipeline):
        # The reason the rule is first-form-wins rather than casefold: these are
        # real node suffixes and renaming them would break every reference.
        codes = ["FI00_dheat_HKI", "FI00_dheat_TKU", "FI00_dheat_TRE"]
        assert pipeline.compile_domain_df(codes, "node")["node"].tolist() == sorted(codes)

    def test_an_empty_list_gives_an_empty_frame(self, pipeline):
        assert pipeline.compile_domain_df([], "node").empty

    def test_non_string_values_are_ignored(self, pipeline):
        # Domain members are GAMS set elements; a stray number is not one.
        out = pipeline.compile_domain_df([None, 42, "FI_elec"], "node")
        assert out["node"].tolist() == ["FI_elec"]

    def test_a_list_with_nothing_usable_gives_an_empty_frame(self, pipeline):
        assert pipeline.compile_domain_df([None, 42], "node").empty

    def test_the_column_is_named_after_the_domain(self, pipeline):
        assert pipeline.compile_domain_df(["a"], "unittype").columns.tolist() == ["unittype"]


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

    def test_the_marker_row_is_blank_in_dimension_columns(self, pipeline):
        out = pipeline.create_fake_MultiIndex(self._flat(), self.DIMENSIONS)
        assert self._is_blank(out.iloc[0]["grid"])
        assert self._is_blank(out.iloc[0]["node"])

    def test_and_repeats_the_name_in_parameter_columns(self, pipeline):
        # This is what GDXXRW reads as the parameter label, so it is the
        # contract rather than decoration.
        out = pipeline.create_fake_MultiIndex(self._flat(), self.DIMENSIONS)
        assert out.iloc[0]["capacity"] == "capacity"
        assert out.iloc[0]["vomCosts"] == "vomCosts"

    def test_the_data_is_pushed_down_by_exactly_one_row(self, pipeline):
        flat = self._flat()
        out = pipeline.create_fake_MultiIndex(flat, self.DIMENSIONS)
        assert len(out) == len(flat) + 1
        assert out.iloc[1]["node"] == "FI_elec"

    def test_dropping_it_restores_the_original(self, pipeline):
        # Round trip: several methods flatten, edit and rebuild, so a lossy
        # conversion would corrupt the sheet rather than merely reformat it.
        flat = self._flat()
        restored = pipeline.drop_fake_MultiIndex(
            pipeline.create_fake_MultiIndex(flat, self.DIMENSIONS)
        )
        pd.testing.assert_frame_equal(
            restored.reset_index(drop=True),
            flat.reset_index(drop=True),
            check_dtype=False,
        )

    def test_the_round_trip_preserves_values_exactly(self, pipeline):
        flat = self._flat()
        restored = pipeline.drop_fake_MultiIndex(
            pipeline.create_fake_MultiIndex(flat, self.DIMENSIONS)
        )
        assert restored["capacity"].tolist() == [100.0, 200.0]

    def test_an_empty_frame_survives_the_round_trip(self, pipeline):
        empty = pd.DataFrame(columns=["grid", "node", "capacity"])
        out = pipeline.create_fake_MultiIndex(empty, self.DIMENSIONS)
        assert pipeline.drop_fake_MultiIndex(out).empty
