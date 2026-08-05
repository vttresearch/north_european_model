"""Node-name construction: ``build_node_column`` and ``build_from_to_columns``,
plus the per-connection grid/node columns from ``build_unit_grid_and_node_columns``.

These three functions all turn country/grid/suffix cells into GAMS set element
names, so they must agree about what they tolerate.  Two of the tests below are
regressions for cases where they did not.
"""

import pandas as pd
import pytest

from src.source_data.source_data_loader import (
    build_from_to_columns,
    build_node_column,
    build_unit_grid_and_node_columns,
)
from tests._common.contracts import assert_normalized
from tests._common.fixtures import FakeLogger


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


class TestBuildNodeColumn:
    def test_joins_country_and_grid(self):
        out = build_node_column(_frame(country=["FI"], grid=["elec"]), FakeLogger())
        assert out["node"].tolist() == ["FI_elec"]

    def test_appends_a_node_suffix_when_present(self):
        out = build_node_column(
            _frame(country=["FI"], grid=["heat"], node_suffix=["dh"]), FakeLogger()
        )
        assert out["node"].tolist() == ["FI_heat_dh"]

    def test_omits_the_suffix_when_absent(self):
        out = build_node_column(
            _frame(country=["FI"], grid=["heat"], node_suffix=[pd.NA]), FakeLogger()
        )
        assert out["node"].tolist() == ["FI_heat"]

    @pytest.mark.parametrize("missing", ["country", "grid"])
    def test_returns_empty_and_warns_when_a_required_column_is_missing(self, missing):
        columns = {"country": ["FI"], "grid": ["elec"]}
        del columns[missing]
        logger = FakeLogger()

        out = build_node_column(_frame(**columns), logger)

        # Error-handling policy (CLAUDE.md): after logger init, log and continue
        # with a safe default -- never raise.
        assert out.empty
        logger.assert_logged(missing, level="warn")


class TestBuildFromToColumns:
    def test_joins_both_ends_with_the_grid(self):
        out = build_from_to_columns(
            _frame(from_country=["FI"], to_country=["SE"], grid=["elec"]), FakeLogger()
        )
        assert out["from_node"].tolist() == ["FI_elec"]
        assert out["to_node"].tolist() == ["SE_elec"]

    def test_appends_per_end_suffixes(self):
        out = build_from_to_columns(
            _frame(
                from_country=["FI"],
                to_country=["FI"],
                grid=["heat"],
                from_suffix=["dh"],
                to_suffix=["industry"],
            ),
            FakeLogger(),
        )
        assert out["from_node"].tolist() == ["FI_heat_dh"]
        assert out["to_node"].tolist() == ["FI_heat_industry"]

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(pd.Timestamp("2030-01-01").to_pydatetime(), id="datetime"),
            pytest.param(1.0, id="float"),
            pytest.param(2030, id="int"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_tolerates_a_non_string_country_cell(self, value):
        """Regression: raw '+' concatenation raised TypeError on non-strings.

        Excel silently converts things to dates and numbers, so a country or
        grid cell arriving as something other than text is a user-data problem,
        not an impossible one.  ``build_node_column`` has always coped (it
        stringifies via an f-string); ``build_from_to_columns`` did the same job
        with ``+`` and crashed with a bare TypeError instead of producing a node
        name.  Two functions with one job must tolerate the same input.
        """
        out = build_from_to_columns(
            _frame(from_country=[value], to_country=["SE"], grid=["elec"]), FakeLogger()
        )

        assert len(out) == 1
        assert out["from_node"].iloc[0].endswith("_elec")

    def test_agrees_with_build_node_column_on_the_same_inputs(self):
        """The two functions must name the same node the same way.

        Pinned deliberately: node names are GAMS set elements, so a divergence
        between the two builders silently splits one node into two.
        """
        single = build_node_column(_frame(country=["FI"], grid=["elec"]), FakeLogger())
        pair = build_from_to_columns(
            _frame(from_country=["FI"], to_country=["FI"], grid=["elec"]), FakeLogger()
        )
        assert pair["from_node"].iloc[0] == single["node"].iloc[0]

    def test_returns_empty_and_warns_when_required_columns_are_missing(self):
        logger = FakeLogger()
        out = build_from_to_columns(_frame(from_country=["FI"]), logger)
        assert out.empty
        logger.assert_logged("missing required columns", level="warn")


class TestBuildUnitGridAndNodeColumns:
    UNITTYPES = pd.DataFrame(
        {
            "generator_id": pd.Series(["chp"], dtype="object"),
            "grid_input1": pd.Series(["biomass"], dtype="object"),
            "grid_output1": pd.Series(["elec"], dtype="object"),
        }
    )

    def test_builds_one_grid_and_node_pair_per_declared_connection(self):
        unitdata = pd.DataFrame(
            {
                "country": pd.Series(["FI"], dtype="object"),
                "generator_id": pd.Series(["chp"], dtype="object"),
            }
        )
        out = build_unit_grid_and_node_columns(unitdata, self.UNITTYPES, FakeLogger())

        assert out["grid_input1"].tolist() == ["biomass"]
        assert out["node_input1"].tolist() == ["FI_biomass"]
        assert out["grid_output1"].tolist() == ["elec"]
        assert out["node_output1"].tolist() == ["FI_elec"]

    def test_unmatched_generator_ids_yield_pd_na_not_float_nan(self):
        """Regression: the unmatched cells were seeded with ``np.nan``.

        ``pd.Series(np.nan, ..., dtype=object)`` puts a real float NaN inside an
        object column.  Everything downstream is entitled to assume the
        normalized convention -- ``pd.NA`` for missing in object columns -- and
        code that checks ``value is pd.NA`` or relies on pandas' NA semantics
        behaves differently for a float NaN.  This is exactly the "one function
        returns NaN instead of pd.NA and the next one cannot handle it" failure.
        """
        unitdata = pd.DataFrame(
            {
                "country": pd.Series(["FI"], dtype="object"),
                "generator_id": pd.Series(["not_in_unittypedata"], dtype="object"),
            }
        )

        out = build_unit_grid_and_node_columns(unitdata, self.UNITTYPES, FakeLogger())

        assert out["grid_input1"].iloc[0] is pd.NA
        assert out["node_input1"].iloc[0] is pd.NA
        assert_normalized(out, where="build_unit_grid_and_node_columns, no join match")

    def test_warns_when_unittypedata_declares_no_grid_columns(self):
        logger = FakeLogger()
        unittypes = pd.DataFrame({"generator_id": pd.Series(["chp"], dtype="object")})
        unitdata = pd.DataFrame(
            {
                "country": pd.Series(["FI"], dtype="object"),
                "generator_id": pd.Series(["chp"], dtype="object"),
            }
        )

        build_unit_grid_and_node_columns(unitdata, unittypes, logger)

        logger.assert_logged("grid_input1", level="warn")
