"""``add_storage_starts`` -- and what a dropped parameter column does to it.

A storage node needs a starting state, in one of two forms. Where the node data
gives a ``relative`` share, the node starts at that share of its band:
``boundStartRelative`` and a ``relative`` row. Otherwise it starts at a level,
``boundStart`` and a ``reference`` of 70% of a maximum the function looks for in
two places, in order: the node's ``upwardLimit`` constant in
``df_boundarydata``, and then the unit's ``upperLimitCapacityRatio`` times its
capacity.

The unit source is the fragile one, because ``upperLimitCapacityRatio`` is an
ordinary ``PARAM_GNU`` entry and an all-empty parameter column is dropped from
``p_gnu_io`` before this ever runs (``utils.drop_empty_parameter_columns``). A
model where no unit sets it therefore hands this function a frame without the
column -- and the first source having come up empty is precisely when it gets
read.

A share is used only where Backbone can use it: within 0-1, beside a ceiling,
and not with ``boundStartToEnd``. Backbone aborts on each of the others, so here
they fall back to the level and say so.

The p_gn and boundary frames are flat, as every create_*() returns them.
``df_boundarydata`` is a plain source-side table, where 0 and NA still differ.
"""

import pandas as pd
import pytest

from tests._common.bb_excel import make_pipeline
from tests._common.fixtures import FakeLogger



def _boundarydata(*rows: dict) -> pd.DataFrame:
    """The long boundary table, as the source stage hands it over."""
    defaults = {
        "grid": "elec",
        "node": "FI_elec",
        "param_gnboundarytypes": "upwardLimit",
        "useconstant": pd.NA,
        "constant": pd.NA,
        "usetimeseries": pd.NA,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


#: No boundary rows at all -- the node's state is bounded by nothing the table
#: knows about, which is what sends add_storage_starts to its second source.
NO_BOUNDARIES = pd.DataFrame()

#: The warning for a storage node left with no start. Named once, so a negative
#: control cannot drift out of step with the message it is meant to miss.
NO_START = "No storage start level could be determined"


def _shared(share, **upward) -> pd.DataFrame:
    """df_boundarydata with an upwardLimit of 200 and a ``relative`` share."""
    return _boundarydata(
        {"useconstant": 1, "constant": 200.0, **upward},
        {"param_gnboundarytypes": "relative", "useconstant": 1, "constant": share},
    )


def _band(flag: str = "useConstant", node: str = "FI_elec") -> pd.DataFrame:
    """A boundary sheet with a floor and a ceiling, which a share needs.

    The ceiling carries a constant only when it is one; a series row on the
    sheet has its flag and nothing else, as create_p_gnBoundaryPropertiesForStates
    writes it.
    """
    ceiling = {"param_gnBoundaryTypes": "upwardLimit", flag: 1}
    if flag == "useConstant":
        ceiling["constant"] = 200
    return pd.DataFrame([
        {"grid": "elec", "node": node, "param_gnBoundaryTypes": "downwardLimit",
         "useConstant": 1, "constant": "Eps"},
        {"grid": "elec", "node": node, **ceiling},
    ])


def _rows(sheet: pd.DataFrame, boundary_type: str, node: str = "FI_elec") -> pd.DataFrame:
    return sheet[(sheet["node"] == node) & (sheet["param_gnBoundaryTypes"] == boundary_type)]


def _flag(gn: pd.DataFrame, column: str, node: str = "FI_elec"):
    """A p_gn flag, 0 where the column was dropped because no node set it."""
    if column not in gn.columns:
        return 0
    return gn.loc[gn["node"] == node, column].iloc[0]


@pytest.fixture
def logger():
    return FakeLogger()


@pytest.fixture
def pipeline(logger):
    return make_pipeline(logger=logger)


def _p_gn(**overrides) -> pd.DataFrame:
    row = {
        "grid": "elec",
        "node": "FI_elec",
        "isActive": 1,
        "energyStoredPerUnitOfState": 1,
        **overrides,
    }
    return pd.DataFrame([row])


def _boundaries(**overrides) -> pd.DataFrame:
    """A boundary sheet that exists but says nothing about our node's upwardLimit.

    Non-empty matters: the function returns untouched on an empty one, so an
    empty sheet would hide every path below.
    """
    row = {
        "grid": "elec",
        "node": "FI_elec",
        "param_gnBoundaryTypes": "downwardLimit",
        "useConstant": 1,
        "constant": 0,
        **overrides,
    }
    return pd.DataFrame([row])


def _gnu_flat(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{
        "grid": "elec",
        "node": "FI_elec",
        "unit": "u1",
        "input_output": "output",
        "capacity": 100.0,
        **overrides,
    }])


class TestAMissingUpperLimitCapacityRatio:
    def test_a_storage_node_survives_a_gnu_frame_without_the_column(self, pipeline):
        """Regression: this raised a bare KeyError and killed the build.

        Every guard on the way in was about the *frame* -- ``not
        p_gnu_io_flat.empty`` -- and none about the column, so a model whose
        units never set upperLimitCapacityRatio crashed as soon as one node was
        a storage node by some other route.
        """
        gn_out, _ = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(), NO_BOUNDARIES
        )

        assert not gn_out.empty

    def test_a_missing_column_behaves_exactly_like_a_column_with_no_match(self, pipeline):
        """The equivalence the guard is supposed to establish.

        "No unit sets upperLimitCapacityRatio" and "no unit on this node sets it
        above zero" are the same statement about the model, so they must reach
        the same outcome. Asserting the equivalence rather than a literal is what
        keeps this test about the guard: what the two runs produce is the
        no-start-level case, which the class below owns.
        """
        absent_gn, absent_boundaries = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(), NO_BOUNDARIES
        )
        present_gn, present_boundaries = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(upperLimitCapacityRatio=0.0), NO_BOUNDARIES
        )

        pd.testing.assert_frame_equal(
            absent_gn,
            present_gn,
        )
        pd.testing.assert_frame_equal(
            absent_boundaries,
            present_boundaries,
        )

    def test_the_column_is_still_used_when_it_is_there(self, pipeline):
        """Negative control: the guard must not disable the second source.

        capacity 100 * ratio 0.5 = 50, and the reference constant is 70% of it.
        """
        p_gn, boundaries = _p_gn(), _boundaries()

        gn_out, boundary_out = pipeline.add_storage_starts(
            p_gn, boundaries, _gnu_flat(upperLimitCapacityRatio=0.5), NO_BOUNDARIES
        )

        flat = gn_out
        assert flat.loc[flat["node"] == "FI_elec", "boundStart"].iloc[0] == 1

        boundary_flat = boundary_out
        reference = boundary_flat[boundary_flat["param_gnBoundaryTypes"] == "reference"]
        assert len(reference) == 1
        assert reference["constant"].iloc[0] == 35.0


class TestTheUpwardLimitWinsFirst:
    def test_the_boundary_table_constant_is_used_first(self, pipeline):
        # Source 1 short-circuits the other, so the missing gnu column is never
        # reached -- worth pinning, because it is why that went unnoticed.
        _, boundary_out = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(),
            _boundarydata({"useconstant": 1, "constant": 200.0}),
        )

        boundary_flat = boundary_out
        reference = boundary_flat[boundary_flat["param_gnBoundaryTypes"] == "reference"]
        assert reference["constant"].iloc[0] == 140.0

    def test_it_is_read_from_the_table_even_when_the_series_wins_the_sheet(self, pipeline):
        """A node whose limit comes from a series still has a level to start at.

        The sheet carries no constant beside a ``useTimeseries`` row -- exactly
        one flag is written and the number is not one of them -- so reading the
        start level off the sheet would find nothing for precisely the nodes that
        need it most. It comes from ``df_boundarydata``, which keeps both.
        """
        _, boundary_out = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(),
            _boundarydata({"useconstant": 1, "constant": 200.0, "usetimeseries": 1}),
        )

        boundary_flat = boundary_out
        reference = boundary_flat[boundary_flat["param_gnBoundaryTypes"] == "reference"]
        assert reference["constant"].iloc[0] == 140.0

    def test_a_boundary_of_another_type_is_not_mistaken_for_the_limit(self, pipeline):
        # maxSpill says what may leave the node, not how full it starts.
        _, boundary_out = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(),
            _boundarydata({
                "param_gnboundarytypes": "maxSpill", "useconstant": 1, "constant": 300.0
            }),
        )

        boundary_flat = boundary_out
        assert boundary_flat[boundary_flat["param_gnBoundaryTypes"] == "reference"].empty


class TestAStartLevelThatCannotBeDetermined:
    """Both sources missed, so there is no level to start the storage at.

    A 0 is not a level either. Backbone binds a flagged reference of 0, so
    ``boundStart=1`` beside one would start the store empty -- a claim about the
    data that nobody made. Writing nothing leaves the start free, and the log
    names the node, because the fix is in the data.
    """

    def test_nothing_is_written_for_the_node(self, pipeline):
        gn_out, boundaries = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(), NO_BOUNDARIES
        )

        # No start level was found, so nothing bounds the node: 0 = not set, and a
        # flag column no node set is dropped rather than written as zeros.
        assert "boundStart" not in gn_out.columns
        assert "boundStartRelative" not in gn_out.columns

        assert boundaries[boundaries["param_gnBoundaryTypes"] == "reference"].empty

    def test_the_node_is_named_in_a_warning(self, pipeline, logger):
        # The fix is in the user's data, so the message has to say which node
        # and what would bound it.
        pipeline.add_storage_starts(_p_gn(), _boundaries(), _gnu_flat(), NO_BOUNDARIES)

        logger.assert_logged("FI_elec", level="warn")
        logger.assert_logged("upperLimitCapacityRatio", level="warn")

    def test_a_node_that_resolves_is_not_warned_about(self, pipeline, logger):
        # Negative control: the warning must not fire on the ordinary path.
        pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(upperLimitCapacityRatio=0.5), NO_BOUNDARIES
        )

        logger.assert_not_logged(NO_START)

    def test_a_non_storage_node_is_not_warned_about(self, pipeline, logger):
        # Only nodes carrying a state variable are candidates, so a plain node
        # must not produce noise.
        p_gn = pd.DataFrame([{"grid": "elec", "node": "FI_elec", "isActive": 1}])
        pipeline.add_storage_starts(p_gn, _boundaries(), _gnu_flat(), NO_BOUNDARIES)

        logger.assert_not_logged(NO_START)


class TestTheBoundarySheetItLeavesBehind:
    """This function is the last thing to touch p_gnBoundaryPropertiesForStates.

    It appends a 'reference' row per storage node, built from a five-key dict,
    so every other column arrives through the concat as NaN. The fill meant to
    clear that was assigning to the fake-MultiIndex frame while the return value
    was rebuilt from the flat one, so it did nothing at all -- 78 NaN reached an
    OT2030 workbook. Asserted on the frame rather than a written workbook on
    purpose: Excel stores '' as an empty cell, so a read-back cannot tell a
    filled blank from a NaN.
    """

    def test_the_appended_reference_row_carries_no_na(self, pipeline):
        _, boundaries = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(upperLimitCapacityRatio=0.5), NO_BOUNDARIES
        )

        flat = boundaries
        offenders = [c for c in flat.columns if flat[c].isna().any()]
        assert not offenders, f"p_gnBoundaryPropertiesForStates emits NaN in {offenders}"

    def test_an_empty_property_column_is_dropped(self, pipeline):
        # slackCost is set by nothing in this project, so it was written as a
        # column of blanks on every build.
        _, boundaries = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(upperLimitCapacityRatio=0.5), NO_BOUNDARIES
        )

        assert "slackCost" not in boundaries.columns
        assert "useConstant" in boundaries.columns   # the kept column dimension

    def test_a_property_in_use_survives(self, pipeline):
        # Negative control for the drop.
        _, boundaries = pipeline.add_storage_starts(
            _p_gn(), _boundaries( slackCost=250), _gnu_flat(), NO_BOUNDARIES
        )
        assert "slackCost" in boundaries.columns


class TestNodesThatAreNotStorage:
    def test_a_dropped_energy_stored_column_is_tolerated(self, pipeline):
        """``energyStoredPerUnitOfState`` is droppable too, and already guarded.

        Pinned rather than assumed: it is the same class of failure as the one
        above, and the guard is what keeps the storage loop from running at all.
        """
        p_gn = pd.DataFrame([{"grid": "elec", "node": "FI_elec", "isActive": 1}])

        gn_out, _ = pipeline.add_storage_starts(p_gn, _boundaries(), _gnu_flat(), NO_BOUNDARIES)

        # The node survives, and no storage parameter is invented for it.
        assert gn_out["node"].tolist() == ["FI_elec"]
        assert "boundStart" not in gn_out.columns
        assert "boundStartRelative" not in gn_out.columns

    def test_a_share_on_a_node_without_state_is_not_written(self, pipeline, logger):
        """Nothing reads a share on a node with no state, so it is not written.

        The workbook value still reached nothing, which is what a warning is
        for: the reader wrote a number that the model will never see.
        """
        p_gn = pd.DataFrame([{"grid": "elec", "node": "FI_elec", "isActive": 1}])

        gn_out, boundaries = pipeline.add_storage_starts(
            p_gn, _boundaries(), _gnu_flat(), _shared(0.6)
        )

        assert _rows(boundaries, "relative").empty
        assert "boundStartRelative" not in gn_out.columns
        logger.assert_logged("nothing reads it", level="warn")
        logger.assert_logged("FI_elec", level="warn")


class TestAShareStartsTheNodeOnItsBand:
    """A ``relative`` share in the node data: the start follows the band."""

    SHARE = 0.6

    def test_the_share_sets_the_relative_flag_and_not_the_level_flag(self, pipeline):
        gn_out, _ = pipeline.add_storage_starts(
            _p_gn(), _band(), _gnu_flat(), _shared(self.SHARE)
        )

        assert _flag(gn_out, "boundStartRelative") == 1
        # Backbone aborts on a node with both.
        assert _flag(gn_out, "boundStart") == 0

    def test_the_share_reaches_the_sheet_as_given(self, pipeline):
        _, boundaries = pipeline.add_storage_starts(
            _p_gn(), _band(), _gnu_flat(), _shared(self.SHARE)
        )

        relative = _rows(boundaries, "relative")
        assert len(relative) == 1
        assert relative["useConstant"].iloc[0] == 1
        assert relative["constant"].iloc[0] == self.SHARE

    def test_no_level_is_written_beside_it(self, pipeline):
        # The 70% reference is the other form; beside a share it would only
        # look as though it said something.
        _, boundaries = pipeline.add_storage_starts(
            _p_gn(), _band(), _gnu_flat(), _shared(self.SHARE)
        )

        assert _rows(boundaries, "reference").empty

    def test_a_share_of_zero_is_written_with_its_flag(self, pipeline):
        """0 is the floor, not "not set": Backbone reads the share by its flag.

        The one place in this class where a 0 writes a row. Without the row the
        flag would bind nothing, and Backbone would say so at run time.
        """
        gn_out, boundaries = pipeline.add_storage_starts(
            _p_gn(), _band(), _gnu_flat(), _shared(0)
        )

        relative = _rows(boundaries, "relative")
        assert len(relative) == 1
        assert relative["useConstant"].iloc[0] == 1
        assert float(relative["constant"].iloc[0]) == 0
        assert _flag(gn_out, "boundStartRelative") == 1

    def test_a_ceiling_given_only_as_a_series_is_enough(self, pipeline, logger):
        """The case the level form cannot start: a limit with no constant at all.

        The band is read at the starting step, so a series is as good a ceiling
        as a constant.
        """
        series_only = _boundarydata(
            {"usetimeseries": 1},
            {"param_gnboundarytypes": "relative", "useconstant": 1, "constant": self.SHARE},
        )

        gn_out, _ = pipeline.add_storage_starts(
            _p_gn(), _band("useTimeseries"), _gnu_flat(), series_only
        )

        assert _flag(gn_out, "boundStartRelative") == 1
        logger.assert_not_logged(NO_START)

    def test_a_unit_ratio_alone_is_a_ceiling(self, pipeline):
        # Backbone counts the storage of units with upperLimitCapacityRatio as
        # part of the band, so a node with no upwardLimit row still has one.
        share_only = _boundarydata(
            {"param_gnboundarytypes": "relative", "useconstant": 1, "constant": self.SHARE},
        )

        gn_out, _ = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(upperLimitCapacityRatio=0.5), share_only
        )

        assert _flag(gn_out, "boundStartRelative") == 1

    def test_the_appended_share_row_carries_no_na(self, pipeline):
        _, boundaries = pipeline.add_storage_starts(
            _p_gn(), _band(), _gnu_flat(), _shared(self.SHARE)
        )

        offenders = [c for c in boundaries.columns if boundaries[c].isna().any()]
        assert not offenders, f"p_gnBoundaryPropertiesForStates emits NaN in {offenders}"


class TestAShareBackboneCouldNotUse:
    """Each of these would abort Backbone. Here the node starts at a level instead."""

    @pytest.mark.parametrize("share", [-0.1, 1.5, 70])
    def test_a_share_outside_zero_to_one_falls_back_to_a_level(self, pipeline, logger, share):
        gn_out, boundaries = pipeline.add_storage_starts(
            _p_gn(), _band(), _gnu_flat(), _shared(share)
        )

        assert _flag(gn_out, "boundStart") == 1
        assert _flag(gn_out, "boundStartRelative") == 0
        assert _rows(boundaries, "relative").empty
        assert _rows(boundaries, "reference")["constant"].iloc[0] == 140.0
        logger.assert_logged("outside 0-1", level="warn")
        logger.assert_logged("FI_elec", level="warn")

    def test_a_share_with_boundStartToEnd_falls_back_to_a_level(self, pipeline, logger):
        # boundStartToEnd reads the first solve's end from reference, and only
        # under boundStart, so beside a share that end would be left free.
        gn_out, boundaries = pipeline.add_storage_starts(
            _p_gn(boundStartToEnd=1), _band(), _gnu_flat(), _shared(0.6)
        )

        assert _flag(gn_out, "boundStart") == 1
        assert _flag(gn_out, "boundStartRelative") == 0
        assert _rows(boundaries, "relative").empty
        logger.assert_logged("boundStartToEnd", level="warn")

    def test_a_share_with_no_ceiling_leaves_the_start_unbounded(self, pipeline, logger):
        # No band to be a share of, and no maximum for a level either.
        share_only = _boundarydata(
            {"param_gnboundarytypes": "relative", "useconstant": 1, "constant": 0.6},
        )

        gn_out, boundaries = pipeline.add_storage_starts(
            _p_gn(), _boundaries(), _gnu_flat(), share_only
        )

        assert "boundStart" not in gn_out.columns
        assert "boundStartRelative" not in gn_out.columns
        assert _rows(boundaries, "relative").empty
        logger.assert_logged(NO_START, level="warn")
        logger.assert_logged("FI_elec", level="warn")

    def test_a_usable_share_is_not_warned_about(self, pipeline, logger):
        # Negative control for the three above.
        pipeline.add_storage_starts(_p_gn(), _band(), _gnu_flat(), _shared(0.6))

        logger.assert_clean()


class TestTheTwoFormsSideBySide:
    def test_each_node_gets_exactly_one_flag(self, pipeline):
        """A share on one node, a level on the other, in the same model."""
        p_gn = pd.concat([_p_gn(), _p_gn(node="SE_elec")], ignore_index=True)
        sheet = pd.concat([_band(), _band(node="SE_elec")], ignore_index=True)
        boundarydata = pd.concat([
            _shared(0.6),
            _boundarydata({"node": "SE_elec", "useconstant": 1, "constant": 300.0}),
        ], ignore_index=True)

        gn_out, boundaries = pipeline.add_storage_starts(
            p_gn, sheet, _gnu_flat(), boundarydata
        )

        assert _flag(gn_out, "boundStartRelative", "FI_elec") == 1
        assert _flag(gn_out, "boundStart", "FI_elec") == 0
        assert _flag(gn_out, "boundStart", "SE_elec") == 1
        assert _flag(gn_out, "boundStartRelative", "SE_elec") == 0
        assert _rows(boundaries, "reference", "SE_elec")["constant"].iloc[0] == 210.0
        assert _rows(boundaries, "relative", "SE_elec").empty

    def test_a_relative_flag_without_a_share_is_not_kept(self, pipeline):
        """Both flags belong to this function.

        A flag read from the node data with no share beside it would bind
        nothing, and next to the level flag written here it would abort Backbone.
        """
        gn_out, _ = pipeline.add_storage_starts(
            _p_gn(boundStartRelative=1), _band(), _gnu_flat(),
            _boundarydata({"useconstant": 1, "constant": 200.0}),
        )

        assert _flag(gn_out, "boundStart") == 1
        assert _flag(gn_out, "boundStartRelative") == 0
