"""``add_storage_starts`` -- and what a dropped parameter column does to it.

A storage node needs a starting state. The function looks for one in three
places, in order: the timeseries upwardLimit, a constant in
``p_gnBoundaryPropertiesForStates``, and finally the unit's
``upperLimitCapacityRatio`` times its capacity.

That third source is the fragile one, because ``upperLimitCapacityRatio`` is an
ordinary ``PARAM_GNU`` entry and an all-empty parameter column is dropped from
``p_gnu_io`` before this ever runs (``utils.drop_empty_parameter_columns``). A
model where no unit sets it therefore hands this function a frame without the
column -- and the first two sources having come up empty is precisely when it
gets read.

The frames here carry the fake MultiIndex because the function drops it itself;
building them any other way would test a shape the pipeline never produces.
"""

import pandas as pd
import pytest

from tests._common.bb_excel import make_pipeline
from tests._common.fixtures import FakeLogger

GN_DIMENSIONS = ["grid", "node"]
BOUNDARY_DIMENSIONS = ["grid", "node", "param_gnBoundaryTypes"]


@pytest.fixture
def logger():
    return FakeLogger()


@pytest.fixture
def pipeline(logger):
    return make_pipeline(logger=logger)


def _p_gn(pipeline, **overrides) -> pd.DataFrame:
    row = {
        "grid": "elec",
        "node": "FI_elec",
        "isActive": 1,
        "energyStoredPerUnitOfState": 1,
        **overrides,
    }
    return pipeline.create_fake_MultiIndex(pd.DataFrame([row]), GN_DIMENSIONS)


def _boundaries(pipeline, **overrides) -> pd.DataFrame:
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
    return pipeline.create_fake_MultiIndex(pd.DataFrame([row]), BOUNDARY_DIMENSIONS)


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
            _p_gn(pipeline), _boundaries(pipeline), _gnu_flat(), {}
        )

        assert not pipeline.drop_fake_MultiIndex(gn_out).empty

    def test_a_missing_column_behaves_exactly_like_a_column_with_no_match(self, pipeline):
        """The equivalence the guard is supposed to establish.

        "No unit sets upperLimitCapacityRatio" and "no unit on this node sets it
        above zero" are the same statement about the model, so they must reach
        the same outcome. Asserting the equivalence rather than a literal keeps
        this test honest about a quirk it does not own: `start_value >= 0`
        accepts zero, so an underivable start still writes boundStart=1 and a
        reference constant of 0. That is pre-existing behaviour, unchanged here,
        and pinning the literal would quietly bless it.
        """
        absent_gn, absent_boundaries = pipeline.add_storage_starts(
            _p_gn(pipeline), _boundaries(pipeline), _gnu_flat(), {}
        )
        present_gn, present_boundaries = pipeline.add_storage_starts(
            _p_gn(pipeline), _boundaries(pipeline), _gnu_flat(upperLimitCapacityRatio=0.0), {}
        )

        pd.testing.assert_frame_equal(
            pipeline.drop_fake_MultiIndex(absent_gn),
            pipeline.drop_fake_MultiIndex(present_gn),
        )
        pd.testing.assert_frame_equal(
            pipeline.drop_fake_MultiIndex(absent_boundaries),
            pipeline.drop_fake_MultiIndex(present_boundaries),
        )

    def test_the_column_is_still_used_when_it_is_there(self, pipeline):
        """Negative control: the guard must not disable the third source.

        capacity 100 * ratio 0.5 = 50, and the reference constant is 70% of it.
        """
        p_gn, boundaries = _p_gn(pipeline), _boundaries(pipeline)

        gn_out, boundary_out = pipeline.add_storage_starts(
            p_gn, boundaries, _gnu_flat(upperLimitCapacityRatio=0.5), {}
        )

        flat = pipeline.drop_fake_MultiIndex(gn_out)
        assert flat.loc[flat["node"] == "FI_elec", "boundStart"].iloc[0] == 1

        boundary_flat = pipeline.drop_fake_MultiIndex(boundary_out)
        reference = boundary_flat[boundary_flat["param_gnBoundaryTypes"] == "reference"]
        assert len(reference) == 1
        assert reference["constant"].iloc[0] == 35.0


class TestTheEarlierSourcesStillWin:
    def test_a_timeseries_upward_limit_is_used_first(self, pipeline):
        # Source 1 short-circuits the other two, so the missing column is never
        # reached -- worth pinning, because it is why this went unnoticed.
        ts_storage_limits = {
            "any": pd.DataFrame([{
                "node": "FI_elec",
                "param_gnBoundaryTypes": "upwardLimit",
                "average_value": 200.0,
            }])
        }

        _, boundary_out = pipeline.add_storage_starts(
            _p_gn(pipeline), _boundaries(pipeline), _gnu_flat(), ts_storage_limits
        )

        boundary_flat = pipeline.drop_fake_MultiIndex(boundary_out)
        reference = boundary_flat[boundary_flat["param_gnBoundaryTypes"] == "reference"]
        assert reference["constant"].iloc[0] == 140.0

    def test_a_boundary_constant_is_used_second(self, pipeline):
        boundaries = _boundaries(pipeline, param_gnBoundaryTypes="upwardLimit", constant=300.0)

        _, boundary_out = pipeline.add_storage_starts(
            _p_gn(pipeline), boundaries, _gnu_flat(), {}
        )

        boundary_flat = pipeline.drop_fake_MultiIndex(boundary_out)
        reference = boundary_flat[boundary_flat["param_gnBoundaryTypes"] == "reference"]
        assert reference["constant"].iloc[0] == 210.0


class TestAStartLevelThatCannotBeDetermined:
    """All three sources missed, so there is no level to start the storage at.

    Backbone gates the bound on the reference constant's own value
    (``3d_setVariableLimits.gms``: ``v_state.fx(...)$p_gnBoundary...('constant')``),
    and a 0 there is indistinguishable from absent -- the project's own
    ``0 = NA`` convention seen from the GAMS side. So writing ``boundStart=1``
    with a 0 reference bound nothing at all, while the workbook showed a flag
    set and a reference row present. ``docs/dictionary.md`` says as much: a 0
    reference "leaves the start free, which can let a storage initialise full".

    Not writing them leaves exactly the same unconstrained model. The change is
    that the log says so.
    """

    def test_nothing_is_written_for_the_node(self, pipeline):
        gn_out, boundaries = pipeline.add_storage_starts(
            _p_gn(pipeline), _boundaries(pipeline), _gnu_flat(), {}
        )

        gn_flat = pipeline.drop_fake_MultiIndex(gn_out)
        assert gn_flat.loc[gn_flat["node"] == "FI_elec", "boundStart"].iloc[0] == 0

        boundary_flat = pipeline.drop_fake_MultiIndex(boundaries)
        assert boundary_flat[boundary_flat["param_gnBoundaryTypes"] == "reference"].empty

    def test_the_node_is_named_in_a_warning(self, pipeline, logger):
        # The fix is in the user's data, so the message has to say which node
        # and what would bound it.
        pipeline.add_storage_starts(_p_gn(pipeline), _boundaries(pipeline), _gnu_flat(), {})

        logger.assert_logged("FI_elec", level="warn")
        logger.assert_logged("upperLimitCapacityRatio", level="warn")

    def test_a_node_that_resolves_is_not_warned_about(self, pipeline, logger):
        # Negative control: the warning must not fire on the ordinary path.
        pipeline.add_storage_starts(
            _p_gn(pipeline), _boundaries(pipeline), _gnu_flat(upperLimitCapacityRatio=0.5), {}
        )

        logger.assert_not_logged("Could not determine a storage start level")

    def test_a_non_storage_node_is_not_warned_about(self, pipeline, logger):
        # Only nodes carrying a state variable are candidates, so a plain node
        # must not produce noise.
        p_gn = pipeline.create_fake_MultiIndex(
            pd.DataFrame([{"grid": "elec", "node": "FI_elec", "isActive": 1}]),
            GN_DIMENSIONS,
        )
        pipeline.add_storage_starts(p_gn, _boundaries(pipeline), _gnu_flat(), {})

        logger.assert_not_logged("Could not determine a storage start level")


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
            _p_gn(pipeline), _boundaries(pipeline), _gnu_flat(upperLimitCapacityRatio=0.5), {}
        )

        flat = pipeline.drop_fake_MultiIndex(boundaries)
        offenders = [c for c in flat.columns if flat[c].isna().any()]
        assert not offenders, f"p_gnBoundaryPropertiesForStates emits NaN in {offenders}"

    def test_an_empty_property_column_is_dropped(self, pipeline):
        # slackCost is set by nothing in this project, so it was written as a
        # column of blanks on every build.
        _, boundaries = pipeline.add_storage_starts(
            _p_gn(pipeline), _boundaries(pipeline), _gnu_flat(upperLimitCapacityRatio=0.5), {}
        )

        assert "slackCost" not in boundaries.columns
        assert "useConstant" in boundaries.columns   # the kept column dimension

    def test_a_property_in_use_survives(self, pipeline):
        # Negative control for the drop.
        _, boundaries = pipeline.add_storage_starts(
            _p_gn(pipeline), _boundaries(pipeline, slackCost=250), _gnu_flat(), {}
        )
        assert "slackCost" in boundaries.columns


class TestNodesThatAreNotStorage:
    def test_a_dropped_energy_stored_column_is_tolerated(self, pipeline):
        """``energyStoredPerUnitOfState`` is droppable too, and already guarded.

        Pinned rather than assumed: it is the same class of failure as the one
        above, and the guard is what keeps the storage loop from running at all.
        """
        p_gn = pipeline.create_fake_MultiIndex(
            pd.DataFrame([{"grid": "elec", "node": "FI_elec", "isActive": 1}]),
            GN_DIMENSIONS,
        )

        gn_out, _ = pipeline.add_storage_starts(p_gn, _boundaries(pipeline), _gnu_flat(), {})

        flat = pipeline.drop_fake_MultiIndex(gn_out)
        assert flat["boundStart"].tolist() == [0]
