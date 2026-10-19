"""Unit availability limits: a time step number in the unit data, a t-label out.

``create_utAvailabilityLimits`` writes Backbone's ``utAvailabilityLimits`` set
from the ``becomeAvailable`` / ``becomeUnavailable`` columns. Backbone derives the
``p_unit`` values of the same names from that set, so the set is the input and
``p_unit`` must not carry them.
"""

import pandas as pd
import pytest

from tests._common.bb_excel import make_pipeline
from tests._common.fixtures import FakeLogger, make_config


def _limits(unitdata_rows, *, units=("u1",), logger=None, days=2):
    """create_utAvailabilityLimits on hand-made frames; the window is `days` long."""
    pipeline = make_pipeline(logger=logger, config=make_config(bb_timeseries_length=days))
    p_unit = pd.DataFrame({"unit": list(units)})
    return pipeline.create_utAvailabilityLimits(p_unit, pd.DataFrame(unitdata_rows))


def _rows(frame):
    return {(r["unit"], r["availabilityLimits"], r["t"]) for r in frame.to_dict("records")}


class TestAStepNumberBecomesATLabel:
    def test_become_available(self):
        out = _limits([{"unit": "u1", "becomeavailable": 13}])
        assert out.to_dict("records") == [
            {"unit": "u1", "t": "t000013", "availabilityLimits": "becomeAvailable"}
        ]

    def test_both_limits_on_one_unit_are_two_rows(self):
        out = _limits([{"unit": "u1", "becomeavailable": 13, "becomeunavailable": 30}])
        assert _rows(out) == {
            ("u1", "becomeAvailable", "t000013"),
            ("u1", "becomeUnavailable", "t000030"),
        }

    def test_one_is_the_first_hour_of_the_window(self):
        # Counted the way t_start is: 1 is t000001, not t000000.
        out = _limits([{"unit": "u1", "becomeavailable": 1}])
        assert out["t"].tolist() == ["t000001"]

    def test_the_last_hour_of_the_window_is_allowed(self):
        out = _limits([{"unit": "u1", "becomeunavailable": 48}], days=2)
        assert out["t"].tolist() == ["t000048"]

    def test_the_unit_is_spelled_as_p_unit_spells_it(self):
        # The unit data is matched case-insensitively, as create_p_unit does.
        out = _limits([{"unit": "fi_wind", "becomeavailable": 13}], units=("FI_Wind",))
        assert out["unit"].tolist() == ["FI_Wind"]


class TestNotSet:
    @pytest.mark.parametrize("value", [0, None, pd.NA])
    def test_zero_and_empty_write_nothing(self, value):
        out = _limits([{"unit": "u1", "becomeavailable": value}])
        # No columns either: an empty sheet, as restype is, not a header over nothing.
        assert out.empty and len(out.columns) == 0

    def test_unit_data_without_either_column(self):
        assert _limits([{"unit": "u1", "capacity_output1": 100}]).empty

    def test_no_units(self):
        assert _limits([{"unit": "u1", "becomeavailable": 13}], units=()).empty


class TestOutsideTheWindow:
    @pytest.mark.parametrize("value", [49, -5, 12.5])
    def test_is_reported_and_left_out(self, value):
        logger = FakeLogger()
        out = _limits([{"unit": "u1", "becomeavailable": value}], logger=logger, days=2)
        assert out.empty
        logger.assert_logged("not a time step of the 48 h window", level="warn")
        assert "u1 becomeAvailable" in logger.matching("not a time step")[0]

    def test_a_good_value_beside_a_bad_one_is_still_written(self):
        logger = FakeLogger()
        out = _limits(
            [{"unit": "u1", "becomeavailable": 13, "becomeunavailable": 5000}],
            logger=logger,
        )
        assert _rows(out) == {("u1", "becomeAvailable", "t000013")}
        assert len(logger.matching("not a time step")) == 1


class TestOnlyUnitsInTheModel:
    def test_a_unit_not_in_p_unit_is_not_written(self):
        """Backbone domain-checks the set against `unit`, so a dropped unit's row
        would stop the run rather than be ignored."""
        out = _limits([
            {"unit": "u1", "becomeavailable": 13},
            {"unit": "dropped", "becomeavailable": 13},
        ])
        assert out["unit"].tolist() == ["u1"]


class TestPUnitDoesNotCarryThem:
    def test_p_unit_has_no_availability_columns(self):
        p_unit = make_pipeline().create_p_unit(
            pd.DataFrame([{"unit": "u1", "unittype": "WindOn"}]),
            pd.DataFrame([{"unit": "u1", "becomeavailable": 13, "becomeunavailable": 30}]),
        )
        assert not {"becomeAvailable", "becomeUnavailable"} & set(p_unit.columns)

    def test_a_connection_suffix_is_reported(self):
        # Unit-level like every param_unit column, so a suffix is an error.
        logger = FakeLogger()
        pipeline = make_pipeline(
            logger=logger,
            df_unitdata=pd.DataFrame([{"unit": "u1", "becomeavailable_output1": 13}]),
        )
        pipeline._coerce_numeric_dtypes()
        logger.assert_logged("becomeavailable_output1", level="warn")
