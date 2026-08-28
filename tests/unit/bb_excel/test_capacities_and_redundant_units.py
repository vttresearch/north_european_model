"""Capacity inference and the dropping of units that would do nothing.

``fill_capacities`` fills in a capacity the author did not state, using the
efficiency to relate input to output. It is deliberately conservative -- it only
acts where the answer is unambiguous -- so the tests are as much about the cases
it *declines* as the ones it fills.

``drop_redundant_units`` removes units that can neither run nor be invested in.
Both conditions matter: a unit with no capacity is still meaningful if the model
is allowed to build it.
"""

import math

import pandas as pd
import pytest

from tests._common.bb_excel import gnu_frame, make_pipeline, unit_frame
from tests._common.fixtures import FakeLogger

def _capacity(flat: pd.DataFrame, io_type: str, unit: str = "u1") -> float:
    match = flat[(flat["unit"] == unit) & (flat["input_output"] == io_type)]
    assert len(match) == 1, f"expected one {io_type} row for {unit}, got {len(match)}"
    return float(match.iloc[0]["capacity"])


class TestFillCapacitiesOneInOneOut:
    def _run(self, *, input_cap, output_cap, eff=0.5):
        pipeline = make_pipeline()
        gnu = gnu_frame(
            {"grid": "gas", "node": "FI_gas", "input_output": "input", "capacity": input_cap},
            {"grid": "elec", "node": "FI_elec", "input_output": "output", "capacity": output_cap},
        )
        unit = unit_frame({"eff00": eff})
        return pipeline.fill_capacities(gnu, unit)

    def test_a_missing_input_is_derived_from_the_output(self):
        # 100 MW out at 50% efficiency needs 200 MW of fuel in.
        flat = self._run(input_cap=0, output_cap=100, eff=0.5)
        assert _capacity(flat, "input") == 200.0

    def test_a_missing_output_is_derived_from_the_input(self):
        flat = self._run(input_cap=200, output_cap=0, eff=0.5)
        assert _capacity(flat, "output") == 100.0

    def test_the_result_is_rounded_up_to_one_decimal(self):
        """``math.ceil(x * 10) / 10`` -- rounded *up*, deliberately.

        Rounding a derived input capacity down would make the unit unable to
        reach the output capacity it was given, which is a silent de-rating.
        """
        flat = self._run(input_cap=0, output_cap=100, eff=0.3)
        assert flat is not None
        expected = math.ceil(100 / 0.3 * 10) / 10   # 333.4, not 333.3
        assert _capacity(flat, "input") == expected

    def test_nothing_is_touched_when_both_are_given(self):
        flat = self._run(input_cap=180, output_cap=100, eff=0.5)
        assert _capacity(flat, "input") == 180.0     # the author's number stands
        assert _capacity(flat, "output") == 100.0

    def test_nothing_is_derived_when_both_are_missing(self):
        # There is no anchor to derive from; inventing one would be a guess.
        flat = self._run(input_cap=0, output_cap=0, eff=0.5)
        assert _capacity(flat, "input") == 0.0
        assert _capacity(flat, "output") == 0.0

    @pytest.mark.parametrize("eff", [0, None])
    def test_nothing_is_derived_without_a_usable_efficiency(self, eff):
        # Dividing by a zero or absent efficiency is meaningless, so the unit
        # is skipped rather than filled with an infinity.
        flat = self._run(input_cap=0, output_cap=100, eff=eff)
        assert _capacity(flat, "input") == 0.0


class TestFillCapacitiesMultipleOutputs:
    def _run(self, *, cv=None, output_caps=(100, 50), eff=0.8):
        pipeline = make_pipeline()
        rows = [
            {"grid": "bio", "node": "FI_bio", "input_output": "input", "capacity": 0},
        ]
        for index, cap in enumerate(output_caps):
            row = {
                "grid": f"out{index}",
                "node": f"FI_out{index}",
                "input_output": "output",
                "capacity": cap,
            }
            if cv is not None:
                row["cv"] = cv if index == 0 else 0
            rows.append(row)
        gnu = gnu_frame(*rows)
        unit = unit_frame({"eff00": eff})
        return pipeline.fill_capacities(gnu, unit)

    def test_the_input_is_derived_from_the_total_of_the_outputs(self):
        flat = self._run(eff=0.8, output_caps=(100, 50))
        assert _capacity(flat, "input") == math.ceil(150 / 0.8 * 10) / 10

    def test_nothing_is_derived_when_cv_is_set(self):
        """A ``cv`` unit trades heat against power, so its outputs do not add up.

        Summing them would overstate the fuel input. This is the rule the CHP
        route fixture exercises end to end -- its input capacity stays 0.
        """
        flat = self._run(cv=0.15)
        assert _capacity(flat, "input") == 0.0

    def test_nothing_is_derived_when_an_output_capacity_is_missing(self):
        # The total would be wrong, so the derived input would be too.
        flat = self._run(output_caps=(100, 0))
        assert _capacity(flat, "input") == 0.0

    def test_a_stated_input_is_never_overwritten(self):
        pipeline = make_pipeline()
        gnu = gnu_frame(
            {"grid": "bio", "node": "FI_bio", "input_output": "input", "capacity": 999},
            {"grid": "a", "node": "FI_a", "input_output": "output", "capacity": 100},
            {"grid": "b", "node": "FI_b", "input_output": "output", "capacity": 50},
        )
        flat = pipeline.fill_capacities(gnu, unit_frame({"eff00": 0.8}))
        assert _capacity(flat, "input") == 999.0


class TestFillCapacitiesEdges:
    def test_an_empty_input_frame_returns_empty(self):
        pipeline = make_pipeline()
        assert pipeline.fill_capacities(pd.DataFrame(), pd.DataFrame()).empty

    def test_units_are_filled_independently(self):
        pipeline = make_pipeline()
        gnu = gnu_frame(
            {"unit": "u1", "grid": "gas", "node": "FI_gas", "input_output": "input", "capacity": 0},
            {"unit": "u1", "grid": "elec", "node": "FI_elec", "input_output": "output", "capacity": 100},
            {"unit": "u2", "grid": "gas", "node": "FI_gas", "input_output": "input", "capacity": 0},
            {"unit": "u2", "grid": "elec", "node": "FI_elec", "input_output": "output", "capacity": 0},
        )
        unit = unit_frame({"unit": "u1", "eff00": 0.5}, {"unit": "u2", "eff00": 0.5})
        flat = pipeline.fill_capacities(gnu, unit)
        assert _capacity(flat, "input", "u1") == 200.0
        assert _capacity(flat, "input", "u2") == 0.0     # nothing to derive from


class TestDropRedundantUnits:
    def _run(self, gnu_extra=None, unit_extra=None):
        logger = FakeLogger()
        pipeline = make_pipeline(logger=logger)
        gnu = gnu_frame({"capacity": 0, **(gnu_extra or {})})
        unit = unit_frame({**(unit_extra or {})})
        gnu_out, unit_out = pipeline.drop_redundant_units(gnu, unit)
        return (
            gnu_out,
            unit_out,
            logger,
        )

    def test_a_unit_with_no_capacity_and_no_investment_is_dropped(self):
        gnu, unit, logger = self._run()
        assert gnu.empty and unit.empty
        logger.assert_logged("Dropped 1 unit(s)", level="skip")

    def test_a_unit_with_capacity_is_kept(self):
        gnu, unit, _ = self._run(gnu_extra={"capacity": 100})
        assert len(gnu) == 1 and len(unit) == 1

    def test_a_unit_with_investment_costs_is_kept(self):
        """Zero capacity is not redundant if the model may build the unit.

        This is the case the two conditions exist to separate: an investment
        candidate legitimately starts at zero capacity.
        """
        gnu, unit, _ = self._run(gnu_extra={"invCosts": 1000})
        assert len(gnu) == 1

    def test_a_unit_with_a_max_unit_count_is_kept(self):
        gnu, unit, _ = self._run(unit_extra={"maxUnitCount": 5})
        assert len(gnu) == 1

    def test_only_the_redundant_unit_is_dropped(self):
        logger = FakeLogger()
        pipeline = make_pipeline(logger=logger)
        gnu = gnu_frame(
            {"unit": "keeper", "capacity": 100},
            {"unit": "dropme", "capacity": 0},
        )
        unit = unit_frame({"unit": "keeper"}, {"unit": "dropme"})
        gnu_out, unit_out = pipeline.drop_redundant_units(gnu, unit)

        assert gnu_out["unit"].tolist() == ["keeper"]
        assert unit_out["unit"].tolist() == ["keeper"]

    def test_a_unit_is_kept_when_any_of_its_connections_has_capacity(self):
        # A CHP whose heat output is sized but whose input is not must survive.
        logger = FakeLogger()
        pipeline = make_pipeline(logger=logger)
        gnu = gnu_frame(
            {"grid": "bio", "node": "FI_bio", "input_output": "input", "capacity": 0},
            {"grid": "heat", "node": "FI_heat", "input_output": "output", "capacity": 50},
        )
        gnu_out, _ = pipeline.drop_redundant_units(
            gnu, unit_frame({"unit": "u1"})
        )
        assert len(gnu_out) == 2

    def test_empty_frames_pass_through(self):
        pipeline = make_pipeline()
        gnu, unit = pipeline.drop_redundant_units(pd.DataFrame(), pd.DataFrame())
        assert gnu.empty and unit.empty

    def test_dropping_is_reported_once_and_names_the_units(self):
        # Units vanishing from the model is worth reporting -- silently shrinking
        # the model is how a scenario ends up smaller than intended -- but on one
        # line, naming up to three of them. A missing input file can leave a
        # hundred units with no capacity, and a hundred lines is a hundred lines
        # nobody reads.
        _, _, logger = self._run()
        lines = logger.matching("Dropped")
        assert len(lines) == 1
        assert "u1" in lines[0]
