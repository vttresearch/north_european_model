"""Shifting a capacity-factor profile's mean without breaking it.

Scaling a capacity factor is not multiplication. A CF lives in [0, 1], and
multiplying a profile that already peaks near 1 by 1.2 produces values above 1 --
a turbine generating more than its rated power, which the optimiser will happily
exploit.

``_apply_logit_scaling`` moves the profile into logit space, shifts it there, and
maps back. The bounds are preserved by construction because the inverse logit
cannot leave (0, 1), and the shape is preserved because every hour moves by the
same amount in latent space. The offset itself is found by binary search, so the
mean lands near the target rather than exactly on it.

Assertions here are mostly properties -- bounded, monotonic, shape-preserving --
rather than pinned numbers, since the search tolerance is an implementation
detail.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from tests._common.fixtures import FakeLogger

_spec = importlib.util.spec_from_file_location(
    "VRE_PECD", "src/timeseries/processors/VRE_PECD.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
VRE_PECD = getattr(_module, "VRE_PECD")


@pytest.fixture(scope="module")
def processor():
    return VRE_PECD(
        input_folder=".",
        country_codes=["FI"],
        start_year=2014,
        end_year=2014,
        attached_grid="elec",
        logger=FakeLogger(),
    )


def _profile(values) -> pd.Series:
    return pd.Series([float(v) for v in values])


#: A profile with a realistic spread, including a calm spell and a near-rated hour.
WIND = _profile([0.0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 0.4, 0.1])


class TestBoundsArePreserved:
    @pytest.mark.parametrize("scaling", [0.5, 0.9, 1.1, 1.5, 2.0])
    def test_the_result_never_leaves_the_unit_interval(self, processor, scaling):
        """The reason the transform exists.

        Plain multiplication by 1.5 would put the 0.95 hour at 1.425 -- a
        turbine above rated power, which the optimiser treats as free capacity.
        """
        out = processor._apply_logit_scaling(WIND, scaling)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_an_already_high_profile_can_still_be_scaled_up(self, processor):
        high = _profile([0.8, 0.85, 0.9, 0.95, 0.99])
        out = processor._apply_logit_scaling(high, 1.15)
        assert out.max() <= 1.0
        assert out.mean() > high.mean()

    def test_exact_zeros_stay_zero(self, processor):
        # A calm hour is calm at any scaling; the transform restores hard 0s
        # that the epsilon clipping would otherwise nudge above zero.
        out = processor._apply_logit_scaling(WIND, 1.5)
        assert out.iloc[0] == 0.0

    def test_exact_ones_stay_one(self, processor):
        full = _profile([1.0, 0.5, 0.25])
        out = processor._apply_logit_scaling(full, 0.8)
        assert out.iloc[0] == 1.0


class TestTheMeanMoves:
    @pytest.mark.parametrize("scaling", [0.6, 0.8, 1.2, 1.4])
    def test_towards_the_requested_multiple(self, processor, scaling):
        # Binary search, so "close to" rather than "equal to" -- the tolerance
        # is an implementation detail and should not be pinned tighter.
        out = processor._apply_logit_scaling(WIND, scaling)
        assert out.mean() == pytest.approx(WIND.mean() * scaling, rel=0.02)

    def test_scaling_by_one_leaves_the_profile_effectively_unchanged(self, processor):
        out = processor._apply_logit_scaling(WIND, 1.0)
        assert out.mean() == pytest.approx(WIND.mean(), rel=1e-3)

    def test_scaling_up_raises_every_hour_and_scaling_down_lowers_it(self, processor):
        # A uniform shift in latent space, so no hour moves against the others.
        interior = WIND[(WIND > 0) & (WIND < 1)]
        up = processor._apply_logit_scaling(WIND, 1.3)[interior.index]
        down = processor._apply_logit_scaling(WIND, 0.7)[interior.index]
        assert (up >= interior).all()
        assert (down <= interior).all()


class TestShapeIsPreserved:
    def test_the_ordering_of_hours_is_unchanged(self, processor):
        """Scaling changes how windy it was, not when.

        Reordering would decorrelate the profile from every other timeseries in
        the model -- the same class of damage as a time shift.
        """
        out = processor._apply_logit_scaling(WIND, 1.4)
        assert list(WIND.rank()) == list(out.rank())

    def test_the_index_is_preserved(self, processor):
        indexed = pd.Series([0.2, 0.5, 0.8], index=pd.date_range("2014-01-01", periods=3, freq="h"))
        out = processor._apply_logit_scaling(indexed, 1.2)
        assert out.index.equals(indexed.index)

    def test_the_length_is_preserved(self, processor):
        assert len(processor._apply_logit_scaling(WIND, 1.5)) == len(WIND)


class TestEdgeCases:
    def test_an_all_zero_profile_is_returned_untouched(self, processor):
        """Nothing to scale, and the target mean would be 0 either way.

        Returning early also avoids a division by zero in the logit.
        """
        calm = _profile([0.0, 0.0, 0.0])
        out = processor._apply_logit_scaling(calm, 2.0)
        assert list(out) == [0.0, 0.0, 0.0]

    def test_a_constant_profile_stays_constant(self, processor):
        flat = _profile([0.4] * 5)
        out = processor._apply_logit_scaling(flat, 1.25)
        assert out.nunique() == 1
        assert out.iloc[0] == pytest.approx(0.5, rel=0.02)

    def test_a_single_hour_profile_is_handled(self, processor):
        out = processor._apply_logit_scaling(_profile([0.5]), 1.5)
        assert 0.0 <= out.iloc[0] <= 1.0

    def test_scaling_to_zero_drives_the_profile_down_but_not_negative(self, processor):
        out = processor._apply_logit_scaling(WIND, 0.0)
        assert out.min() >= 0.0
        assert out.mean() < WIND.mean()
