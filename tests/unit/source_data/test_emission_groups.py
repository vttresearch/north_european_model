"""One emission can be priced in more than one group.

``p_nEmission`` gives a node its factor per emission; the price comes from
``ts_emissionPriceChange``, whose rows are keyed ``(emission, group)`` --
``create_ts_emissionPriceChange`` drops duplicates on exactly that pair. A group
is what carries the price, so two groups pricing the same emission are two
rows there.

The source stage merged emissiondata on ``emission`` alone, a narrower key than
its only consumer's. Two groups for one emission therefore collapsed into one
record before the builder ever saw them, and the later row won as though it were
an override. Nothing reported it, because from the merge's point of view that is
what an override looks like.

Latent rather than live: the shipped workbooks price CO2 in one group, and the
two CO2 rows in ``TYNDP-2024_National_Trends.xlsx`` differ by ``year``, which the
whitelist resolves before the merge. Nothing prevented the second group, and
nothing would have said anything when it arrived.
"""

import pandas as pd

from src.source_data.source_data_loader import merge_row_by_row
from tests._common.fixtures import FakeLogger

KEY = ["emission", "group"]


def _emissions(*pairs):
    return pd.DataFrame({
        "emission": pd.Series([e for e, _, _ in pairs], dtype="object"),
        "group": pd.Series([g for _, g, _ in pairs], dtype="object"),
        "price": pd.Series([p for _, _, p in pairs], dtype="Float64"),
        "method": pd.Series(["replace"] * len(pairs), dtype="object"),
    })


class TestTheKeyIsTheEmissionAndItsGroup:
    def test_two_groups_of_one_emission_are_two_records(self):
        merged = merge_row_by_row(
            [_emissions(("co2", "ets-co2", 80.0), ("co2", "national-co2", 120.0))],
            FakeLogger(),
            key_columns=KEY,
        )

        assert len(merged) == 2
        assert set(merged["group"]) == {"ets-co2", "national-co2"}

    def test_neither_price_is_lost(self):
        merged = merge_row_by_row(
            [_emissions(("co2", "ets-co2", 80.0), ("co2", "national-co2", 120.0))],
            FakeLogger(),
            key_columns=KEY,
        )

        prices = dict(zip(merged["group"], merged["price"]))
        assert prices["ets-co2"] == 80.0
        assert prices["national-co2"] == 120.0

    def test_the_same_pair_still_overrides(self):
        """Narrowing must not cost the override the merge exists for."""
        merged = merge_row_by_row(
            [_emissions(("co2", "ets-co2", 80.0)), _emissions(("co2", "ets-co2", 95.0))],
            FakeLogger(),
            key_columns=KEY,
        )

        assert len(merged) == 1
        assert merged.iloc[0]["price"] == 95.0

    def test_two_emissions_in_one_group_stay_separate(self):
        merged = merge_row_by_row(
            [_emissions(("co2", "ets", 80.0), ("ch4", "ets", 30.0))],
            FakeLogger(),
            key_columns=KEY,
        )

        assert len(merged) == 2
