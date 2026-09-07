"""Both ends of a link have to be modelled.

A transfer row names two countries, and a link is only meaningful if the model
carries both of them. ``SourceDataPipeline.run()`` used to say that in two
``apply_whitelist`` calls, one per end -- which also repeated the scenario and
year comparison, and repeated any warning either of them raised.

``apply_whitelist`` already ANDs its filters, so one call carrying both ends is
the same statement. These tests pin the statement rather than the number of
calls, so the merge stays free to say it however it likes.
"""

import pandas as pd

from src.source_data.source_data_loader import apply_whitelist
from tests._common.fixtures import FakeLogger

COUNTRIES = ["FI00", "SE01"]


def _links(*pairs):
    return pd.DataFrame({
        "grid": pd.Series(["elec"] * len(pairs), dtype="object"),
        "from_country": pd.Series([a for a, _ in pairs], dtype="object"),
        "to_country": pd.Series([b for _, b in pairs], dtype="object"),
        "scenario": pd.Series(["all"] * len(pairs), dtype="object"),
        "year": pd.Series([1] * len(pairs), dtype="Float64"),
    })


def _filter(frame):
    logger = FakeLogger()
    filters = {
        "scenario": ["observed trends"],
        "year": [2030],
        "from_country": COUNTRIES,
        "to_country": COUNTRIES,
    }
    return apply_whitelist(frame, filters, logger, "transferdata"), logger


class TestALinkNeedsBothEnds:
    def test_a_link_between_two_modelled_countries_survives(self):
        kept, _ = _filter(_links(("FI00", "SE01")))
        assert len(kept) == 1

    def test_a_link_leaving_the_model_is_dropped(self):
        kept, _ = _filter(_links(("FI00", "DE00")))
        assert kept.empty

    def test_a_link_arriving_from_outside_it_is_dropped(self):
        """The end that used to need a second pass to be checked."""
        kept, _ = _filter(_links(("DE00", "FI00")))
        assert kept.empty

    def test_the_ends_are_judged_independently(self):
        kept, _ = _filter(
            _links(("FI00", "SE01"), ("FI00", "DE00"), ("DE00", "SE01"))
        )
        assert len(kept) == 1
        assert kept.iloc[0]["to_country"] == "SE01"


class TestOneMessagePerProblem:
    def test_a_missing_end_column_is_reported_once(self):
        """Two calls raised the same missing-column warning twice."""
        frame = _links(("FI00", "SE01")).drop(columns=["to_country"])
        _, logger = _filter(frame)

        assert len(logger.matching("missing column", level="warn")) == 1
