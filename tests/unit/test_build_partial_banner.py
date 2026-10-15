"""The PARTIAL banner at the top of a build's summary.log.

Expectations: ``tests/expectations/variants.md``, G1. A build that warned
and survived a variant problem says so before anything else, because later
warnings in the same folder may only be consequences of it.

``build_input_data.partial_banner(causes)`` returns the banner's lines; the build
writes them first. Imported inside the tests so that this module fails alone
until the helper exists, rather than breaking test_build_input_data.py's import.
"""


def _banner(causes):
    from build_input_data import partial_banner
    return partial_banner(causes)


def test_a_clean_build_has_no_banner():
    assert _banner([]) == []


def test_the_banner_says_partial_first():
    lines = _banner(["dh-level: FI00 has no rows for option tallinn"])
    assert "PARTIAL" in lines[0]


def test_every_cause_is_listed():
    causes = [
        "dh-level: FI00 has no rows for option tallinn",
        "co2-price: emissiondata has no country",
    ]
    text = "\n".join(_banner(causes))
    for cause in causes:
        assert cause in text


def test_it_warns_that_later_warnings_may_follow_from_it():
    text = "\n".join(_banner(["anything"]))
    assert "may follow" in text
