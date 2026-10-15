"""The variant checks: ``totals`` and ``coverage``, run on every option of every build.

Expectations: ``tests/expectations/variants.md``, group F. Each test names
its row.

A check merges each option's **check view**: the variant's ``:all`` rows plus
that option's rows, with their methods. Untagged rows never enter a check, so they
are never flagged and never cover anything. A build merges them; a check does not.
"""

import pytest

from tests._common.variants import (
    COVERAGE_MISSING,
    DEMAND_HEADER,
    NODE_HEADER,
    TOTALS_DIFFER,
    UNMATCHED_ARITHMETIC,
    declarations,
    run_variants,
    sheet,
)

# --- totals ------------------------------------------------------------------

DH_ROWS = (
    ("dh-level", "national", "yes", "totals"),
    ("dh-level", "grouped", "", "totals"),
    ("dh-level", "cities", "", "totals"),
)


def dh_demand(*rows: str) -> str:
    return declarations(*DH_ROWS) + sheet("demanddata", DEMAND_HEADER, *rows)


def dh_rows(tre_cities: float = 4, cap: float = 6) -> list[str]:
    """National 10; grouped CAP + TRE; cities HKI 6 + TRE. Consistent by default."""
    return [
        "FI00 | dheat |     | all | 1 | 10 |  | dh-level:national",
        "FI00 | dheat | HKI | all | 1 | 6  |  | dh-level:cities",
        f"FI00 | dheat | TRE | all | 1 | {tre_cities} |  | dh-level:cities",
        "FI00 | dheat | TRE | all | 1 | 4  |  | dh-level:grouped",
        f"FI00 | dheat | CAP | all | 1 | {cap} |  | dh-level:grouped",
    ]


def check(tmp_path, book, build="", variants=None):
    """One build; returns the pipeline and the logger."""
    return run_variants(tmp_path, {"book.xlsx": book},
                        variants=variants or {"": {}}, build=build)


class TestTotals:
    def test_consistent_options_are_silent(self, tmp_path):
        _, logger = check(tmp_path, dh_demand(*dh_rows()))
        logger.assert_not_logged(TOTALS_DIFFER)

    def test_a_broken_option_shows_in_a_build_that_does_not_use_it(self, tmp_path):
        # F1: 'grouped' is 9, the default 'national' build still says so.
        _, logger = check(tmp_path, dh_demand(*dh_rows(cap=5)))
        logger.assert_logged(TOTALS_DIFFER, level="warn")
        assert any("grouped" in m for m in logger.matching(TOTALS_DIFFER))

    def test_the_message_names_country_grid_options_and_sums(self, tmp_path):
        # F5: one city's row missing from 'cities'.
        _, logger = check(tmp_path, dh_demand(*dh_rows(tre_cities=3)))
        message = " ".join(logger.matching(TOTALS_DIFFER))
        for part in ("FI00", "dheat", "national", "cities", "10", "9"):
            assert part in message, part

    @pytest.mark.parametrize("tre, warns", [(3.995, False), (3.98, True)])
    def test_demand_totals_allow_a_tenth_of_a_percent(self, tmp_path, tre, warns):
        # F4: 10 against 9.995 is 0.05 %, against 9.98 is 0.2 %.
        _, logger = check(tmp_path, dh_demand(*dh_rows(tre_cities=tre)))
        assert bool(logger.matching(TOTALS_DIFFER)) is warns

    def test_unit_capacity_totals_per_grid(self, tmp_path):
        # F3: national 300 MW of heat against cities 200 + 90.
        units = declarations(*DH_ROWS) + sheet(
            "unittypedata", "scenario | year | unittype | grid_output1 | eff00",
            "all | 1 | HOB | dheat | 0.9",
        ) + sheet(
            "unitdata",
            "Country | unittype | unit_name_prefix | node_suffix_output1 | Scenario | Year | capacity_output1 | variant",
            "FI00 | HOB |     |     | all | 1 | 300 | dh-level:national",
            "FI00 | HOB | HKI | HKI | all | 1 | 200 | dh-level:cities,grouped",
            "FI00 | HOB | TRE | TRE | all | 1 | 90  | dh-level:cities,grouped",
        )
        _, logger = check(tmp_path, units)
        message = " ".join(logger.matching(TOTALS_DIFFER))
        for part in ("FI00", "dheat", "300", "290"):
            assert part in message, part

    def test_a_country_where_every_option_is_the_same_is_silent(self, tmp_path):
        # F6: SE01 has only an ':all' row.
        book = dh_demand(*dh_rows(), "SE01 | dheat |  | all | 1 | 3 |  | dh-level:all")
        _, logger = check(tmp_path, book)
        logger.assert_not_logged(TOTALS_DIFFER)

    def test_only_the_options_a_country_has_are_compared(self, tmp_path):
        # F3: Finland has no 'tallinn' rows, so it is not compared against it.
        book = declarations(*DH_ROWS, ("dh-level", "tallinn", "", "totals")) + sheet(
            "demanddata", DEMAND_HEADER,
            *dh_rows(),
            "EE00 | dheat |     | all | 1 | 3 |  | dh-level:national",
            "EE00 | dheat | TLL | all | 1 | 3 |  | dh-level:tallinn",
        )
        _, logger = check(tmp_path, book)
        logger.assert_not_logged(TOTALS_DIFFER)

    def test_a_remove_counts_in_the_totals(self, tmp_path):
        # F13: 'grouped' removes the ':all' ROF row, the others keep it.
        book = dh_demand(
            "FI00 | dheat | ROF | all | 1 | 5 |        | dh-level:all",
            *dh_rows(),
            "FI00 | dheat | ROF | all | 1 |   | remove | dh-level:grouped",
        )
        _, logger = check(tmp_path, book)
        logger.assert_logged(TOTALS_DIFFER, level="warn")
        assert any("grouped" in m for m in logger.matching(TOTALS_DIFFER))

    def test_untagged_rows_never_enter_the_totals(self, tmp_path):
        # F2: an untagged FI00 row sits in 'grouped's way in a build, not in a check.
        book = dh_demand(*dh_rows(), "FI00 | dheat | CAP | all | 1 | 50 |  |")
        _, logger = check(tmp_path, book)
        logger.assert_not_logged(TOTALS_DIFFER)


# --- coverage ----------------------------------------------------------------

BIO = declarations(
    ("bio-price", "low", "", "coverage"),
    ("bio-price", "ref", "yes", "coverage"),
    ("bio-price", "high", "", "coverage"),
)


def prices(*rows: str) -> str:
    return BIO + sheet("nodedata", NODE_HEADER, *rows)


def coverage_messages(logger) -> str:
    return " ".join(logger.matching(COVERAGE_MISSING))


class TestCoverage:
    def test_every_option_resolving_every_key_is_silent(self, tmp_path):
        # F8: 'ref' has no rows of its own and resolves from the ':all' row.
        book = prices(
            "FI00 | biomass | all | 1 | 40  |          | bio-price:all",
            "FI00 | biomass | all | 1 | 1.3 | multiply | bio-price:high",
            "FI00 | biomass | all | 1 | 30  |          | bio-price:low",
        )
        _, logger = check(tmp_path, book)
        logger.assert_not_logged(COVERAGE_MISSING)

    def test_a_key_only_one_option_mentions_is_missing_from_the_others(self, tmp_path):
        # F7, F9
        book = prices(
            "FI00 | biomass | all | 1 | 40 |  | bio-price:all",
            "FI00 | gas     | all | 1 | 25 |  | bio-price:high",
        )
        _, logger = check(tmp_path, book)
        logger.assert_logged(COVERAGE_MISSING, level="warn")
        message = coverage_messages(logger)
        for part in ("FI00_gas", "low", "ref"):
            assert part in message, part

    def test_an_untagged_row_covers_nothing(self, tmp_path):
        # F10: the same warning, although the build itself would have gas.
        book = prices(
            "FI00 | biomass | all | 1 | 40 |  | bio-price:all",
            "FI00 | gas     | all | 1 | 20 |  |",
            "FI00 | gas     | all | 1 | 25 |  | bio-price:high",
        )
        _, logger = check(tmp_path, book)
        message = coverage_messages(logger)
        assert "FI00_gas" in message and "ref" in message

    def test_a_multiplier_over_an_untagged_base_covers_nothing(self, tmp_path):
        # F11: a multiply establishes nothing in the check view, so the key is
        # missing from every option, 'high' included.
        book = prices(
            "FI00 | biomass | all | 1 | 40  |          |",
            "FI00 | biomass | all | 1 | 1.3 | multiply | bio-price:high",
        )
        _, logger = check(tmp_path, book)
        message = coverage_messages(logger)
        for part in ("FI00_biomass", "high"):
            assert part in message, part

    def test_tagging_the_base_row_fixes_it(self, tmp_path):
        # F11's way out.
        book = prices(
            "FI00 | biomass | all | 1 | 40  |          | bio-price:all",
            "FI00 | biomass | all | 1 | 1.3 | multiply | bio-price:high",
        )
        _, logger = check(tmp_path, book)
        logger.assert_not_logged(COVERAGE_MISSING)

    def test_an_explicit_remove_covers_its_key(self, tmp_path):
        # F12: removing something is a legitimate scenario.
        book = prices(
            "FI00 | biomass | all | 1 | 40 |        | bio-price:all",
            "FI00 | biomass | all | 1 |    | remove | bio-price:low",
        )
        _, logger = check(tmp_path, book)
        logger.assert_not_logged(COVERAGE_MISSING)

    def test_offenders_are_summarised(self, tmp_path):
        # F15: four countries' gas, three named and one counted.
        book = prices(
            "FI00 | biomass | all | 1 | 40 |  | bio-price:all",
            "all  | gas     | all | 1 | 25 |  | bio-price:high",
        )
        _, logger = check(tmp_path, book)
        assert "1 more" in coverage_messages(logger)


class TestNone:
    def test_nothing_is_checked(self, tmp_path):
        # F14
        book = declarations(
            ("scen", "a", "yes", "none"),
            ("scen", "b", "", "none"),
        ) + sheet(
            "demanddata", DEMAND_HEADER,
            "FI00 | dheat |  | all | 1 | 10 |  | scen:a",
            "FI00 | dheat |  | all | 1 | 20 |  | scen:b",
            "FI00 | elec  |  | all | 1 | 5  |  | scen:b",
        )
        _, logger = check(tmp_path, book)
        logger.assert_not_logged(TOTALS_DIFFER)
        logger.assert_not_logged(COVERAGE_MISSING)
        # What the build does say: each option's rows, and which workbook has them.
        lines = [m for m in logger.matching("check none") if "scen" in m]
        assert lines, logger.messages
        assert "a 1" in lines[0] and "b 2" in lines[0] and "book.xlsx" in lines[0]


class TestWhatChecksLeaveBehind:
    def test_check_merges_are_silent(self, tmp_path):
        # F16: the unmatched multiply is reported once, by the build that merges
        # it, and not once more per option the checks merge.
        book = prices(
            "FI00 | biomass | all | 1 | 40  |          | bio-price:all",
            "FI00 | gas     | all | 1 | 1.3 | multiply | bio-price:high",
        )
        variants = {"": {}, "hi": {"bio-price": {"FI00": "high"}}}
        _, logger = check(tmp_path, book, build="hi", variants=variants)
        assert len(logger.matching(UNMATCHED_ARITHMETIC)) == 1
        _, logger = check(tmp_path, book, build="", variants=variants)
        assert logger.matching(UNMATCHED_ARITHMETIC) == []

    @pytest.mark.parametrize("build", ["", "c"])
    def test_a_failed_check_marks_every_build_partial(self, tmp_path, build):
        # F17
        variants = {"": {}, "c": {"dh-level": {"FI00": "cities"}}}
        pipeline, _ = check(tmp_path, dh_demand(*dh_rows(cap=5)), build=build, variants=variants)
        assert any("dh-level" in cause for cause in pipeline.partial_causes)
