"""Merging a build: the base first, then the chosen options, with every method.

Expectations: ``tests/expectations/variants.md``, group E. Each test names
its row.

Pass 1 merges untagged rows and ``:all`` rows as today. Pass 2 merges the chosen
options' rows on top, so a variant overrides the base whatever the file order.
The arithmetic is ``merge_row_by_row``'s documented truth table (tests/README.md,
pinning case 3), applied in pass 2.
"""

import pytest

from tests._common.variants import (
    DEMAND_HEADER,
    NODE_HEADER,
    UNMATCHED_ARITHMETIC,
    WRITTEN_TWICE,
    declarations,
    demand,
    price,
    rows_where,
    run_variants,
    sheet,
    value_at,
)

BASE, LOW, HIGH = 40, 30, 55

BIO_ROWS = (
    ("bio-price", "low", "", "coverage"),
    ("bio-price", "ref", "yes", "coverage"),
    ("bio-price", "high", "", "coverage"),
)
BIO = declarations(*BIO_ROWS)

BUILDS = {"": {}, "lo": {"bio-price": {"FI00": "low"}}, "hi": {"bio-price": {"FI00": "high"}}}
ALL_HIGH = {"hi": {"bio-price": "high"}}


def prices(*rows: str, name: str = "nodedata") -> str:
    return sheet(name, NODE_HEADER, *rows)


def fi_price(tmp_path, book, build, variants=BUILDS, **kw):
    pipeline, logger = run_variants(tmp_path, book if isinstance(book, dict) else {"book.xlsx": book},
                                    variants=variants, build=build, **kw)
    return price(pipeline, "FI00"), pipeline, logger


class TestPrecedence:
    @pytest.mark.parametrize("build, expected", [("", BASE), ("hi", HIGH)])
    def test_an_option_beats_an_untagged_row_in_a_later_file(self, tmp_path, build, expected):
        # E1: under the old alternatives, the later base row won.
        early = BIO + prices(f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high")
        late = prices(f"FI00 | biomass | all | 1 | {BASE} |  |")
        value, _, _ = fi_price(tmp_path, {"early.xlsx": early, "late.xlsx": late}, build)
        assert value == expected

    @pytest.mark.parametrize("build, expected", [("", BASE), ("hi", HIGH)])
    def test_a_replace_option_over_an_all_row(self, tmp_path, build, expected):
        # E2
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
        )
        value, _, _ = fi_price(tmp_path, book, build)
        assert value == expected

    @pytest.mark.parametrize("build, expected", [("", BASE * 1.5), ("hi", HIGH)])
    def test_an_all_multiply_is_part_of_the_base(self, tmp_path, build, expected):
        # E14: applied in pass 1, like an untagged row, so the option's replace
        # in pass 2 still wins.
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |          |",
            "FI00 | biomass | all | 1 | 1.5    | multiply | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |          | bio-price:high",
        )
        value, _, _ = fi_price(tmp_path, book, build)
        assert value == pytest.approx(expected)


class TestMultiply:
    @pytest.mark.parametrize("build, expected", [("", BASE), ("hi", BASE * 1.3)])
    def test_a_multiplier_option(self, tmp_path, build, expected):
        # E3
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |          | bio-price:all",
            "FI00 | biomass | all | 1 | 1.3    | multiply | bio-price:high",
        )
        value, _, _ = fi_price(tmp_path, book, build)
        assert value == pytest.approx(expected)

    def test_an_all_multiplier_scales_each_countrys_own_base(self, tmp_path):
        # E4
        book = BIO + prices(
            "FI00 | biomass | all | 1 | 40  |          | bio-price:all",
            "SE01 | biomass | all | 1 | 50  |          | bio-price:all",
            "all  | biomass | all | 1 | 1.3 | multiply | bio-price:high",
        )
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": book}, variants=ALL_HIGH, build="hi")
        assert price(pipeline, "FI00") == pytest.approx(40 * 1.3)
        assert price(pipeline, "SE01") == pytest.approx(50 * 1.3)

    def test_a_multiplier_with_nothing_to_scale_creates_nothing(self, tmp_path):
        # E5: today's message, from the build's own merge.
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |          | bio-price:all",
            "FI00 | gas     | all | 1 | 1.3    | multiply | bio-price:high",
        )
        _, pipeline, logger = fi_price(tmp_path, book, "hi")
        logger.assert_logged(UNMATCHED_ARITHMETIC, level="warn")
        assert price(pipeline, "FI00", "gas") is None

    @pytest.mark.parametrize("first, second", [(1.3, 1.1), (1.1, 1.3)])
    def test_two_multipliers_stack_in_either_order(self, tmp_path, first, second):
        # E6
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE}   |          | bio-price:all",
            f"FI00 | biomass | all | 1 | {first}  | multiply | bio-price:high",
            f"FI00 | biomass | all | 1 | {second} | multiply | bio-price:high",
        )
        value, _, _ = fi_price(tmp_path, book, "hi")
        assert value == pytest.approx(BASE * 1.43)

    def test_a_later_replace_beats_an_earlier_multiplier(self, tmp_path):
        # E7: file order inside pass 2. The replace sits in a later file, so this
        # is about precedence, not about a sheet repeating itself.
        early = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |          | bio-price:all",
            "FI00 | biomass | all | 1 | 1.3    | multiply | bio-price:high",
        )
        late = BIO + prices("FI00 | biomass | all | 1 | 60 |  | bio-price:high")
        value, _, _ = fi_price(tmp_path, {"early.xlsx": early, "late.xlsx": late}, "hi")
        assert value == 60


class TestAddAndRemove:
    DEMAND = declarations(
        ("demand", "low", "", "none"),
        ("demand", "ref", "yes", "none"),
        ("demand", "high", "", "none"),
    )
    VARIANTS = {"lo": {"demand": {"FI00": "low"}}, "hi": {"demand": {"FI00": "high"}}}

    def demand_book(self, *rows):
        return self.DEMAND + sheet("demanddata", DEMAND_HEADER, *rows)

    def test_an_add_option(self, tmp_path):
        # E8
        book = self.demand_book(
            "FI00 | elec |  | all | 1 | 7   |     | demand:all",
            "FI00 | elec |  | all | 1 | 0.5 | add | demand:high",
        )
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": book}, variants=self.VARIANTS, build="hi")
        assert demand(pipeline, "FI00_elec") == pytest.approx(7.5)

    def test_an_add_with_nothing_to_add_to_creates_the_record(self, tmp_path):
        # E9
        book = self.demand_book(
            "FI00 | elec  |  | all | 1 | 7   |     | demand:all",
            "FI00 | dheat |  | all | 1 | 0.5 | add | demand:high",
        )
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book}, variants=self.VARIANTS, build="hi")
        logger.assert_logged(UNMATCHED_ARITHMETIC, level="warn")
        assert demand(pipeline, "FI00_dheat") == pytest.approx(0.5)

    def test_add_non_negative_stops_at_zero(self, tmp_path):
        # E10
        book = self.demand_book(
            "FI00 | elec |  | all | 1 | 7   |                  | demand:all",
            "FI00 | elec |  | all | 1 | -10 | add-non-negative | demand:low",
        )
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": book}, variants=self.VARIANTS, build="lo")
        assert demand(pipeline, "FI00_elec") == 0

    NUCLEAR = declarations(
        ("nuclear", "with", "yes", "none"),
        ("nuclear", "without", "", "none"),
    ) + sheet(
        "unittypedata", "scenario | year | unittype | grid_output1 | eff00",
        "all | 1 | SMR | elec | 0.33",
    )

    @pytest.mark.parametrize("build, present", [("", True), ("wo", False)])
    def test_a_remove_option(self, tmp_path, build, present):
        # E11
        book = self.NUCLEAR + sheet(
            "unitdata", "Country | unittype | unit_name_prefix | Scenario | Year | capacity_output1 | method | variant",
            "FI00 | SMR |  | all | 1 | 300 |        | nuclear:all",
            "FI00 | SMR |  | all | 1 |     | remove | nuclear:without",
        )
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": book},
                                   variants={"": {}, "wo": {"nuclear": "without"}}, build=build)
        found = not rows_where(pipeline.df_unitdata, country="FI00", unittype="SMR").empty
        assert found is present

    def test_a_remove_matching_nothing_is_silent(self, tmp_path):
        # E12
        book = self.NUCLEAR + sheet(
            "unitdata", "Country | unittype | unit_name_prefix | Scenario | Year | capacity_output1 | method | variant",
            "SE01 | SMR |  | all | 1 | 300 |        | nuclear:all",
            "FI00 | SMR |  | all | 1 |     | remove | nuclear:without",
        )
        _, logger = run_variants(tmp_path, {"book.xlsx": book},
                                 variants={"wo": {"nuclear": "without"}}, build="wo")
        logger.assert_not_logged(UNMATCHED_ARITHMETIC)
        assert not [m for m in logger.warnings if "remove" in m.lower()]


class TestReplacePartial:
    def test_only_the_filled_columns_change(self, tmp_path):
        # E13
        book = BIO + sheet(
            "nodedata", "Country | Grid | Scenario | Year | price | emission_CO2 | method | variant",
            "FI00 | gas | all | 1 | 55 | 0.198 |                 | bio-price:all",
            "FI00 | gas | all | 1 | 80 |       | replace-partial | bio-price:high",
        )
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": book}, variants=BUILDS, build="hi")
        assert price(pipeline, "FI00", "gas") == 80
        assert value_at(pipeline.df_nodedata, "emission_co2", node="FI00_gas") == pytest.approx(0.198)


class TestWrittenTwice:
    """The "written twice in one sheet" message, at the seam with variants.

    It must fire only where a row really is dead text in this build. Rows of
    different options never merge together, and an ``:all`` row is never in
    the same pass as an option row, so neither is dead text.
    """

    @pytest.mark.parametrize("build", ["", "lo", "hi"])
    def test_two_options_on_one_key_are_not_dead_text(self, tmp_path, build):
        # E15
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {LOW}  |  | bio-price:low",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
        )
        _, _, logger = fi_price(tmp_path, book, build)
        logger.assert_not_logged(WRITTEN_TWICE)

    def test_an_all_row_and_an_option_row_are_not_dead_text(self, tmp_path):
        # E16
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
        )
        _, _, logger = fi_price(tmp_path, book, "hi")
        logger.assert_not_logged(WRITTEN_TWICE)

    DH = declarations(
        ("dh-level", "national", "yes", "none"),
        ("dh-level", "grouped", "", "none"),
        ("dh-level", "cities", "", "none"),
    ) + sheet(
        "demanddata", DEMAND_HEADER,
        "FI00 | dheat |     | all | 1 | 10 |  | dh-level:national",
        "FI00 | dheat | TRE | all | 1 | 4  |  | dh-level:grouped,cities",
        "FI00 | dheat | TRE | all | 1 | 5  |  | dh-level:cities",
    )

    def test_two_rows_that_meet_in_one_option_are_dead_text_there(self, tmp_path):
        # E17
        _, logger = run_variants(tmp_path, {"book.xlsx": self.DH},
                                 variants={"c": {"dh-level": {"FI00": "cities"}}}, build="c")
        logger.assert_logged(WRITTEN_TWICE, level="warn")

    def test_and_only_there(self, tmp_path):
        # E17's control
        _, logger = run_variants(tmp_path, {"book.xlsx": self.DH},
                                 variants={"g": {"dh-level": {"FI00": "grouped"}}}, build="g")
        logger.assert_not_logged(WRITTEN_TWICE)

    def test_untagged_rows_are_judged_as_today(self, tmp_path):
        # E18
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  |",
            f"FI00 | biomass | all | 1 | {HIGH} |  |",
        )
        _, _, logger = fi_price(tmp_path, book, "")
        logger.assert_logged(WRITTEN_TWICE, level="warn")
