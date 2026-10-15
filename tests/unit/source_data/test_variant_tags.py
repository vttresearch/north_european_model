"""Reading the ``variant`` column and the ``variantdata`` declarations.

Expectations: ``tests/expectations/variants.md``, group A. Each test names
its row.

The prices are inputs, so asserting which one a build ends up with says which row
won (provenance), not what a price should be.
"""

import pytest

from tests._common.variants import (
    DECLARED_DIFFERENTLY,
    MALFORMED,
    NODE_HEADER,
    NO_ROWS_ANYWHERE,
    RESERVED,
    UNDECLARED,
    UNKNOWN_CHECK,
    declarations,
    price,
    run_variants,
    sheet,
)

BASE, LOW, HIGH, STRAY = 40, 30, 55, 99

BIO = declarations(
    ("bio-price", "low", "", "coverage"),
    ("bio-price", "ref", "yes", "coverage"),
    ("bio-price", "high", "", "coverage"),
)

BUILDS = {"": {}, "lo": {"bio-price": {"FI00": "low"}}, "hi": {"bio-price": {"FI00": "high"}}}


def prices(*rows: str) -> str:
    return sheet("nodedata", NODE_HEADER, *rows)


STANDARD = BIO + prices(
    f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
    f"FI00 | biomass | all | 1 | {LOW}  |  | bio-price:low",
    f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
)


class TestWhatATagMeans:
    def test_a_sheet_without_the_column_is_untagged(self, tmp_path):
        # A1
        plain = sheet("nodedata_plain", "Country | Grid | Scenario | Year | price",
                      "FI00 | gas | all | 1 | 20")
        for build in BUILDS:
            pipeline, _ = run_variants(tmp_path, {"book.xlsx": STANDARD + plain},
                                       variants=BUILDS, build=build)
            assert price(pipeline, "FI00", "gas") == 20

    def test_an_empty_cell_is_read_in_every_build(self, tmp_path):
        # A2
        book = STANDARD.replace(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all\nFI00 | gas | all | 1 | 20 |  |",
        )
        for build in BUILDS:
            pipeline, logger = run_variants(tmp_path, {"book.xlsx": book},
                                            variants=BUILDS, build=build)
            assert price(pipeline, "FI00", "gas") == 20
            logger.assert_not_logged(MALFORMED)
            logger.assert_not_logged(UNDECLARED)

    @pytest.mark.parametrize("build, expected", [("", BASE), ("lo", LOW), ("hi", HIGH)])
    def test_an_all_row_is_the_base_an_option_builds_on(self, tmp_path, build, expected):
        # A3: 'bio-price' has a default, so every country has an option.
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": STANDARD},
                                   variants=BUILDS, build=build)
        assert price(pipeline, "FI00") == expected

    @pytest.mark.parametrize("build, expected", [("", BASE), ("lo", HIGH), ("hi", HIGH)])
    def test_a_row_listing_two_options_is_in_both(self, tmp_path, build, expected):
        # A4
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:low,high",
        )
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": book},
                                   variants=BUILDS, build=build)
        assert price(pipeline, "FI00") == expected


class TestHowATagIsWritten:
    @pytest.mark.parametrize("tag", ["Bio-Price:HIGH", " bio-price : high ", "BIO-PRICE:High"])
    def test_case_and_spaces_do_not_matter_in_excel(self, tmp_path, tag):
        # A5
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | {tag}",
        )
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book},
                                        variants=BUILDS, build="hi")
        assert price(pipeline, "FI00") == HIGH
        logger.assert_not_logged(MALFORMED)
        logger.assert_not_logged(UNDECLARED)

    @pytest.mark.parametrize("tag", [
        "bio-price",                   # no options: 'every option' is written ':all'
        "bio-price:",                  # no option
        "bio-price:all,high",          # 'all' mixed with an option
        ":high",                       # no variant
        "bio-price:high:x",            # two colons
        "bio price:high",              # space inside a name
        "bio_price:high",              # '_' separates node-name parts
        "bio-price:high,",             # an empty option
        "bio-price:high; fuel:low",    # A7: one variant per cell
    ])
    def test_a_malformed_tag_is_warned_about_and_its_row_dropped(self, tmp_path, tag):
        # A6, A7
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE}  |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {STRAY} |  | {tag}",
        )
        for build in ("", "hi"):
            pipeline, logger = run_variants(tmp_path, {"book.xlsx": book},
                                            variants=BUILDS, build=build)
            logger.assert_logged(MALFORMED, level="warn")
            assert any("nodedata" in m for m in logger.matching(MALFORMED))
            assert price(pipeline, "FI00") == BASE

    def test_a_malformed_tag_in_another_years_row_is_still_reported(self, tmp_path):
        # A16: form is judged before relevance.
        book = BIO + prices(
            f"FI00 | biomass | all | 1    | {BASE}  |  | bio-price:all",
            f"FI00 | biomass | all | 2050 | {STRAY} |  | bio-price:",
        )
        _, logger = run_variants(tmp_path, {"book.xlsx": book}, variants=BUILDS)
        logger.assert_logged(MALFORMED, level="warn")


class TestDeclarations:
    @pytest.mark.parametrize("tag", ["bio-prize:high", "bio-price:hihg"])
    def test_an_undeclared_tag_is_dropped_not_read_as_untagged(self, tmp_path, tag):
        # A8. The stray row sits after the base row: read as untagged, it would
        # replace the base in pass 1 and every build would show it.
        book = BIO + prices(
            f"FI00 | biomass | all | 1 | {BASE}  |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {STRAY} |  | {tag}",
        )
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book}, variants=BUILDS)
        logger.assert_logged(UNDECLARED, level="warn")
        assert any(tag.split(":")[0] in m.lower() for m in logger.matching(UNDECLARED))
        assert price(pipeline, "FI00") == BASE

    def test_a_workbook_may_use_only_what_it_declares_itself(self, tmp_path):
        # A9: b.xlsx has no variantdata sheet; a.xlsx's does not cover it.
        a = STANDARD.replace(f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high\n", "")
        b = prices(f"SE01 | biomass | all | 1 | {HIGH} |  | bio-price:high")
        variants = {"": {}, "hi": {"bio-price": "high"}}
        pipeline, logger = run_variants(tmp_path, {"a.xlsx": a, "b.xlsx": b},
                                        variants=variants, build="hi")
        logger.assert_logged(UNDECLARED, level="warn")
        assert any("b.xlsx" in m for m in logger.matching(UNDECLARED))
        assert price(pipeline, "SE01") is None

    def test_options_are_pooled_across_workbooks(self, tmp_path):
        # A10
        a = declarations(("bio-price", "ref", "yes", "coverage"),
                         ("bio-price", "high", "", "coverage")) + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
        )
        b = declarations(("bio-price", "ref", "yes", "coverage"),
                         ("bio-price", "low", "", "coverage")) + prices(
            f"FI00 | biomass | all | 1 | {LOW} |  | bio-price:low",
        )
        for build, expected in (("lo", LOW), ("hi", HIGH)):
            pipeline, logger = run_variants(tmp_path, {"a.xlsx": a, "b.xlsx": b},
                                            variants=BUILDS, build=build)
            assert price(pipeline, "FI00") == expected
            logger.assert_not_logged(UNDECLARED)
            logger.assert_not_logged(DECLARED_DIFFERENTLY)

    @pytest.mark.parametrize("b_rows", [
        # a different default
        (("bio-price", "ref", "", "coverage"), ("bio-price", "high", "yes", "coverage")),
        # a different check
        (("bio-price", "ref", "yes", "totals"), ("bio-price", "high", "", "totals")),
    ])
    def test_disagreeing_declarations_keep_the_first_and_say_so_loudly(self, tmp_path, b_rows):
        # A11: a.xlsx comes first in the config, so its declaration wins.
        a = STANDARD
        b = declarations(*b_rows) + prices(f"SE01 | biomass | all | 1 | {BASE} |  | bio-price:all")
        pipeline, logger = run_variants(tmp_path, {"a.xlsx": a, "b.xlsx": b}, variants=BUILDS)
        logger.assert_logged(DECLARED_DIFFERENTLY, level="warn")
        message = " ".join(logger.matching(DECLARED_DIFFERENTLY))
        assert "a.xlsx" in message and "b.xlsx" in message
        assert price(pipeline, "FI00") == BASE        # 'ref' stayed the default
        assert any("bio-price" in cause for cause in pipeline.partial_causes)

    def test_two_defaults_keep_the_first(self, tmp_path):
        # A12
        book = declarations(
            ("bio-price", "ref", "yes", "coverage"),
            ("bio-price", "high", "yes", "coverage"),
        ) + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
        )
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book}, variants={"": {}})
        logger.assert_logged(DECLARED_DIFFERENTLY, level="warn")
        assert price(pipeline, "FI00") == BASE

    def test_an_unknown_check_drops_its_declaration(self, tmp_path):
        # A13: 'totalss' is neither read as 'none' nor guessed at. The option it
        # declared is then undeclared, so its rows go too (A8) and a build that
        # chooses it falls back to the default (D4).
        book = declarations(
            ("bio-price", "ref", "yes", "coverage"),
            ("bio-price", "high", "", "totalss"),
        ) + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
        )
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book},
                                        variants=BUILDS, build="hi")
        logger.assert_logged(UNKNOWN_CHECK, level="warn")
        logger.assert_logged(UNDECLARED, level="warn")
        assert price(pipeline, "FI00") == BASE

    def test_a_declared_option_nothing_provides_is_warned_about(self, tmp_path):
        # A14
        book = declarations(
            ("bio-price", "ref", "yes", "coverage"),
            ("bio-price", "extreme", "", "coverage"),
        ) + prices(f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all")
        _, logger = run_variants(tmp_path, {"book.xlsx": book}, variants={"": {}})
        logger.assert_logged(NO_ROWS_ANYWHERE, level="warn")
        assert any("extreme" in m for m in logger.matching(NO_ROWS_ANYWHERE))

    @pytest.mark.parametrize("row", [
        ("bio-price", "all", "", "coverage"),   # an option named 'all'
        ("all", "x", "yes", "none"),            # a variant named 'all'
    ])
    def test_all_is_reserved(self, tmp_path, row):
        # A17: 'all' means every option, so nothing may be called that. The
        # declaration row is dropped; a row tagged with it is then undeclared (A8).
        book = declarations(("bio-price", "ref", "yes", "coverage"), row) + prices(
            f"FI00 | biomass | all | 1 | {BASE} |  | bio-price:all",
        )
        _, logger = run_variants(tmp_path, {"book.xlsx": book}, variants={"": {}})
        logger.assert_logged(RESERVED, level="warn")

    def test_a_default_without_rows_of_its_own_is_silent(self, tmp_path):
        # A15: STANDARD's 'ref' has no rows of its own.
        _, logger = run_variants(tmp_path, {"book.xlsx": STANDARD}, variants=BUILDS)
        logger.assert_not_logged(NO_ROWS_ANYWHERE)
