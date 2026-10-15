"""Which rows a build keeps, per country and per link, and what happens when it can't.

Expectations: ``tests/expectations/variants.md``, groups C and D, and G2
(``variant_choices``). Each test names its row.
"""

import pytest

from tests._common.variants import (
    ALL_REACHED,
    ALL_REACHES_NOTHING,
    COUNTRIES,
    DEMAND_HEADER,
    DORMANT,
    ENDS_DISAGREE,
    NODE_HEADER,
    NO_COUNTRY,
    NO_ROWS_FOR_OPTION,
    UNDECLARED_OPTION,
    UNDECLARED_VARIANT,
    declarations,
    price,
    rows_where,
    run_variants,
    sheet,
    value_at,
)

BASE, HIGH = 40, 55

BIO_ROWS = (
    ("bio-price", "low", "", "coverage"),
    ("bio-price", "ref", "yes", "coverage"),
    ("bio-price", "high", "", "coverage"),
)
PRICE_SHEET = sheet(
    "nodedata", NODE_HEADER,
    f"all | biomass | all | 1 | {BASE} |  | bio-price:all",
    f"all | biomass | all | 1 | {HIGH} |  | bio-price:high",
    "all | gas     | all | 1 | 20     |  |",
)
PRICES = declarations(*BIO_ROWS) + PRICE_SHEET

DH_ROWS = (
    ("dh-level", "national", "yes", "totals"),
    ("dh-level", "grouped", "", "totals"),
    ("dh-level", "cities", "", "totals"),
)
#: Each option sums to 10 TWh: national 10, grouped CAP 6 + TRE 4, cities HKI 6 + TRE 4.
DH_SHEET = sheet(
    "demanddata", DEMAND_HEADER,
    "FI00 | dheat |     | all | 1 | 10 |  | dh-level:national",
    "FI00 | dheat | HKI | all | 1 | 6  |  | dh-level:cities",
    "FI00 | dheat | TRE | all | 1 | 4  |  | dh-level:grouped,cities",
    "FI00 | dheat | CAP | all | 1 | 6  |  | dh-level:grouped",
)
DH_DEMAND = declarations(*DH_ROWS) + DH_SHEET


def has_node(df, node: str) -> bool:
    return not rows_where(df, node=node).empty


class TestRowsWithACountry:
    @pytest.mark.parametrize("build", ["", "hi"])
    def test_untagged_and_all_rows_are_everywhere(self, tmp_path, build):
        # C1: ':all' rows are read wherever the country has an option, and
        # 'bio-price' has a default, so here that is everywhere.
        variants = {"": {}, "hi": {"bio-price": {"FI00": "high"}}}
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": PRICES}, variants=variants, build=build)
        for country in COUNTRIES:
            assert price(pipeline, country, "gas") == 20
            assert price(pipeline, country) is not None

    def test_an_all_row_is_expanded_then_chosen_per_country(self, tmp_path):
        # C2
        variants = {"hi": {"bio-price": {"FI00": "high"}}}
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": PRICES}, variants=variants, build="hi")
        assert price(pipeline, "FI00") == HIGH
        for country in ("SE01", "SE02", "EE00"):
            assert price(pipeline, country) == BASE

    @pytest.mark.parametrize("option, present, absent", [
        ("national", ["FI00_dheat"], ["FI00_dheat_HKI", "FI00_dheat_TRE", "FI00_dheat_CAP"]),
        ("grouped", ["FI00_dheat_TRE", "FI00_dheat_CAP"], ["FI00_dheat", "FI00_dheat_HKI"]),
        ("cities", ["FI00_dheat_HKI", "FI00_dheat_TRE"], ["FI00_dheat", "FI00_dheat_CAP"]),
    ])
    def test_a_country_keeps_the_rows_of_its_option(self, tmp_path, option, present, absent):
        # C3
        variants = {"b": {"dh-level": {"FI00": option}}}
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": DH_DEMAND}, variants=variants, build="b")
        for node in present:
            assert has_node(pipeline.df_demanddata, node), node
        for node in absent:
            assert not has_node(pipeline.df_demanddata, node), node

    def test_rows_of_other_years_and_scenarios_are_gone_first(self, tmp_path):
        # C4
        book = declarations(*BIO_ROWS) + sheet(
            "nodedata", NODE_HEADER,
            f"FI00 | biomass | all   | 1    | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all   | 1    | {HIGH} |  | bio-price:high",
            "FI00 | biomass | all   | 2050 | 77     |  | bio-price:high",
            "FI00 | biomass | other | 1    | 88     |  | bio-price:high",
        )
        variants = {"hi": {"bio-price": {"FI00": "high"}}}
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": book}, variants=variants, build="hi")
        assert price(pipeline, "FI00") == HIGH

    def test_a_tagged_row_on_an_excluded_node_is_gone(self, tmp_path):
        # C5
        variants = {"c": {"dh-level": {"FI00": "cities"}}}
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": DH_DEMAND}, variants=variants,
                                   build="c", exclude_nodes=["FI00_dheat_HKI"])
        assert not has_node(pipeline.df_demanddata, "FI00_dheat_HKI")
        assert has_node(pipeline.df_demanddata, "FI00_dheat_TRE")

    def test_user_constraints_choose_per_country(self, tmp_path):
        # C13
        book = declarations(
            ("uc", "tight", "yes", "coverage"),
            ("uc", "loose", "", "coverage"),
        ) + sheet(
            "userconstraintdata",
            "Scenario | year | country | group | 1st dimension | 2nd dimension | parameter | value | variant",
            "all | 1 | FI00 | elecLimit | FI00_elec | v_state | coefficient | 1   | uc:all",
            "all | 1 | FI00 | elecLimit |           |         | constant    | 250 | uc:all",
            "all | 1 | FI00 | elecLimit |           |         | constant    | 400 | uc:loose",
        )
        for build, expected in (("", 250), ("l", 400)):
            pipeline, _ = run_variants(tmp_path, {"book.xlsx": book},
                                       variants={"": {}, "l": {"uc": {"FI00": "loose"}}}, build=build)
            assert value_at(pipeline.df_userconstraintdata, "value",
                            group="elecLimit", parameter="constant") == expected


class TestRowsWithoutACountry:
    CO2 = declarations(
        ("co2-price", "ref", "yes", "coverage"),
        ("co2-price", "high", "", "coverage"),
    ) + sheet(
        "emissiondata", "Scenario | Year | emission | group | price | variant",
        "all | 1 | CO2 | ETS | 95  | co2-price:all",
        "all | 1 | CO2 | ETS | 150 | co2-price:high",
    )

    def co2(self, pipeline):
        return value_at(pipeline.df_emissiondata, "price", emission="CO2", group="ETS")

    def test_they_take_the_all_choice(self, tmp_path):
        # C6
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": self.CO2},
                                   variants={"h": {"co2-price": "high"}}, build="h")
        assert self.co2(pipeline) == 150

    def test_a_per_country_choice_cannot_reach_them(self, tmp_path):
        # C7: the rows take 'all' or the default, and the build says why.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": self.CO2},
                                        variants={"h": {"co2-price": {"FI00": "high"}}}, build="h")
        assert self.co2(pipeline) == 95
        logger.assert_logged(NO_COUNTRY, level="warn")
        assert any("co2-price" in cause for cause in pipeline.partial_causes)


class TestTransfers:
    IC = declarations(
        ("ic", "base", "yes", "coverage"),
        ("ic", "ext", "", "coverage"),
    ) + sheet(
        "transferdata", "grid | from_country | to_country | scenario | Year | transferCap | variant",
        "elec | FI00 | SE01 | all | 1 | 1000 | ic:all",
        "elec | FI00 | EE00 | all | 1 | 500  | ic:all",
        "elec | FI00 | SE01 | all | 1 | 2000 | ic:ext",
        "elec | FI00 | EE00 | all | 1 | 900  | ic:ext",
        "elec | SE01 | FI00 | all | 1 | 1800 | ic:ext",
    )

    def cap(self, pipeline, a, b):
        return value_at(pipeline.df_transferdata, "transfercap", from_country=a, to_country=b)

    def run(self, tmp_path, choice):
        return run_variants(tmp_path, {"book.xlsx": self.IC},
                            variants={"x": {"ic": choice}}, build="x")

    def test_a_link_takes_an_option_both_ends_chose(self, tmp_path):
        # C8, C9
        pipeline, _ = self.run(tmp_path, {"FI": "ext", "SE": "ext"})
        assert self.cap(pipeline, "FI00", "SE01") == 2000
        assert self.cap(pipeline, "FI00", "EE00") == 500

    def test_naming_estonia_too_extends_its_link(self, tmp_path):
        # C10
        pipeline, _ = self.run(tmp_path, {"EE": "ext", "FI": "ext"})
        assert self.cap(pipeline, "FI00", "EE00") == 900
        assert self.cap(pipeline, "FI00", "SE01") == 1000

    @pytest.mark.parametrize("choice, kept", [
        ({"FI": "ext", "SE": "ext"}, True),
        ({"FI": "ext"}, False),
        ({"SE": "ext"}, False),
    ])
    def test_the_order_of_a_links_ends_does_not_matter(self, tmp_path, choice, kept):
        # C12: SE01 -> FI00 is judged by the same two ends as FI00 -> SE01.
        pipeline, _ = self.run(tmp_path, choice)
        assert bool(self.cap(pipeline, "SE01", "FI00") == 1800) is kept

    def test_all_picks_only_the_variants_own_links(self, tmp_path):
        # C15: 'all' reaches the countries with 'ext' rows at either end, here
        # FI00, SE01 and EE00. A link takes 'ext' only where it has 'ext' rows, so
        # the SE01-EE00 link, which no 'ic' row mentions, is untouched, and no
        # link is made between countries just because both reached 'ext'.
        book = self.IC + "elec | SE01 | EE00 | all | 1 | 300 |\n"
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book},
                                        variants={"x": {"ic": "ext"}}, build="x")
        assert self.cap(pipeline, "FI00", "SE01") == 2000
        assert self.cap(pipeline, "FI00", "EE00") == 900
        assert self.cap(pipeline, "SE01", "EE00") == 300
        assert self.cap(pipeline, "EE00", "SE01") is None
        logger.assert_not_logged(NO_ROWS_FOR_OPTION)
        logger.assert_not_logged(ENDS_DISAGREE)

    def test_ends_that_disagree_with_no_default_drop_the_link_loudly(self, tmp_path):
        # C11
        book = declarations(
            ("ic2", "ext", "", "none"),
            ("ic2", "mega", "", "none"),
        ) + sheet(
            "transferdata", "grid | from_country | to_country | scenario | Year | transferCap | variant",
            "elec | FI00 | SE01 | all | 1 | 100  | ic2:all",
            "elec | FI00 | SE01 | all | 1 | 3000 | ic2:ext",
        )
        # The link has no option, so neither its 'ext' row nor its ':all' row is read.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book},
                                        variants={"x": {"ic2": {"FI00": "ext"}}}, build="x")
        assert self.cap(pipeline, "FI00", "SE01") is None
        logger.assert_logged(ENDS_DISAGREE, level="warn")
        assert any("ic2" in cause for cause in pipeline.partial_causes)

    def test_ends_that_both_left_it_dormant_say_nothing(self, tmp_path):
        # C11's control, and D6 for a link.
        book = declarations(("ic2", "ext", "", "none")) + sheet(
            "transferdata", "grid | from_country | to_country | scenario | Year | transferCap | variant",
            "elec | FI00 | SE01 | all | 1 | 100  | ic2:all",
            "elec | FI00 | SE01 | all | 1 | 3000 | ic2:ext",
        )
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book}, variants={"": {}})
        assert self.cap(pipeline, "FI00", "SE01") is None
        logger.assert_not_logged(ENDS_DISAGREE)


class TestRule2AndFallbacks:
    DH_WITH_TALLINN = declarations(
        ("dh-level", "national", "yes", "totals"),
        ("dh-level", "cities", "", "totals"),
        ("dh-level", "tallinn", "", "totals"),
    ) + sheet(
        "demanddata", DEMAND_HEADER,
        "FI00 | dheat | ROF | all | 1 | 5  |  | dh-level:all",
        "FI00 | dheat |     | all | 1 | 10 |  | dh-level:national",
        "FI00 | dheat | HKI | all | 1 | 10 |  | dh-level:cities",
        "EE00 | dheat |     | all | 1 | 3  |  | dh-level:national",
        "EE00 | dheat | TLL | all | 1 | 3  |  | dh-level:tallinn",
    )

    def test_a_country_with_rows_for_its_option_is_fine(self, tmp_path):
        # D1
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": self.DH_WITH_TALLINN},
                                        variants={"c": {"dh-level": {"FI00": "cities"}}}, build="c")
        logger.assert_not_logged(NO_ROWS_FOR_OPTION)
        assert pipeline.partial_causes == []
        assert has_node(pipeline.df_demanddata, "FI00_dheat_HKI")

    def test_an_option_the_country_has_no_rows_for_falls_back(self, tmp_path):
        # D2: Finland's ':all' ROF row is no evidence for 'tallinn'.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": self.DH_WITH_TALLINN},
                                        variants={"t": {"dh-level": {"FI00": "tallinn"}}}, build="t")
        logger.assert_logged(NO_ROWS_FOR_OPTION, level="warn")
        message = " ".join(logger.matching(NO_ROWS_FOR_OPTION))
        assert "FI00" in message and "tallinn" in message
        assert has_node(pipeline.df_demanddata, "FI00_dheat")          # national
        assert pipeline.variant_choices["dh-level"]["FI00"] == "national"
        assert pipeline.partial_causes

    def test_two_letter_keys_are_judged_per_code(self, tmp_path):
        # D1's note: 'EE' expands to EE00 before anything asks whether it has rows.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": self.DH_WITH_TALLINN},
                                        variants={"t": {"dh-level": {"EE": "tallinn"}}}, build="t")
        logger.assert_not_logged(NO_ROWS_FOR_OPTION)
        assert has_node(pipeline.df_demanddata, "EE00_dheat_TLL")

    def test_choosing_the_empty_default_is_fine(self, tmp_path):
        # D3: 'ref' has no rows of its own.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": PRICES},
                                        variants={"r": {"bio-price": {"FI00": "ref"}}}, build="r")
        logger.assert_not_logged(NO_ROWS_FOR_OPTION)
        assert pipeline.partial_causes == []
        assert price(pipeline, "FI00") == BASE

    def test_an_undeclared_option_falls_back(self, tmp_path):
        # D4
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": DH_DEMAND},
                                        variants={"c": {"dh-level": {"FI00": "citiez"}}}, build="c")
        logger.assert_logged(UNDECLARED_OPTION, level="warn")
        assert has_node(pipeline.df_demanddata, "FI00_dheat")
        assert pipeline.partial_causes

    def test_an_undeclared_variant_is_ignored_and_the_rest_still_applies(self, tmp_path):
        # D5: report every mistake in one build, never abort on the first.
        book = declarations(*BIO_ROWS, *DH_ROWS) + PRICE_SHEET + DH_SHEET
        variants = {"x": {"dh-levl": "cities", "bio-price": {"FI00": "high"}}}
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book}, variants=variants, build="x")
        logger.assert_logged(UNDECLARED_VARIANT, level="warn")
        assert price(pipeline, "FI00") == HIGH
        assert has_node(pipeline.df_demanddata, "FI00_dheat")
        assert pipeline.partial_causes


#: Three new units with no default: none of them is right unless a build says so.
#: The uranium node is shared by every option and written once, for every country.
NEW_UNITS = declarations(
    ("new-units", "smr", "", "none"),
    ("new-units", "ccgt", "", "none"),
) + sheet(
    "unittypedata", "scenario | year | unittype | grid_output1 | eff00",
    "all | 1 | SMR  | elec | 0.33",
    "all | 1 | CCGT | elec | 0.6",
) + sheet(
    "unitdata", "Country | unittype | unit_name_prefix | Scenario | Year | capacity_output1 | variant",
    "FI00 | SMR  |  | all | 1 | 300 | new-units:smr",
    "SE01 | SMR  |  | all | 1 | 300 | new-units:smr",
    "FI00 | CCGT |  | all | 1 | 400 | new-units:ccgt",
) + sheet(
    "nodedata", NODE_HEADER,
    "all | uranium | all | 1 | 5 |  | new-units:all",
)


def has_unit(pipeline, country, unittype):
    return not rows_where(pipeline.df_unitdata, country=country, unittype=unittype).empty


class TestDormantVariants:
    def test_a_variant_without_a_default_left_unnamed_reads_nothing(self, tmp_path):
        # D6: dormant means none of its rows, ':all' included.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": NEW_UNITS}, variants={"": {}})
        assert not has_unit(pipeline, "FI00", "SMR")
        assert not has_unit(pipeline, "FI00", "CCGT")
        for country in COUNTRIES:
            assert price(pipeline, country, "uranium") is None, country
        assert not [m for m in logger.warnings if "new-units" in m]
        assert pipeline.partial_causes == []
        assert any(DORMANT in m.lower() and "new-units" in m for m in logger.messages)
        assert pipeline.variant_choices["new-units"]["FI00"] is None

    def test_naming_it_for_one_country_builds_it_there_and_nowhere_else(self, tmp_path):
        # C14: the 'all' country row is copied to every code first, and then
        # kept only where the country has an option.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": NEW_UNITS},
                                        variants={"s": {"new-units": {"FI00": "smr"}}}, build="s")
        assert has_unit(pipeline, "FI00", "SMR")
        assert price(pipeline, "FI00", "uranium") == 5
        for country in ("SE01", "SE02", "EE00"):
            assert not has_unit(pipeline, country, "SMR"), country
            assert price(pipeline, country, "uranium") is None, country
        assert not [m for m in logger.warnings if "new-units" in m]

    def test_a_two_letter_key_is_checked_per_code(self, tmp_path):
        # D7: 'SE' lists SE01 and SE02. SE02 has no 'smr' rows, so it warns and,
        # with no default, stays dormant.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": NEW_UNITS},
                                        variants={"s": {"new-units": {"SE": "smr"}}}, build="s")
        assert has_unit(pipeline, "SE01", "SMR")
        logger.assert_logged(NO_ROWS_FOR_OPTION, level="warn")
        assert any("SE02" in m for m in logger.matching(NO_ROWS_FOR_OPTION))
        assert price(pipeline, "SE02", "uranium") is None


class TestTheAllKey:
    """'all' reaches the countries that have rows for the option, never all of country_codes."""

    def test_all_reaches_only_where_the_option_has_rows(self, tmp_path):
        # C16: 'smr' rows exist for FI00 and SE01; SE02 and EE00 get nothing.
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": NEW_UNITS},
                                        variants={"a": {"new-units": {"all": "smr"}}}, build="a")
        assert has_unit(pipeline, "FI00", "SMR") and has_unit(pipeline, "SE01", "SMR")
        for country in ("SE02", "EE00"):
            assert pipeline.variant_choices["new-units"][country] is None, country
            assert price(pipeline, country, "uranium") is None, country
        logger.assert_not_logged(NO_ROWS_FOR_OPTION)

    def test_the_unreached_take_the_default_silently(self, tmp_path):
        # C17: 'high' rows only for FI00, ':all' rows everywhere.
        book = declarations(*BIO_ROWS) + sheet(
            "nodedata", NODE_HEADER,
            f"all  | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
        )
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book},
                                        variants={"h": {"bio-price": "high"}}, build="h")
        assert price(pipeline, "FI00") == HIGH
        for country in ("SE01", "SE02", "EE00"):
            assert price(pipeline, country) == BASE, country
            assert pipeline.variant_choices["bio-price"][country] == "ref", country
        logger.assert_not_logged(NO_ROWS_FOR_OPTION)
        assert pipeline.partial_causes == []

    def test_what_all_reached_is_logged(self, tmp_path):
        # C18: silent fallback, but a clear line saying who was reached and who
        # took the default.
        book = declarations(*BIO_ROWS) + sheet(
            "nodedata", NODE_HEADER,
            f"all  | biomass | all | 1 | {BASE} |  | bio-price:all",
            f"FI00 | biomass | all | 1 | {HIGH} |  | bio-price:high",
        )
        _, logger = run_variants(tmp_path, {"book.xlsx": book},
                                 variants={"h": {"bio-price": "high"}}, build="h")
        lines = logger.matching(ALL_REACHED)
        assert lines, logger.messages
        text = " ".join(lines)
        for part in ("bio-price", "high", "FI00", "ref"):
            assert part in text, part

    def test_all_reaching_no_country_warns(self, tmp_path):
        # C19: alongside A14's "no rows in any workbook".
        book = declarations(("new-units", "smr", "", "none"), ("new-units", "wind", "", "none")) + sheet(
            "unittypedata", "scenario | year | unittype | grid_output1 | eff00",
            "all | 1 | SMR | elec | 0.33",
        ) + sheet(
            "unitdata", "Country | unittype | unit_name_prefix | Scenario | Year | capacity_output1 | variant",
            "FI00 | SMR |  | all | 1 | 300 | new-units:smr",
        )
        _, logger = run_variants(tmp_path, {"book.xlsx": book},
                                 variants={"w": {"new-units": {"all": "wind"}}}, build="w")
        logger.assert_logged(ALL_REACHES_NOTHING, level="warn")

    def test_an_option_written_for_every_country_reaches_every_country(self, tmp_path):
        # C20: "defined in the variant" is judged after country 'all' is copied.
        book = NEW_UNITS.replace(
            "FI00 | SMR  |  | all | 1 | 300 | new-units:smr\n"
            "SE01 | SMR  |  | all | 1 | 300 | new-units:smr",
            "all  | SMR  |  | all | 1 | 300 | new-units:smr",
        )
        assert book != NEW_UNITS
        pipeline, _ = run_variants(tmp_path, {"book.xlsx": book},
                                   variants={"a": {"new-units": {"all": "smr"}}}, build="a")
        for country in COUNTRIES:
            assert has_unit(pipeline, country, "SMR"), country


class TestOneChoiceAcrossTables:
    def test_rows_in_any_table_are_evidence_and_every_table_follows(self, tmp_path):
        # C21: 'cities' rows live only in demanddata. They are FI00's evidence, so
        # no warning; and nodedata, which has only a 'national' row, follows the
        # same choice and drops it.
        book = declarations(*DH_ROWS) + DH_SHEET + sheet(
            "nodedata", "Country | Grid | Node_suffix | Scenario | Year | maxSpill | variant",
            "FI00 | dheat |  | all | 1 | 154 | dh-level:national",
        )
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": book},
                                        variants={"c": {"dh-level": {"FI00": "cities"}}}, build="c")
        logger.assert_not_logged(NO_ROWS_FOR_OPTION)
        assert not has_node(pipeline.df_nodedata, "FI00_dheat")
        assert has_node(pipeline.df_demanddata, "FI00_dheat_HKI")


class TestTheResolvedChoice:
    def test_every_country_has_its_option(self, tmp_path):
        # G2
        pipeline, logger = run_variants(tmp_path, {"book.xlsx": PRICES},
                                        variants={"hi": {"bio-price": {"FI00": "high"}}}, build="hi")
        assert pipeline.variant_choices["bio-price"] == {
            "FI00": "high", "SE01": "ref", "SE02": "ref", "EE00": "ref",
        }
        assert any("bio-price" in m and "high" in m for m in logger.messages)
