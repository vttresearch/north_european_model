"""The config's ``variants`` key: named builds, each choosing options per country.

Expectations: ``tests/expectations/variants.md``, groups B and G3-G8.

Config parsing runs before the logger exists, so a mistake here raises (CLAUDE.md,
"Error handling policy") and the tests use ``pytest.raises`` on a stable substring.

The normalised shape is the contract the source data phase reads, so it is pinned
(tests/README.md, pinning case 4, format contracts): build name as written ->
variant, lower case -> {``'all'`` or an exact country code: option, lower case}.
Two-letter keys are expanded to every code they match, and the most specific key
wins. Folder names are pinned for the reason ``TestOutputFolderName`` gives:
they name folders on disk.
"""

import textwrap

import pytest

from src.infrastructure.config_reader import (
    build_name,
    config_output_folder_names,
    load_config,
    output_folder_name,
    parse_variants,
    scenario_tags,
    selected_builds,
)

COUNTRIES = ["FI00", "SE01", "SE02", "EE00"]

MINIMAL_INI = """\
[inputdata]
scenarios = ['test']
scenario_years = [2030]
climate_data = 2014-2015
country_codes = ['FI00', 'SE01', 'SE02', 'EE00']
"""


def _write_ini(tmp_path, body: str):
    path = tmp_path / "config.ini"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


class TestTheKeyInTheIni:
    def test_no_key_is_one_build_with_every_default(self, tmp_path):
        # B1: an existing config keeps building one folder under today's name.
        config = load_config(_write_ini(tmp_path, MINIMAL_INI))
        assert config["variants"] == {"": {}}

    def test_the_key_is_read_and_normalised(self, tmp_path):
        body = MINIMAL_INI + (
            "variants = {'': {}, 'sens1': {'bio-price': {'FI00': 'high'}}}\n"
        )
        config = load_config(_write_ini(tmp_path, body))
        assert config["variants"][""] == {}
        assert config["variants"]["sens1"] == {"bio-price": {"FI00": "high"}}

    @pytest.mark.parametrize("key", [
        "scenario_alternatives", "scenario_alternatives2",
        "scenario_alternatives3", "scenario_alternatives4",
    ])
    @pytest.mark.parametrize("value", ["[]", "['alt1']"])
    def test_the_old_alternative_keys_are_refused(self, tmp_path, key, value):
        # B10: refused even when empty, which is how every shipped config wrote
        # it, so a stale config is told where its replacement is.
        with pytest.raises(ValueError, match="variants"):
            load_config(_write_ini(tmp_path, MINIMAL_INI + f"{key} = {value}\n"))

    def test_a_country_key_is_judged_against_country_codes(self, tmp_path):
        # B9 through the ini: the codes come from the same file.
        body = MINIMAL_INI + "variants = {'s': {'bio-price': {'DE00': 'high'}}}\n"
        with pytest.raises(ValueError, match="DE00"):
            load_config(_write_ini(tmp_path, body))


class TestBuildNames:
    def test_the_unsuffixed_build_is_allowed(self):
        # B2
        assert parse_variants({"": {}}, COUNTRIES) == {"": {}}

    def test_build_names_keep_their_spelling(self):
        # They become folder names, written as the author wrote them.
        assert list(parse_variants({"Sens1": {}}, COUNTRIES)) == ["Sens1"]

    def test_build_names_are_unique_case_insensitively(self):
        # B3: Windows would put both in one folder.
        with pytest.raises(ValueError, match="(?i)sens1"):
            parse_variants({"sens1": {}, "Sens1": {}}, COUNTRIES)

    @pytest.mark.parametrize("name", ["sens 1", "a/b", "a:b", "a_b", " sens1"])
    def test_a_build_name_must_be_a_folder_segment(self, name):
        # B4 and B12. No '_' either: it separates the parts of a folder name.
        with pytest.raises(ValueError, match="build name"):
            parse_variants({name: {}}, COUNTRIES)


class TestChoices:
    def test_a_string_applies_to_every_country(self):
        # B5
        assert parse_variants({"s": {"bio-price": "high"}}, COUNTRIES) == {
            "s": {"bio-price": {"all": "high"}}
        }

    def test_an_exact_code(self):
        # B6
        assert parse_variants({"s": {"bio-price": {"FI00": "high"}}}, COUNTRIES) == {
            "s": {"bio-price": {"FI00": "high"}}
        }

    def test_a_two_letter_key_covers_every_code_it_starts(self):
        # B7
        assert parse_variants({"s": {"bio-price": {"SE": "high"}}}, COUNTRIES) == {
            "s": {"bio-price": {"SE01": "high", "SE02": "high"}}
        }

    def test_the_most_specific_key_wins(self):
        # B8: exact code, then two letters, then all.
        choice = parse_variants(
            {"s": {"bio-price": {"all": "low", "SE": "high", "SE02": "ref"}}}, COUNTRIES
        )["s"]["bio-price"]
        assert choice == {"all": "low", "SE01": "high", "SE02": "ref"}

    def test_key_order_does_not_decide(self):
        # Specificity, not position: the exact code written first still wins.
        choice = parse_variants(
            {"s": {"bio-price": {"SE02": "ref", "SE": "high"}}}, COUNTRIES
        )["s"]["bio-price"]
        assert choice["SE02"] == "ref"
        assert choice["SE01"] == "high"

    def test_two_letter_codes_match_exactly_first(self):
        # A config whose codes are themselves two letters (the test default).
        assert parse_variants({"s": {"v": {"FI": "x"}}}, ["FI", "SE"]) == {
            "s": {"v": {"FI": "x"}}
        }

    def test_names_are_lower_cased_for_matching(self):
        # Excel cells match case-insensitively (A5); the config meets them there.
        assert parse_variants({"s": {"Bio-Price": {"FI00": "HIGH"}}}, COUNTRIES) == {
            "s": {"bio-price": {"FI00": "high"}}
        }

    @pytest.mark.parametrize("key", ["XX", "DE00", "DE", "FI0", "all-countries"])
    def test_a_key_matching_no_country_is_refused(self, key):
        # B9: a typo here would silently give that country the default.
        with pytest.raises(ValueError, match=key):
            parse_variants({"s": {"bio-price": {key: "high"}}}, COUNTRIES)


class TestShapeAndSpelling:
    @pytest.mark.parametrize("raw", [
        [],
        {"s": []},
        {"s": {"bio-price": 5}},
        {"s": {"bio-price": ["high"]}},
        {"s": {"bio-price": {"FI00": 5}}},
        {"s": {"bio-price": {"FI00": {"x": "y"}}}},
    ])
    def test_a_wrong_shape_is_refused_with_the_expected_one(self, raw):
        # B11
        with pytest.raises(ValueError, match="variants"):
            parse_variants(raw, COUNTRIES)

    @pytest.mark.parametrize("variant", ["bio price", "bio:price", "bio,price", "bio_price", ""])
    def test_a_variant_name_must_be_well_formed(self, variant):
        # B12: strict, like the other config dicts. '_' is out because the
        # source tables drop any value containing it.
        with pytest.raises(ValueError, match="variant"):
            parse_variants({"s": {variant: "high"}}, COUNTRIES)

    @pytest.mark.parametrize("option", ["hi gh", "hi:gh", "hi,gh", "hi_gh", "", " high"])
    def test_an_option_name_must_be_well_formed(self, option):
        # B12: no trimming in the config, unlike Excel.
        with pytest.raises(ValueError, match="option"):
            parse_variants({"s": {"bio-price": {"FI00": option}}}, COUNTRIES)

    @pytest.mark.parametrize("raw", [
        {"s": {"bio-price": "all"}},              # a build cannot choose every option
        {"s": {"bio-price": {"FI00": "all"}}},
        {"s": {"bio-price": {"all": "ALL"}}},
        {"s": {"all": "high"}},                   # nor name a variant 'all'
    ])
    def test_all_is_reserved(self, raw):
        # B12: 'all' means every option in a data tag. As a country key it still
        # means every country, which is a different place in the dict.
        with pytest.raises(ValueError, match="reserved"):
            parse_variants(raw, COUNTRIES)

    def test_all_as_a_country_key_is_still_fine(self):
        assert parse_variants({"s": {"bio-price": {"all": "high"}}}, COUNTRIES) == {
            "s": {"bio-price": {"all": "high"}}
        }


class TestOutputFolderNames:
    def test_the_build_name_is_the_suffix(self):
        # G3
        assert output_folder_name("input", "Observed Trends", 2030, "sens1") == (
            "input_ObservedTrends_2030_sens1"
        )

    def test_the_unsuffixed_build_keeps_todays_name(self):
        assert output_folder_name("input", "Observed Trends", 2030, "") == (
            "input_ObservedTrends_2030"
        )

    def test_the_name_without_its_prefix(self):
        # What add_scen_tags carries as 'name' (G5).
        assert build_name("Observed Trends", 2030, "sens1") == "ObservedTrends_2030_sens1"
        assert build_name("Observed Trends", 2030) == "ObservedTrends_2030"

    def test_every_build_of_every_scenario_and_year(self):
        # B14: scenario, then year, then the builds in the order written.
        config = {
            "output_folder_prefix": "input",
            "scenarios": ["A", "B"],
            "scenario_years": [2030, 2040],
            "variants": {"": {}, "s1": {"v": {"all": "x"}}},
        }
        assert config_output_folder_names(config) == [
            "input_A_2030", "input_A_2030_s1",
            "input_A_2040", "input_A_2040_s1",
            "input_B_2030", "input_B_2030_s1",
            "input_B_2040", "input_B_2040_s1",
        ]


class TestScenarioTags:
    """The add_scen_tags row. Spine Toolbox reads it by cell position (G4, G5)."""

    def test_a_named_build(self):
        assert scenario_tags("Observed Trends", 2030, "sens1") == [
            "Observed Trends", "2030", "sens1", "ObservedTrends_2030_sens1",
        ]

    def test_the_unsuffixed_build(self):
        assert scenario_tags("Observed Trends", 2030, "") == [
            "Observed Trends", "2030", "", "ObservedTrends_2030",
        ]


class TestSelectedBuilds:
    CONFIG = {"variants": {"": {}, "sens1": {}, "sens2": {}}}

    def test_no_selection_is_every_build_in_order(self):
        # G8
        assert selected_builds(self.CONFIG) == ["", "sens1", "sens2"]

    def test_one_build(self):
        # G6
        assert selected_builds(self.CONFIG, "sens2") == ["sens2"]

    def test_the_selection_matches_case_insensitively(self):
        # Build names are unique case-insensitively (B3), so this is unambiguous.
        assert selected_builds(self.CONFIG, "SENS1") == ["sens1"]

    def test_an_unknown_build_is_refused_listing_the_known_ones(self):
        # G7
        with pytest.raises(ValueError, match="sens1"):
            selected_builds(self.CONFIG, "nope")
