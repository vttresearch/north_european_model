"""Tests for ``src/infrastructure/config_reader.py``.

Config parsing runs *before* the logger exists, so the error policy here is the
opposite of the rest of the pipeline: these functions raise rather than log
(CLAUDE.md, "Error handling policy").  Tests therefore use ``pytest.raises``,
matching on a stable substring rather than the whole message.

The parsing and range arithmetic is pinned exactly -- it is the contract
(pinning case 2 in tests/README.md).
"""

import textwrap

import pytest

from src.infrastructure.config_reader import (
    FORECAST_BRANCH_ENDS,
    _parse_bb_timeseries_start,
    _parse_climate_data,
    _parse_forecast_branches,
    _safe_eval_int,
    _validate_timeseries_specs,
    config_output_folder_names,
    load_config,
    output_folder_name,
    branch_energy_days,
    spec_energy_quantiles,
)

MINIMAL_INI = """\
[inputdata]
scenarios = ['test']
scenario_years = [2030]
climate_data = 2014-2015
country_codes = ['FI', 'SE']
"""


def _write_ini(tmp_path, body: str):
    path = tmp_path / "config.ini"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


class TestParseClimateData:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("2014", (2014, 2014)),
            ("2014-2016", (2014, 2016)),
            (" 1982-2016 ", (1982, 2016)),
            ("1982", (1982, 1982)),
            ("2016", (2016, 2016)),
        ],
    )
    def test_accepts_a_single_year_or_an_inclusive_range(self, value, expected):
        assert _parse_climate_data(value) == expected

    @pytest.mark.parametrize(
        "value, reason",
        [
            ("14-16", "Invalid climate_data format"),
            ("2014-", "Invalid climate_data format"),
            ("2014/2016", "Invalid climate_data format"),
            ("", "Invalid climate_data format"),
            ("1981-2016", "between 1982 and 2016"),
            ("2014-2017", "between 1982 and 2016"),
            ("2016-2014", "must not be later than"),
        ],
    )
    def test_rejects_bad_formats_and_out_of_range_years(self, value, reason):
        with pytest.raises(ValueError, match=reason):
            _parse_climate_data(value)


class TestParseBbTimeseriesStart:
    @pytest.mark.parametrize("value", ["01-01", "07-01", "12-31", " 02-29 "])
    def test_accepts_valid_month_day(self, value):
        assert _parse_bb_timeseries_start(value) == value.strip()

    @pytest.mark.parametrize(
        "value, reason",
        [
            ("1-1", "Invalid bb_timeseries_start format"),
            ("2030-01-01", "Invalid bb_timeseries_start format"),
            ("13-01", "month 13 is out of range"),
            ("00-01", "month 0 is out of range"),
            ("01-32", "day 32 is out of range"),
            ("01-00", "day 0 is out of range"),
        ],
    )
    def test_rejects_bad_formats_and_out_of_range_values(self, value, reason):
        with pytest.raises(ValueError, match=reason):
            _parse_bb_timeseries_start(value)


class TestSafeEvalInt:
    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("365", 365),
            ("365*5", 1825),
            ("24 * 7", 168),
            ("(365 + 1) * 2", 732),
            ("730 / 2", 365),
            ("7 // 2", 3),
            ("2 ** 10", 1024),
            ("-5 + 10", 5),
        ],
    )
    def test_evaluates_plain_arithmetic(self, expr, expected):
        # bb_timeseries_length accepts expressions so that "365*5" reads as
        # five years rather than as an opaque 1825.
        assert _safe_eval_int(expr) == expected

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('echo pwned')",
            "open('secret.txt').read()",
            "some_name",
            "len([1,2,3])",
            "().__class__",
        ],
    )
    def test_rejects_anything_that_is_not_arithmetic(self, expr):
        # The AST whitelist is a security boundary, not a convenience check:
        # config files are data, and eval on data must not reach the runtime.
        with pytest.raises((ValueError, SyntaxError)):
            _safe_eval_int(expr)

    def test_rejects_a_non_integer_result(self):
        with pytest.raises(ValueError, match="not integer-valued"):
            _safe_eval_int("365 / 2")

    def test_accepts_a_float_that_is_integer_valued(self):
        assert _safe_eval_int("730 / 2") == 365


class TestValidateTimeseriesSpecs:
    def _spec(self, **overrides):
        spec = {
            "processor_name": "VRE_PECD",
            "bb_parameter": "ts_cf",
            "bb_parameter_dimensions": ["flow", "node", "f", "t"],
        }
        spec.update(overrides)
        return {"pv": spec}

    def test_fills_optional_fields_with_defaults(self):
        out = _validate_timeseries_specs(self._spec())
        entry = out["pv"]
        assert entry["rounding_precision"] == 0
        assert entry["gdx_name_suffix"] == ""
        assert entry["cutoff_below"] is None

    def test_does_not_overwrite_values_the_user_set(self):
        out = _validate_timeseries_specs(self._spec(rounding_precision=4))
        assert out["pv"]["rounding_precision"] == 4

    @pytest.mark.parametrize(
        "missing", ["processor_name", "bb_parameter", "bb_parameter_dimensions"]
    )
    def test_rejects_a_spec_missing_a_mandatory_field(self, missing):
        spec = self._spec()
        del spec["pv"][missing]
        with pytest.raises(ValueError, match=missing):
            _validate_timeseries_specs(spec)

    @pytest.mark.parametrize("bad", [[], "a string", 42, None])
    def test_rejects_specs_that_are_not_a_dict(self, bad):
        with pytest.raises(ValueError, match="must be a dictionary"):
            _validate_timeseries_specs(bad)

    def test_rejects_an_entry_that_is_not_a_dict(self):
        with pytest.raises(ValueError, match="entry 'pv' must be a dictionary"):
            _validate_timeseries_specs({"pv": ["not", "a", "dict"]})

    def test_mutates_the_dict_it_is_given(self):
        """Characterisation, not endorsement.

        ``entry.setdefault`` at config_reader.py:155 writes into the caller's
        dict. Harmless in production (it runs once, on a freshly parsed literal)
        but it means any test fixture sharing a specs dict leaks defaults into
        the next test -- which is why make_config deep-copies.
        """
        spec = self._spec()
        _validate_timeseries_specs(spec)
        assert "rounding_precision" in spec["pv"]


class TestLoadConfig:
    def test_reads_a_minimal_config(self, tmp_path):
        config = load_config(_write_ini(tmp_path, MINIMAL_INI))
        assert config["scenarios"] == ["test"]
        assert config["country_codes"] == ["FI", "SE"]
        assert (config["start_year"], config["end_year"]) == (2014, 2015)

    def test_applies_documented_defaults(self, tmp_path):
        config = load_config(_write_ini(tmp_path, MINIMAL_INI))
        assert config["output_folder_prefix"] == "output"
        assert config["force_full_rerun"] is False
        assert config["bb_timeseries_start"] == "01-01"
        assert config["bb_timeseries_length"] == 365
        assert config["bb_horizon_weeks"] == 52
        assert config["timeseries_specs"] == {}
        assert config["exclude_grids"] == []

    @pytest.mark.parametrize(
        "missing", ["scenarios", "scenario_years", "climate_data", "country_codes"]
    )
    def test_rejects_a_config_missing_a_mandatory_key(self, tmp_path, missing):
        body = "\n".join(
            line for line in MINIMAL_INI.splitlines() if not line.startswith(missing)
        )
        with pytest.raises(ValueError, match=missing):
            load_config(_write_ini(tmp_path, body + "\n"))

    def test_rejects_a_file_without_an_inputdata_section(self, tmp_path):
        with pytest.raises(ValueError, match="inputdata"):
            load_config(_write_ini(tmp_path, "[other]\nscenarios = ['x']\n"))

    def test_bb_timeseries_length_accepts_an_expression(self, tmp_path):
        # Climate range widened to fit: a 5-year window needs 5 years of data.
        body = MINIMAL_INI.replace("climate_data = 2014-2015", "climate_data = 2005-2016")
        config = load_config(_write_ini(tmp_path, body + "bb_timeseries_length = 365*5\n"))
        assert config["bb_timeseries_length"] == 1825

    def test_bb_horizon_weeks_accepts_an_expression(self, tmp_path):
        config = load_config(_write_ini(tmp_path, MINIMAL_INI + "bb_horizon_weeks = 52+18\n"))
        assert config["bb_horizon_weeks"] == 70

    @pytest.mark.parametrize("value", ["2", "157", "70.5", "seventy"])
    def test_rejects_a_horizon_outside_whole_weeks_3_to_156(self, tmp_path, value):
        # Below 3 the weekly interval block has no room after the first two weeks;
        # a fraction would leave a partial week the block cannot step through.
        with pytest.raises(ValueError, match="bb_horizon_weeks"):
            load_config(_write_ini(tmp_path, MINIMAL_INI + f"bb_horizon_weeks = {value}\n"))

    def test_rejects_a_window_longer_than_the_available_climate_data(self, tmp_path):
        """Cross-validation at config_reader.py:226-241.

        Worth its own test because the failure it prevents is silent: without
        it the build would run and produce a timeseries that simply stops part
        way through the requested horizon.
        """
        with pytest.raises(ValueError, match="complete 1825-day window"):
            load_config(
                _write_ini(tmp_path, MINIMAL_INI + "bb_timeseries_length = 365*5\n")
            )

    def test_empty_scenario_alternatives_are_normalised_to_one_blank_entry(self, tmp_path):
        # The scenario loop is a cartesian product; an empty axis would multiply
        # out to zero iterations and silently build nothing.
        config = load_config(
            _write_ini(tmp_path, MINIMAL_INI + "scenario_alternatives = []\n")
        )
        assert config["scenario_alternatives"] == [""]

    @pytest.mark.parametrize("key", ["fueldata_files", "storagedata_files"])
    def test_rejects_the_deprecated_data_file_keys(self, tmp_path, key):
        # These were merged into nodedata_files. Failing loudly beats silently
        # loading nothing, which is what the old unitTest.xlsx fixture did.
        with pytest.raises(ValueError, match="no longer supported"):
            load_config(_write_ini(tmp_path, MINIMAL_INI + f"{key} = ['x.xlsx']\n"))

    def test_rejects_forecast_weights_that_do_not_sum_to_one(self, tmp_path):
        body = MINIMAL_INI + (
            "energy_quantiles = {'f01': 0.5, 'f02': 0.9}\n"
            "forecast_weights = {'f01': 0.5, 'f02': 0.9}\n"
        )
        with pytest.raises(ValueError):
            load_config(_write_ini(tmp_path, body))

    def test_rejects_forecast_weights_whose_keys_do_not_match_the_quantiles(self, tmp_path):
        body = MINIMAL_INI + (
            "energy_quantiles = {'f01': 0.5}\n"
            "forecast_weights = {'f02': 1.0}\n"
        )
        with pytest.raises(ValueError):
            load_config(_write_ini(tmp_path, body))

    def test_rejects_f00_as_a_forecast_branch(self, tmp_path):
        # f00 is the realized weather branch, not a forecast.
        body = MINIMAL_INI + "energy_quantiles = {'f00': 0.5}\n"
        with pytest.raises(ValueError):
            load_config(_write_ini(tmp_path, body))

    @pytest.mark.parametrize("value", ["1.5", "-0.1", "'low'", "True"])
    def test_rejects_an_energy_quantile_outside_zero_to_one(self, tmp_path, value):
        body = MINIMAL_INI + f"energy_quantiles = {{'f01': {value}}}\n"
        with pytest.raises(ValueError, match="between 0 and 1"):
            load_config(_write_ini(tmp_path, body))

    def test_the_old_key_stops_the_build_rather_than_being_read_another_way(self, tmp_path):
        # forecast_quantiles took a quantile of every hour; energy_quantiles takes
        # one of energy. Reading an old config under the new meaning would change
        # every branch without a word, so the old key is refused with the reason.
        body = MINIMAL_INI + "forecast_quantiles = {'f01': 0.5, 'f02': 0.1}\n"
        with pytest.raises(ValueError, match="now energy_quantiles"):
            load_config(_write_ini(tmp_path, body))

    def test_the_old_key_in_a_spec_stops_the_build_too(self, tmp_path):
        body = MINIMAL_INI + textwrap.dedent("""\
            timeseries_specs = {
                'wind': {'processor_name': 'VRE_PECD', 'bb_parameter': 'ts_cf',
                         'bb_parameter_dimensions': ['flow', 'node', 'f', 't'],
                         'forecast_quantiles': {'f02': 0.45}},
                }
            """)
        with pytest.raises(ValueError, match="'wind': forecast_quantiles is now energy_quantiles"):
            load_config(_write_ini(tmp_path, body))

    def test_a_branch_s_energy_is_measured_over_its_own_length(self, tmp_path):
        body = MINIMAL_INI + textwrap.dedent("""\
            energy_quantiles = {'f01': 0.5, 'f02': 0.1, 'f03': 0.9}
            forecast_weights = {'f01': 0.6, 'f02': 0.2, 'f03': 0.2}
            forecast_branches = {'f02': {'length_days': 252, 'end': 'continue'}}
            """)
        config = load_config(_write_ini(tmp_path, body))
        # f01 is not named: it reaches the horizon and is measured over the window.
        assert branch_energy_days(config) == {"f02": 252, "f03": 149}

    def test_a_spec_quantile_replaces_the_global_one_for_that_series_only(self, tmp_path):
        body = MINIMAL_INI + textwrap.dedent("""\
            energy_quantiles = {'f01': 0.5, 'f02': 0.1}
            timeseries_specs = {
                'wind': {'processor_name': 'VRE_PECD', 'bb_parameter': 'ts_cf',
                         'bb_parameter_dimensions': ['flow', 'node', 'f', 't'],
                         'energy_quantiles': {'f02': 0.45}},
                'hydro': {'processor_name': 'hydro_inflow_MAF2019', 'bb_parameter': 'ts_influx',
                          'bb_parameter_dimensions': ['grid', 'node', 'f', 't']},
                }
            """)
        config = load_config(_write_ini(tmp_path, body))
        specs = config["timeseries_specs"]
        assert spec_energy_quantiles(config, specs["wind"]) == {"f01": 0.5, "f02": 0.45}
        assert spec_energy_quantiles(config, specs["hydro"]) == {"f01": 0.5, "f02": 0.1}
        # The global map is the default, not a copy the override writes into.
        assert config["energy_quantiles"] == {"f01": 0.5, "f02": 0.1}

    @pytest.mark.parametrize(
        "override, reason",
        [
            ("{'f09': 0.5}", "does not have"),
            ("{'f02': 1.5}", "between 0 and 1"),
            ("{'f02': 'low'}", "between 0 and 1"),
            ("[0.45]", "must be a dict"),
        ],
    )
    def test_rejects_a_spec_quantile_the_branches_cannot_take(self, tmp_path, override, reason):
        body = MINIMAL_INI + textwrap.dedent(f"""\
            energy_quantiles = {{'f01': 0.5, 'f02': 0.1}}
            timeseries_specs = {{
                'wind': {{'processor_name': 'VRE_PECD', 'bb_parameter': 'ts_cf',
                         'bb_parameter_dimensions': ['flow', 'node', 'f', 't'],
                         'energy_quantiles': {override}}},
                }}
            """)
        with pytest.raises(ValueError, match=reason):
            load_config(_write_ini(tmp_path, body))

    def test_returns_a_plain_dict(self, tmp_path):
        # The main DI seam: because this is a plain dict, tests everywhere else
        # can synthesise configs without touching configparser.
        assert type(load_config(_write_ini(tmp_path, MINIMAL_INI))) is dict


class TestForecastBranches:
    """``forecast_branches`` says how long each branch is its own and how it ends.

    The parser returns every branch beside the central one, filled in, because
    ``scheduleInit.gms`` is written from it: a branch the config says nothing
    about still needs its length stated.
    """

    QUANTILES = {"f01": 0.5, "f02": 0.1, "f03": 0.9, "f04": 0.05}

    def _parse(self, raw, quantiles=None, weeks=70):
        return _parse_forecast_branches(raw, self.QUANTILES if quantiles is None else quantiles, weeks)

    def test_a_branch_the_config_does_not_name_is_149_days_and_cut(self):
        out = self._parse(None)
        assert list(out) == ["f02", "f03", "f04"]
        assert all(b == {"length_days": 149, "end": "cut", "blend_days": 0} for b in out.values())

    def test_stated_settings_are_kept_and_the_rest_filled_in(self):
        out = self._parse({
            "f01": "central",
            "f02": {"length_days": 252, "end": "continue", "blend_days": 28},
            "f04": {"length_days": 5},
        })
        assert out["f02"] == {"length_days": 252, "end": "continue", "blend_days": 28}
        assert out["f03"] == {"length_days": 149, "end": "cut", "blend_days": 0}
        assert out["f04"] == {"length_days": 5, "end": "cut", "blend_days": 0}
        # The central branch takes no settings, so it is not among the results.
        assert "f01" not in out

    def test_every_end_has_a_backbone_value(self):
        # 'end' is written into scheduleInit.gms as boundForecastEnds.
        assert FORECAST_BRANCH_ENDS == {"cut": 0, "bound": 1, "continue": 2}

    def test_only_f01_can_be_central(self):
        # scheduleInit.gms names f01 in mf_central and changes.inc reads its data by
        # label, so another central branch would be central in the config alone.
        with pytest.raises(ValueError, match="cannot be 'central'"):
            self._parse({"f01": "central", "f02": "central"})

    def test_f01_takes_nothing_but_central(self):
        with pytest.raises(ValueError, match="must be 'central'"):
            self._parse({"f01": {"length_days": 100}})

    def test_branches_need_f01_among_the_quantiles(self):
        with pytest.raises(ValueError, match="must include 'f01'"):
            self._parse(None, quantiles={"f02": 0.1})

    @pytest.mark.parametrize(
        "raw, reason",
        [
            ({"f09": {"length_days": 5}}, "does not have"),
            ({"f02": {"length": 5}}, "unknown setting"),
            ({"f02": {"end": "stop"}}, "must be one of"),
            ({"f02": {"length_days": 1}}, "between 2 and"),
            ({"f02": {"length_days": 491}}, "between 2 and"),
            ({"f02": {"length_days": 5.5}}, "whole number"),
            ({"f02": {"length_days": True}}, "whole number"),
            ({"f02": {"blend_days": -1, "end": "continue"}}, "must not be negative"),
            ({"f02": {"blend_days": 7, "end": "cut"}}, "belongs to"),
            ({"f02": 252}, "must be a dict"),
            (["f02"], "must be a dict"),
        ],
    )
    def test_rejects_what_the_templates_could_not_honour(self, raw, reason):
        with pytest.raises(ValueError, match=reason):
            self._parse(raw)

    def test_the_longest_length_is_the_horizon(self):
        assert self._parse({"f02": {"length_days": 490}})["f02"]["length_days"] == 490
        assert self._parse({"f02": {"length_days": 21}}, weeks=3)["f02"]["length_days"] == 21
        with pytest.raises(ValueError, match="between 2 and"):
            self._parse({"f02": {"length_days": 22}}, weeks=3)

    def test_the_default_length_gives_way_to_a_shorter_horizon(self):
        # A config that says nothing about its branches must still load with a
        # horizon under 149 days.
        out = self._parse(None, weeks=3)
        assert [b["length_days"] for b in out.values()] == [21, 21, 21]

    def test_a_deterministic_config_has_no_branches(self):
        assert self._parse(None, quantiles={}) == {}
        with pytest.raises(ValueError, match="deterministic"):
            self._parse({"f02": {"length_days": 5}}, quantiles={})

    def test_load_config_reads_the_key(self, tmp_path):
        body = MINIMAL_INI + textwrap.dedent("""\
            forecast_branches = {
                'f01': 'central',
                'f03': {'length_days': 5, 'end': 'bound'},
                }
            """)
        config = load_config(_write_ini(tmp_path, body))
        assert config["forecast_branches"]["f03"] == {"length_days": 5, "end": "bound", "blend_days": 0}
        assert config["forecast_branches"]["f02"]["length_days"] == 149


class TestOutputFolderName:
    """The folder-name rule is shared by the builder and by ``run_model.py``.

    ``build_input_data.py`` uses it to decide where to write and ``run_model.py``
    to decide what to pass as Backbone's ``--input_dir``, so a run cannot name a
    folder a build would not have written. Pinned exactly: it names folders on
    disk, and a change here silently points a run somewhere else.
    """

    def test_spaces_are_removed_from_every_part(self):
        assert output_folder_name("input", "Observed Trends", 2030) == "input_ObservedTrends_2030"

    def test_prefix_may_itself_carry_underscores(self):
        assert output_folder_name(
            "input_tyndp2024", "National Trends", 2030
        ) == "input_tyndp2024_NationalTrends_2030"

    def test_year_is_stringified(self):
        assert output_folder_name("input", "s", 2030) == "input_s_2030"
        assert output_folder_name("input", "s", "2030") == "input_s_2030"

    def test_empty_alternatives_add_no_segment(self):
        assert output_folder_name("input", "s", 2030, ["", "", "", ""]) == "input_s_2030"

    def test_active_alternatives_append_in_order(self):
        assert output_folder_name(
            "input", "s", 2030, ["alt1", "", "alt3", ""]
        ) == "input_s_2030_alt1_alt3"

    def test_alternatives_default_to_none(self):
        assert output_folder_name("input", "s", 2030) == "input_s_2030"


class TestConfigOutputFolderNames:
    """Every folder a build writes, in the order the builder's product loop writes them."""

    def test_single_combination(self, tmp_path):
        config = load_config(_write_ini(tmp_path, MINIMAL_INI))
        assert config_output_folder_names(config) == ["output_test_2030"]

    def test_cartesian_product_order(self):
        config = {
            "output_folder_prefix": "input",
            "scenarios": ["A", "B"],
            "scenario_years": [2030, 2040],
            "scenario_alternatives": ["x", "y"],
            "scenario_alternatives2": [""],
            "scenario_alternatives3": [""],
            "scenario_alternatives4": [""],
        }
        assert config_output_folder_names(config) == [
            "input_A_2030_x", "input_A_2030_y",
            "input_A_2040_x", "input_A_2040_y",
            "input_B_2030_x", "input_B_2030_y",
            "input_B_2040_x", "input_B_2040_y",
        ]

    def test_all_four_alternative_axes_contribute(self):
        config = {
            "output_folder_prefix": "input",
            "scenarios": ["s"],
            "scenario_years": [2030],
            "scenario_alternatives": ["a1"],
            "scenario_alternatives2": ["a2"],
            "scenario_alternatives3": ["a3"],
            "scenario_alternatives4": ["a4"],
        }
        assert config_output_folder_names(config) == ["input_s_2030_a1_a2_a3_a4"]
