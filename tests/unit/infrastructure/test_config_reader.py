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
    _parse_bb_timeseries_start,
    _parse_climate_data,
    _safe_eval_int,
    _validate_timeseries_specs,
    load_config,
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
        assert entry["is_input_data_dependent"] is True
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
            "forecast_quantiles = {'f01': 0.5, 'f02': 0.9}\n"
            "forecast_weights = {'f01': 0.5, 'f02': 0.9}\n"
        )
        with pytest.raises(ValueError):
            load_config(_write_ini(tmp_path, body))

    def test_rejects_forecast_weights_whose_keys_do_not_match_the_quantiles(self, tmp_path):
        body = MINIMAL_INI + (
            "forecast_quantiles = {'f01': 0.5}\n"
            "forecast_weights = {'f02': 1.0}\n"
        )
        with pytest.raises(ValueError):
            load_config(_write_ini(tmp_path, body))

    def test_rejects_f00_as_a_forecast_quantile(self, tmp_path):
        # f00 is the realized weather branch, not a quantile.
        body = MINIMAL_INI + "forecast_quantiles = {'f00': 0.5}\n"
        with pytest.raises(ValueError):
            load_config(_write_ini(tmp_path, body))

    def test_returns_a_plain_dict(self, tmp_path):
        # The main DI seam: because this is a plain dict, tests everywhere else
        # can synthesise configs without touching configparser.
        assert type(load_config(_write_ini(tmp_path, MINIMAL_INI))) is dict
