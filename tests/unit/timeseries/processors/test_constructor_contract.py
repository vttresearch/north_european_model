"""Every processor validates its own kwargs before doing anything.

Processors are loaded dynamically by name, and ``ProcessorRunner`` passes a
single wide kwargs dict built from the config and the spec. A processor that
silently accepted a missing parameter would fail much later -- inside file
reading, or worse, by producing a plausible-looking result from a default it
invented.

Table-driven so adding a processor is one row. The table also serves as the
inventory: every module in ``src/timeseries/processors`` must appear, and a test
checks that, so a new processor cannot join without declaring what it needs.
"""

import importlib.util
import pathlib

import pandas as pd
import pytest

from tests._common.fixtures import FakeLogger

PROCESSOR_DIR = pathlib.Path("src/timeseries/processors")

#: processor name -> the kwargs it refuses to start without.
REQUIRED_KWARGS = {
    # No scenario_year: the demand rows reach it already whitelisted to the
    # scenario year, so it had nothing to do with it. demand_grid instead, which
    # it writes into every output row.
    "DH_demand_fromTemperature": (
        "input_folder", "country_codes", "start_year", "end_year",
        "df_annual_demands", "demand_grid",
    ),
    "elec_demand_TYNDP2024": (
        "input_folder", "country_codes", "start_year", "end_year",
        "df_annual_demands", "scenario_year",
    ),
    "hydro_inflow_MAF2019": ("input_folder", "country_codes", "start_year", "end_year"),
    "hydro_storage_limits_MAF2019": (
        "input_folder", "country_codes", "start_year", "end_year",
        "df_nodedata",
    ),
    "VRE_PECD": (
        "input_folder", "country_codes", "start_year", "end_year", "attached_grid",
    ),
}

#: Values that satisfy the checks without touching disk -- constructors only
#: validate and compute a date range.
SAMPLE_VALUES = {
    "input_folder": ".",
    "country_codes": ["FI"],
    "start_year": 2014,
    "end_year": 2015,
    "df_annual_demands": pd.DataFrame({"grid": ["elec"], "node": ["FI_elec"], "twh/year": [1.0]}),
    "df_nodedata": pd.DataFrame({
        "country": ["FI"], "grid": ["reservoir"], "node": ["FI_reservoir"],
        "upwardlimit": pd.array([1000.0], dtype="Float64"),
    }),
    "scenario_year": 2030,
    "attached_grid": "elec",
    "demand_grid": "dheat",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PROCESSOR_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def _kwargs(name: str, omit: str | None = None) -> dict:
    return {
        key: SAMPLE_VALUES[key]
        for key in REQUIRED_KWARGS[name]
        if key != omit
    } | {"logger": FakeLogger()}


@pytest.mark.parametrize("name", sorted(REQUIRED_KWARGS))
class TestEveryProcessor:
    def test_constructs_when_given_everything_it_asks_for(self, name):
        # Constructors do no I/O, so this exercises the validation and the date
        # range only -- no PECD or TYNDP file is touched.
        assert _load(name)(**_kwargs(name)) is not None

    def test_names_every_missing_parameter_at_once(self, name):
        """One message listing all of them, not one failure per round trip.

        A processor is usually misconfigured in several ways at once when it is
        first written, and reporting them one at a time turns that into a
        guessing game.
        """
        with pytest.raises(ValueError) as excinfo:
            _load(name)(logger=FakeLogger())

        message = str(excinfo.value)
        for parameter in REQUIRED_KWARGS[name]:
            assert parameter in message, f"{name} did not mention {parameter}"

    def test_rejects_each_required_parameter_individually(self, name):
        for parameter in REQUIRED_KWARGS[name]:
            with pytest.raises(ValueError, match=parameter):
                _load(name)(**_kwargs(name, omit=parameter))

    def test_tolerates_the_extra_kwargs_the_runner_passes(self, name):
        """ProcessorRunner hands every processor the whole spec.

        Each one receives keys meant for the others, so a processor that
        rejected unknown kwargs would break as soon as an unrelated spec field
        was added.
        """
        extras = {
            "rounding_precision": 0,
            "cutoff_below": None,
            "gdx_name_suffix": "",
            "some_future_option": "ignored",
        }
        assert _load(name)(**_kwargs(name), **extras) is not None


class TestTheInventoryIsComplete:
    def test_every_processor_module_is_in_the_table(self):
        """A new processor cannot join without declaring what it needs."""
        modules = {
            path.stem
            for path in PROCESSOR_DIR.glob("*.py")
            if path.stem not in ("__init__", "base_processor")
        }
        missing = modules - set(REQUIRED_KWARGS)
        assert not missing, (
            f"processor module(s) {sorted(missing)} are not in REQUIRED_KWARGS. "
            f"Add a row naming the kwargs they refuse to start without."
        )

    def test_every_module_defines_a_class_of_the_same_name(self):
        # ProcessorRunner loads by name and reports a warning otherwise; this
        # catches the mismatch in the suite instead of at build time.
        for name in REQUIRED_KWARGS:
            assert _load(name).__name__ == name
