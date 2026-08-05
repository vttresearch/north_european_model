"""Config factory and logger double.

``make_config`` mirrors the dict that ``config_reader.load_config`` produces.
Because that function returns a *plain dict* and every consumer treats it as
one, tests can synthesise configs directly and never touch configparser --
``.ini`` parsing is then tested once, on its own, in
``tests/unit/infrastructure/test_config_reader.py``.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

# Key-for-key the shape of load_config's return value (config_reader.py:298-341),
# filled with the smallest values that pass its validation.
DEFAULT_CONFIG: dict[str, Any] = {
    # General settings
    "output_folder_prefix": "test",
    "force_full_rerun": False,
    "print_all_elapsed_times": False,
    # Scenario settings
    "scenarios": ["test"],
    "scenario_years": [2030],
    "scenario_alternatives": [""],
    "scenario_alternatives2": [""],
    "scenario_alternatives3": [""],
    "scenario_alternatives4": [""],
    # Climate years -- two years, the minimum that lets forecast code run
    "climate_data": "2014-2015",
    "start_year": 2014,
    "end_year": 2015,
    # Timeseries window -- deliberately tiny; the default of 365 makes every
    # timeseries test 365x slower for no extra coverage.
    "bb_timeseries_start": "01-01",
    "bb_timeseries_length": 2,
    # Topology
    "country_codes": ["FI", "SE"],
    "exclude_grids": [],
    "exclude_nodes": [],
    # Data files -- route helpers usually derive these from the fixture workbooks
    "unittypedata_files": [],
    "nodedata_files": [],
    "emissiondata_files": [],
    "demanddata_files": [],
    "transferdata_files": [],
    "unitdata_files": [],
    "userconstraintdata_files": [],
    # Forecasts
    "forecast_quantiles": {"f01": 0.5, "f02": 0.1, "f03": 0.9},
    "forecast_weights": {"f01": 0.6, "f02": 0.2, "f03": 0.2},
    # No processors by default: the source-data and bb-excel tiers do not need
    # them, and running them would pull in the ~1 GB of real PECD/TYNDP inputs.
    "timeseries_specs": {},
}

_CONFIG_KEYS = frozenset(DEFAULT_CONFIG)


def make_config(**overrides: Any) -> dict[str, Any]:
    """A complete config dict, with `overrides` applied.

    Deep-copied on every call.  ``_validate_timeseries_specs`` mutates the specs
    dict it is handed (``config_reader.py:155`` uses ``entry.setdefault``), so a
    shallow copy would let one test's defaults leak into the next.

    Unknown keys raise rather than being silently accepted -- a typo'd override
    that is quietly ignored produces a test which passes for the wrong reason.
    """
    unknown = set(overrides) - _CONFIG_KEYS
    if unknown:
        raise KeyError(
            f"make_config got unknown config key(s): {sorted(unknown)}. "
            f"Known keys: {sorted(_CONFIG_KEYS)}"
        )
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(copy.deepcopy(overrides))
    return config


class FakeLogger:
    """Stand-in for ``IterationLogger``.

    Implements exactly the surface the pipelines consume -- ``log_status``,
    ``messages``, ``warnings``, ``error_count``, ``has_errors`` -- and nothing
    else, so a change to the real logger's *formatting* cannot break the suite.

    Assertions go through the ``assert_*`` helpers below rather than poking at
    ``messages`` directly, so that a failure reports the whole log rather than
    just ``False``.
    """

    #: Levels the real logger treats as failures (logger.py:75-78).
    WARN_LEVELS = ("warn", "error")

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    # -- write side -------------------------------------------------------

    def log_status(self, message: str, level: str = "none", **_ignored: Any) -> None:
        """Record a message.

        ``**_ignored`` is deliberate.  The real signature carries presentation-only
        kwargs (``section_start_length``, ``add_empty_line_before``,
        ``add_empty_line_after``, ``print_to_screen``); swallowing them means a new
        one can be added to ``IterationLogger`` without editing the test suite.
        That is the no-pinning rule applied to the test double itself.
        """
        self.records.append((level, str(message)))

    # -- read side --------------------------------------------------------

    @property
    def messages(self) -> list[str]:
        return [m for _, m in self.records]

    @property
    def warnings(self) -> list[str]:
        return [m for lvl, m in self.records if lvl in self.WARN_LEVELS]

    @property
    def errors(self) -> list[str]:
        return [m for lvl, m in self.records if lvl == "error"]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    # -- assertions -------------------------------------------------------

    def matching(self, substring: str, *, level: str | None = None) -> list[str]:
        """Messages containing `substring`, optionally restricted to one level."""
        return [
            m
            for lvl, m in self.records
            if substring in m and (level is None or lvl == level)
        ]

    def assert_logged(self, substring: str, *, level: str | None = None) -> None:
        if not self.matching(substring, level=level):
            at = f" at level {level!r}" if level else ""
            raise AssertionError(
                f"expected a log message containing {substring!r}{at}.\n{self._dump()}"
            )

    def assert_not_logged(self, substring: str, *, level: str | None = None) -> None:
        hits = self.matching(substring, level=level)
        if hits:
            at = f" at level {level!r}" if level else ""
            raise AssertionError(
                f"unexpected log message containing {substring!r}{at}:\n"
                + "\n".join(f"  {h}" for h in hits)
            )

    def assert_no_errors(self) -> None:
        if self.has_errors:
            raise AssertionError(
                f"expected no errors, got {self.error_count}:\n{self._dump()}"
            )

    def assert_clean(self) -> None:
        """No errors *and* no warnings."""
        if self.warnings:
            raise AssertionError(
                f"expected a clean log, got {len(self.warnings)} warn/error "
                f"message(s):\n{self._dump()}"
            )

    def _dump(self) -> str:
        if not self.records:
            return "  (nothing was logged)"
        return "\n".join(f"  [{lvl}] {msg}" for lvl, msg in self.records)


def sorted_unique(values: Iterable[Any]) -> list[Any]:
    """Small shared helper: stable, de-duplicated ordering for failure messages."""
    seen: dict[Any, None] = {}
    for v in values:
        seen.setdefault(v, None)
    return list(seen)


def config_files_for(categories: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """Map ``{"unitdata": ["a.xlsx"]}`` to the ``*_files`` keys load_config emits."""
    return {f"{category}_files": files for category, files in categories.items()}
