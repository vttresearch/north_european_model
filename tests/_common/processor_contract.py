"""Harness for exercising the timeseries processor contract.

Processors are the one place where third-party code enters the pipeline: users
add their own, and whatever they return is fed to GDX writing and then to GAMS.
Everything crossing that boundary has to be checked, and the checks themselves
have to be tested -- a guard nobody exercises is a guard nobody can rely on.

``ProcessorRunner`` loads a processor from the path in its own spec
(``timeseries_processor.py:170``), so a fake processor is simply a ``.py`` file
written into ``tmp_path``. No monkeypatching is needed anywhere.

``assert_processor_conforms`` is exported for a different audience: someone
writing their own processor can run it against this gate and find out what is
wrong immediately, rather than from a GAMS error three stages later.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.timeseries.timeseries_processor import ProcessorRunner
from tests._common.contracts import assert_gams_ready
from tests._common.fixtures import FakeLogger, make_config

#: Source template for a fake processor. ``{main_result}`` is spliced in as a
#: Python expression, so a case can return literally anything -- including
#: things that are not DataFrames at all.
FAKE_PROCESSOR_TEMPLATE = '''
import numpy as np
import pandas as pd
from src.timeseries.timeseries_results import ProcessorOutput


class {name}:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run_processor(self):
{body}
        return ProcessorOutput(main_result={main_result})
'''


class StubCacheManager:
    """The slice of CacheManager that ProcessorRunner actually touches."""

    def __init__(self, cache_folder: Path):
        self.cache_folder = cache_folder
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.processor_hashes: dict[str, str] = {}
        self.secondary_results: list[tuple] = []

    def save_processor_hash(self, processor_name: str, hash_value: str) -> None:
        self.processor_hashes[processor_name] = hash_value

    def save_secondary_result(self, processor_name, data, secondary_result_name) -> None:
        self.secondary_results.append((processor_name, data, secondary_result_name))


@dataclass
class FakeRun:
    """Everything a test needs to assert on after running a fake processor."""

    result: Any
    logger: FakeLogger
    output_folder: Path
    cache_manager: StubCacheManager

    @property
    def gdx_files(self) -> list[Path]:
        return sorted(self.output_folder.glob("*.gdx"))

    def assert_no_gdx_written(self) -> None:
        """Assert the *contract*, not the log line.

        Every rejection path promises "No GDX output will be written". Checking
        the message alone would pass even if the promise were broken -- which is
        exactly what happened with the empty-DataFrame branch, whose message said
        one thing while execution carried on regardless.
        """
        files = self.gdx_files
        if files:
            raise AssertionError(
                f"expected no GDX output, found: {[f.name for f in files]}"
            )
        if self.result.ts_domains or self.result.ts_domain_pairs:
            raise AssertionError(
                f"expected an empty ProcessorRunResult, got "
                f"ts_domains={self.result.ts_domains!r} "
                f"ts_domain_pairs={self.result.ts_domain_pairs!r}"
            )


def write_fake_processor(
    folder: Path,
    name: str,
    main_result: str,
    *,
    body: str = "",
) -> Path:
    """Write a one-class processor module returning `main_result`, and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    indented_body = textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8) if body else ""
    source = FAKE_PROCESSOR_TEMPLATE.format(
        name=name, main_result=main_result, body=indented_body
    )
    path = folder / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def run_fake_processor(
    tmp_path: Path,
    main_result: str,
    *,
    dimensions: Sequence[str] = ("grid", "node", "f", "t"),
    name: str = "FakeProcessor",
    body: str = "",
    raw_source: str | None = None,
    config_overrides: dict | None = None,
    spec_overrides: dict | None = None,
) -> FakeRun:
    """Run a synthetic processor through the real ``ProcessorRunner``.

    `raw_source` writes the module verbatim instead of using the template, for
    cases where the *module* is malformed rather than its return value -- e.g. a
    file with no class of the required name.
    """
    if raw_source is not None:
        folder = tmp_path / "processors"
        folder.mkdir(parents=True, exist_ok=True)
        processor_file = folder / f"{name}.py"
        processor_file.write_text(textwrap.dedent(raw_source), encoding="utf-8")
    else:
        processor_file = write_fake_processor(
            tmp_path / "processors", name, main_result, body=body
        )
    output_folder = tmp_path / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    spec = {
        "processor_name": name,
        "bb_parameter": "ts_test",
        "bb_parameter_dimensions": list(dimensions),
        "demand_grid": "",
        "custom_column_value": None,
        "gdx_name_suffix": "",
        "rounding_precision": 0,
        "secondary_output_name": None,
        "input_sub_folder": "",
        "attached_grid": "",
        "is_input_data_dependent": True,
        "scaling_factor": 1,
        "annual_summary": "",
        "cutoff_below": None,
    }
    spec.update(spec_overrides or {})

    logger = FakeLogger()
    cache_manager = StubCacheManager(tmp_path / "cache")
    runner = ProcessorRunner(
        processor_spec={
            "human_name": name,
            "name": name,
            "file": str(processor_file),
            "spec": spec,
        },
        config=make_config(**(config_overrides or {})),
        input_folder=tmp_path / "input",
        output_folder=output_folder,
        cache_manager=cache_manager,
        source_data_pipeline=None,
        logger=logger,
    )

    return FakeRun(
        result=runner.run(),
        logger=logger,
        output_folder=output_folder,
        cache_manager=cache_manager,
    )


def assert_processor_conforms(
    processor_cls: type,
    *,
    dimensions: Sequence[str],
    value_col: str = "value",
    **required_kwargs: Any,
) -> pd.DataFrame:
    """Run a processor class and assert its output meets the documented contract.

    Exported for processor authors: import this in your own test and you get the
    same gate the built-in processors pass, instead of discovering the contract
    from a GAMS error three stages later.

    Checks the interface ``ProcessorRunner`` enforces:

    - ``run_processor()`` returns a ``ProcessorOutput`` carrying a DataFrame;
    - its columns are exactly ``dimensions`` (minus ``t``/``f``) + time + value;
    - ``time`` is datetime and ``value`` is numeric;
    - no dimension value is missing or blank -- these become GAMS set elements;
    - no duplicate ``(dimensions, time)`` rows, which would corrupt t-labelling.

    Missing ``value`` entries are explicitly *allowed*: NaN means "no data" until
    the GDX gate converts it to 0 and reports how many.

    Returns the validated ``main_result`` so callers can assert further.
    """
    required_kwargs.setdefault("logger", FakeLogger())
    instance = processor_cls(**required_kwargs)
    output = instance.run_processor()

    main_result = getattr(output, "main_result", None)
    if not isinstance(main_result, pd.DataFrame):
        raise AssertionError(
            f"{processor_cls.__name__}.run_processor() must return a ProcessorOutput "
            f"whose main_result is a DataFrame; got {type(main_result).__name__}"
        )
    if main_result.empty:
        raise AssertionError(f"{processor_cls.__name__} returned an empty DataFrame")

    group_dims = [d for d in dimensions if d not in ("t", "f")]
    expected = set(group_dims) | {"time", value_col}
    actual = set(main_result.columns)
    if actual != expected:
        raise AssertionError(
            f"{processor_cls.__name__} returned columns {sorted(actual)}, "
            f"expected exactly {sorted(expected)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(main_result["time"]):
        raise AssertionError(
            f"{processor_cls.__name__} returned 'time' as "
            f"{main_result['time'].dtype}, expected datetime"
        )

    duplicates = main_result.duplicated(subset=group_dims + ["time"])
    if duplicates.any():
        raise AssertionError(
            f"{processor_cls.__name__} returned {int(duplicates.sum())} duplicate "
            f"(dimensions, time) row(s); t-label assignment would be corrupted"
        )

    # Dimensions and dtype, but NOT NA in value -- gaps are legal until the gate.
    assert_gams_ready(
        main_result.assign(**{value_col: main_result[value_col].fillna(0)}),
        dimensions=group_dims,
        value_col=value_col,
        where=processor_cls.__name__,
    )
    return main_result
