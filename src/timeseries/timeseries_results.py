"""
Dataclasses for timeseries processing outputs.

These are extracted here so that both ``base_processor.py`` (in processors/)
and ``timeseries_processor.py`` can import them without creating an upward
dependency from processors into pipeline internals.

Dataclass chain
---------------
ProcessorOutput        -- raw output from a processor's ``run_processor()``
ProcessorRunResult     -- compiled per-processor result (ProcessorRunner → TimeseriesPipeline)
TimeseriesPipelineOutput -- final public output of the timeseries pipeline
"""

from dataclasses import dataclass
from typing import Optional, Any
import pandas as pd


@dataclass
class ProcessorOutput:
    """
    Output returned by a processor's ``run_processor()`` method.

    Attributes
    ----------
    main_result : pd.DataFrame
        Wide-format DataFrame in the processor's own output convention,
        before any BB-format conversion or GDX writing.  Column layout
        varies by processor (e.g. dates as index, countries as columns).
    secondary_result : Any or None
        Optional secondary output produced alongside the main timeseries
        (e.g. annual totals, scaling factors) for use in later pipeline
        stages such as ``BBExcelPipeline``.  ``None`` if the processor
        does not produce a secondary output.
    """
    main_result: pd.DataFrame
    secondary_result: Optional[Any] = None


@dataclass
class ProcessorRunResult:
    """
    Compiled results from a single timeseries processing step.

    This dataclass encapsulates all outputs from running a single
    timeseries processor, including the processed data's domain
    information and any secondary outputs. Passed from ProcessorRunner
    to TimeseriesPipeline for aggregation.

    Attributes
    ----------
    processor_name : str
        Name of the processor that was executed
    secondary_result : Any | None
        Optional secondary output (e.g., metadata, statistics)
        that will be cached for use in other pipeline stages
    ts_domains : dict[str, list]
        Mapping of domain names to lists of values found in the
        processed data (e.g., {'grid': ['FI', 'SE'], 'node': [...]})
    ts_domain_pairs : dict[str, list[tuple]]
        Mapping of domain pair keys to lists of tuples representing
        relationships (e.g., {"grid,node": [("elec", "FI00_elec"), ...]})
    """
    processor_name: str
    secondary_result: Optional[Any]
    ts_domains: dict[str, list]
    ts_domain_pairs: dict[str, list[tuple]]


@dataclass
class TimeseriesPipelineOutput:
    """Results from the complete timeseries pipeline execution."""
    secondary_results: dict
    ts_domains: dict[str, list]
    ts_domain_pairs: dict[str, list[tuple]]
