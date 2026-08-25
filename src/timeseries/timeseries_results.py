"""
Dataclasses for timeseries processing outputs.

Here rather than beside their producers so that ``base_processor.py`` and
``timeseries_processor.py`` can both import them without processors depending
upward on pipeline internals.

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
        Long-format output in the processor's own convention, before any
        BB-format conversion or GDX writing.
    secondary_result : Any or None
        Optional extra output (annual totals, scaling factors) for later
        pipeline stages such as ``BBExcelPipeline``.
    """
    main_result: pd.DataFrame
    secondary_result: Optional[Any] = None


@dataclass
class ProcessorRunResult:
    """
    Compiled results from a single timeseries processing step, passed from
    ProcessorRunner to TimeseriesPipeline for aggregation.

    Attributes
    ----------
    processor_name : str
        Name of the processor that was executed.
    secondary_result : Any | None
        Cached for use in other pipeline stages.
    ts_domains : dict[str, list]
        Domain name -> the values found in the processed data, e.g.
        {'grid': ['elec', 'dheat'], 'node': [...]}.
    ts_domain_pairs : dict[str, list[tuple]]
        Domain pair key -> the combinations found, e.g.
        {"grid,node": [("elec", "FI00_elec"), ...]}.
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
