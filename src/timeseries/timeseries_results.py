"""
Dataclasses for timeseries processing outputs.

Here rather than beside their producers so that ``base_processor.py`` and
``timeseries_processor.py`` can both import them without processors depending
upward on pipeline internals.

Dataclass chain
---------------
ProcessorOutput      -- raw output from a processor's ``run_processor()``
ProcessorRunResult   -- compiled per-processor result (ProcessorRunner → TimeseriesPipeline)

``TimeseriesPipeline.run()`` returns a plain ``{table name: DataFrame}`` dict:
once every output travels the same way there is nothing left for a wrapper
dataclass to name.
"""

from dataclasses import dataclass, field
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
    frames : dict of str to pd.DataFrame
        Contributions to the source data tables, keyed by table name without the
        ``df_`` prefix -- ``{'nodedata': ...}`` is merged into ``df_nodedata``.
        Empty for a processor whose whole output is the time series itself,
        which is most of them. See ``source_data_contributions``.
    """
    main_result: pd.DataFrame
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class ProcessorRunResult:
    """
    Compiled results from a single timeseries processing step, passed from
    ProcessorRunner to TimeseriesPipeline for aggregation.

    Attributes
    ----------
    processor_name : str
        Module and class name of the processor that was executed.
    human_name : str
        The processor's ``timeseries_specs`` key, which is what identifies this
        *run* -- three specs share the VRE_PECD processor, and their outputs are
        three different things.
    frames : dict of str to pd.DataFrame
        Validated contributions, as accepted by ``validate_contribution``.
        Empty on every failure path.
    """
    processor_name: str
    human_name: str
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
