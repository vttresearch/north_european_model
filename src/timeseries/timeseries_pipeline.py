"""
Timeseries pipeline -- orchestration of timeseries processor execution.

``TimeseriesPipeline.run()`` decides which processors need to execute, runs them,
handles demand grids that have no explicit processor, and writes the GDX outputs
and GAMS include directives Backbone needs. What a processor must return, and
what happens to it, is in ``timeseries_processor.py``.

What the pipeline accumulates across processors
-----------------------------------------------
One thing: **contributions to the source data tables**, ``{table name:
DataFrame}``, stacked across every spec that ran. They are merged into
``SourceDataPipeline``'s frames after this phase, and the Excel builder then
reads those frames and nothing else.

Most processors contribute nothing at all. A node, a grid or a flow the model
already has needs no announcing -- it is in the workbooks, which is how the
processor found it. What travels this way is what only the processor knows: that
a node's state boundary comes from a series rather than a constant, or that a
demand grid with no profile of its own gets a flat influx.

Output
------
Per executed processor, into ``output_folder``: one or more
``{bb_parameter}_{gdx_name_suffix}.gdx`` files, and a matching ``$gdxin``
directive appended to ``import_timeseries.inc``.
"""

from pathlib import Path
import pandas as pd
import src.source_data.source_data_contributions as source_data_contributions
from src.infrastructure.cache_manager import CacheManager
from src.source_data.source_data_pipeline import SourceDataPipeline
from src.timeseries.timeseries_inputs import TimeseriesPipelineInputs
from src.timeseries.timeseries_processor import ProcessorRunner


class TimeseriesPipeline:
    """
    Orchestrates the execution of timeseries processors based on configuration.
    """

    def __init__(self, inputs: TimeseriesPipelineInputs):
        self.config = inputs.config
        self.input_folder = inputs.input_folder
        self.output_folder = inputs.output_folder
        self.cache_manager = inputs.cache_manager
        self.source_data_pipeline = inputs.source_data_pipeline
        self.scenario_year = inputs.scenario_year
        self.logger = inputs.logger
        self.df_annual_demands = inputs.source_data_pipeline.df_demanddata



    def _create_enriched_processor_specs(self) -> list[dict]:
        """
        Build the list of enriched processor specs used throughout the pipeline.

        ``config_reader.py`` has already validated the mandatory fields and
        injected the defaults, so nothing is checked here. A processor whose
        ``demand_grid`` is in ``exclude_grids`` is skipped.

        Returns
        -------
        list[dict]
            Per processor: ``human_name`` (the config key), ``name``
            (``processor_name``, used for logging and result tracking), ``file``
            (path to the .py), and ``spec`` (the spec dict from config).
        """
        specs: list[dict] = []
        timeseries_specs: dict = self.config["timeseries_specs"]
        exclude_grids: list[str] = self.config["exclude_grids"]
        # Anchored to this file, not the working directory: the processors live
        # beside it and their location has nothing to do with where the build
        # was started from.
        processors_base = Path(__file__).resolve().parent / "processors"

        for human_name, spec in timeseries_specs.items():
            processor_name: str = spec["processor_name"]

            if spec["demand_grid"] and spec["demand_grid"] in exclude_grids:
                self.logger.log_status(
                    f"Skipping {processor_name}: its demand grid '{spec['demand_grid']}' is in "
                    f"exclude_grids.",
                    level="warn"
                )
                continue

            processor_file = processors_base / f"{processor_name}.py"

            enriched_spec: dict = {
                "name": processor_name,
                "file": str(processor_file),
                "spec": spec,
                "human_name": human_name,
            }
            specs.append(enriched_spec)

        return specs


    def _influx_for_grids_without_a_processor(
        self, df_annual_demands: pd.DataFrame, other_demands: set[str]
    ) -> pd.DataFrame:
        """
        A constant influx for the demand grids no explicit processor covers.

        Every (grid, node) whose grid is in ``other_demands`` gets one number:
        the annual ``twh/year`` spread evenly over the hours, negative because it
        is demand. There is no profile to give it -- a grid reaches here
        precisely because nothing knows its shape.

        A constant, not a series
        ------------------------
        This used to write the same number into every hour of the window, ship it
        as ``ts_influx_other_demands.gdx``, and let ``changes.inc`` collapse it
        back to ``p_gn('influx')`` -- which it does for any timeseries that turns
        out to be flat. Nothing was being detected here, though: the value is a
        constant by construction, and inflating it into 8760 identical rows only
        to have GAMS undo that produced the same model by a longer road.

        Returns
        -------
        pd.DataFrame
            A ``nodedata`` contribution: [``grid``, ``node``, ``influx``], one
            row per (grid, node), or empty with those columns.
        """
        columns = ["grid", "node", "influx"]
        required_cols = {"grid", "twh/year", "node"}
        missing_cols = required_cols - set(df_annual_demands.columns)

        if missing_cols:
            self.logger.log_status(
                f"The demand table has no {', '.join(sorted(missing_cols))} column, so the "
                f"grids without a processor of their own cannot be built: "
                f"{', '.join(sorted(other_demands))}. Those nodes get no demand at all.",
                level="warn",
            )
            return pd.DataFrame(columns=columns)

        df_filtered = df_annual_demands[
            df_annual_demands["grid"].str.lower().isin(other_demands)
        ]

        rows: list[dict] = []
        for _, row in df_filtered.iterrows():
            try:
                # Always a nominal 8760-hour year, whatever bb_timeseries_length
                # is: the rate is per hour, and a longer window repeats it rather
                # than dividing the same energy more thinly.
                hourly_value = round(row["twh/year"] * 1e6 / 8760 * -1, 2)
            except Exception as e:
                self.logger.log_status(
                    f"Node '{row.get('node')}' has a twh/year of {row.get('twh/year')!r}, "
                    f"which is not a number ({e}). It gets no demand at all.",
                    level="warn",
                )
                continue

            rows.append({"grid": row["grid"], "node": row["node"], "influx": hourly_value})

        return pd.DataFrame(rows, columns=columns)

    def _get_unprocessed_demand_grids(self) -> set[str]:
        """
        Which demand grids no explicit processor covers.

        The grids in ``df_annual_demands`` against the ``demand_grid`` fields in
        ``timeseries_specs``. ``exclude_grids`` is not applied here -- those are
        dropped upstream, in ``_create_enriched_processor_specs``.

        Returns
        -------
        set[str]
            Lowercased grid names in the demand data that no processor spec
            claims. Empty if ``df_annual_demands`` is absent or has no ``grid``.
        """
        if (self.df_annual_demands is None
            or self.df_annual_demands.empty
            or "grid" not in self.df_annual_demands):
            return set()

        all_demand_grids = set(
            self.df_annual_demands["grid"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )

        timeseries_specs = self.config["timeseries_specs"]

        processed_grids = set()
        for spec in timeseries_specs.values():
            demand_grid = spec.get("demand_grid", "").lower()
            if demand_grid:
                processed_grids.add(demand_grid)

        return all_demand_grids - processed_grids


    def run(self) -> dict[str, pd.DataFrame]:
        """
        Execute the full timeseries processing pipeline.

        The numbered sections below are the workflow: decide what to run, run it,
        and give the demand grids no processor claims a constant influx.

        Returns
        -------
        dict
            ``{table name: DataFrame}`` -- every contribution this phase made to
            the source data tables, stacked across specs. ``apply_contributions``
            merges them; nothing here does.
        """
        # --- 1. Initialization ---
        # A full rerun starts the include file over rather than appending to a
        # file that still registers GDX files it is about to replace.
        if self.cache_manager.full_rerun:
            p = Path(self.output_folder) / "import_timeseries.inc"
            p.unlink(missing_ok=True)

        # --- 2. Determine processors to run ---
        self.logger.log_status(
            "Checking the status of timeseries processors...",
            level="none",
            add_empty_line_before=True
        )

        processors_to_rerun = set()
        self.processors = self._create_enriched_processor_specs()

        # timeseries_changed already folds in config changes and code changes.
        for proc in self.processors:
            human_name = proc["human_name"]
            needs_rerun = (
                self.cache_manager.full_rerun
                or self.cache_manager.timeseries_changed.get(human_name, False)
            )

            if needs_rerun:
                processors_to_rerun.add(human_name)

        self.logger.log_status(
            f"Need to run {len(processors_to_rerun)} timeseries processor(s): "
            f"{', '.join(sorted(processors_to_rerun)) if processors_to_rerun else 'none'}",
            level="info"
        )

        # Said once here rather than per processor: every processor reaches the
        # same answer, and a window longer than a year is the usual cause.
        start_year: int = self.config["start_year"]
        end_year: int = self.config["end_year"]
        bb_ts_start: str = self.config["bb_timeseries_start"]
        bb_ts_length: int = self.config["bb_timeseries_length"]
        data_end = pd.Timestamp(f"{end_year}-12-31 23:00")

        if processors_to_rerun:
            self.logger.log_status(
                f"Building a {bb_ts_length}-day window from {bb_ts_start} for every climate "
                f"year from {start_year} onwards...",
                level="none"
            )

            excluded_years = [
                yr for yr in range(start_year, end_year + 1)
                if pd.Timestamp(f"{yr}-{bb_ts_start}") + pd.Timedelta(bb_ts_length * 24 - 1, unit="h") > data_end
            ]
            if excluded_years:
                self.logger.log_status(
                    f"Climate years {excluded_years[0]} onwards are not built: a {bb_ts_length}-day "
                    f"window opening on {bb_ts_start} would run past the end of {end_year}.",
                    level="none"
                )

        # --- 3. Run selected processors ---
        # Per spec rather than merged: what each one said is kept apart until the
        # very end, so a stack is all this needs to be.
        contributions: dict[str, dict[str, pd.DataFrame]] = {}

        if processors_to_rerun:
            processor_iter = (p for p in self.processors if p['human_name'] in processors_to_rerun)

            for processor in processor_iter:
                runner = ProcessorRunner(
                    processor_spec=processor,
                    config=self.config,
                    input_folder=self.input_folder,
                    output_folder=self.output_folder,
                    source_data_pipeline=self.source_data_pipeline,
                    cache_manager=self.cache_manager,
                    scenario_year=self.scenario_year,
                    logger=self.logger
                )
                self.logger.log_status(f"Running: {processor['name']}", level="run", add_empty_line_before=True)

                result = runner.run()
                contributions[result.human_name] = result.frames

        # --- 4. Demand grids with no processor of their own ---
        self.logger.log_status(f"Remaining timeseries actions", level="run", section_start_length=45, add_empty_line_before=True)

        unprocessed_grids = self._get_unprocessed_demand_grids()
        other_demands = pd.DataFrame()

        if unprocessed_grids:
            self.logger.log_status("Processing other demands", level="none")
            for grid in sorted(unprocessed_grids):
                self.logger.log_status(f" .. {grid}", level="none")

            other_demands = self._influx_for_grids_without_a_processor(
                self.df_annual_demands, unprocessed_grids
            )
            if not other_demands.empty:
                self.logger.log_status(
                    f"A constant influx for {len(other_demands)} node(s), from their "
                    f"annual demand.",
                    level="info",
                )

        # --- 5. Everything every spec said, including the ones that did not run ---
        #
        # Read back from the cache rather than accumulated in memory: on a
        # partial rerun most specs did not execute, and the workbook still needs
        # what they contributed last time. Nothing merged is ever stored, so this
        # is exactly what each processor returned.
        by_spec = self.cache_manager.load_processor_frames()
        by_spec.update(contributions)

        # Other demands is not cached, deliberately. It is a reading of
        # df_demanddata rather than a processor's own work, it costs nothing to
        # redo, and a cached copy could only ever disagree with the workbook it
        # came from. Recomputing it every run is safe because the workbook is
        # never rebuilt without this phase running: rebuild_bb_excel implies
        # needs_timeseries_run, and it implies reimport_source_excels too.
        contributions_in_order = list(by_spec.values())
        if not other_demands.empty:
            contributions_in_order.append({"nodedata": other_demands})

        return source_data_contributions.combine_contributions(
            contributions_in_order, self.logger
        )
