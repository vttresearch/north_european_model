# src/timeseries_pipeline.py
"""
Timeseries pipeline -- orchestration of timeseries processor execution.

Purpose
-------
This module drives the timeseries phase of the build pipeline.
``TimeseriesPipeline.run()`` decides which processors need to execute (based on
cache state), runs or copies them, handles demand grids that have no explicit
processor, and writes all GDX outputs and GAMS include directives required by
Backbone.

Data conventions
----------------
Timeseries DataFrames passed between processors share a common column schema:

    domain(s)  -- one or more Backbone domain columns (str)
    f     -- forecast branch (str, typically 'f00')
    t     -- hourly time step label (str, 't000001' .. 't008760')
    value -- quantity in MWh (negative for demand, positive for supply/influx)

Each processor's output DataFrame may carry different subsets of the Backbone
dimension columns ``['grid', 'node', 'flow', 'group']`` depending on the
parameter it populates.  Domain tracking therefore collects only the columns
that are present in each result:

    ts_domains      : {column_name: set-of-values}
                      e.g. {"grid": {"electricity", "heat"}, "node": {"FI_el"}}

    ts_domain_pairs : {compound_key: set-of-tuples}
                      pairs collected from ``['grid','node']`` and
                      ``['flow','node']`` when both columns exist
                      e.g. {"grid,node": {("electricity", "FI_el"), ...}}

These dicts are accumulated across all processors and merged into shared cache
files (``all_ts_domains.json``, ``all_ts_domain_pairs.json``).  Final
normalization and use of domain names happens downstream in ``BuildInputExcel``.

Secondary results are processor-specific DataFrames or scalars stored under the
processor's Python module name (e.g. ``{"ElecDemandProcessor": <df>}``).  They
are persisted to pickle files in the cache folder so later pipeline phases can
consume them without re-running the processor.

GDX output files are named ``{bb_parameter}_{gdx_name_suffix}.gdx``
(suffix omitted when empty).  For each GDX a matching ``$gdxin`` directive is
appended to ``import_timeseries.inc`` in the output folder.

Output
------
For every executed processor the pipeline writes to ``output_folder``:

- One or more ``.gdx`` files containing Backbone-parameter timeseries.
- Updated ``import_timeseries.inc`` -- GAMS include file that registers each
  GDX for import.
- Optional ``*.csv`` files when ``write_csv_files`` is enabled in config.
- ``Other_demands_1h_MWh.csv`` / ``ts_influx_other_demands.gdx`` for demand
  grids not covered by an explicit processor.

The ``run()`` method returns a ``TimeseriesRunResult`` dataclass with:

- ``secondary_results`` -- dict of processor-name -> secondary output (loaded
  from cache when rebuilding BB Excel so all processors contribute even if they
  did not run this session).
- ``ts_domains`` -- merged domain dict with sorted lists, ready for downstream
  use in ``BuildInputExcel``.
- ``ts_domain_pairs`` -- merged domain-pair dict with sorted tuple lists.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import shutil
import glob as glob_module
import pickle
import pandas as pd
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from src.pipeline.timeseries_processor import ProcessorRunner
import src.utils as utils 
from src.GDX_exchange import write_BB_gdx, update_import_timeseries_inc
import src.json_exchange as json_exchange

@dataclass
class TimeseriesRunResult:
    """Results from the complete timeseries pipeline execution."""
    secondary_results: dict
    ts_domains: dict[str, list]
    ts_domain_pairs: dict[str, list[tuple]]

class TimeseriesPipeline:
    """
    Orchestrates the execution of timeseries processors based on configuration.
    """

    def __init__(self, config: dict, input_folder: Path, output_folder: Path,
                 cache_manager: CacheManager, source_excel_data_pipeline: SourceExcelDataPipeline,
                 reference_ts_folder: Path = None, scenario_year: int = None,
                 logger=None):
        self.config = config
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cache_manager = cache_manager
        self.source_excel_data_pipeline = source_excel_data_pipeline
        self.reference_ts_folder = reference_ts_folder
        self.scenario_year = scenario_year
        self.secondary_results = {}
        self.logger = logger
        self.df_annual_demands = source_excel_data_pipeline.df_demanddata



    def _load_all_processor_specs(self) -> list[dict]:
        """
        Load and validate all timeseries processor specifications from the configuration.

        Each processor spec is checked for required fields:
          - ``processor_name``
          - ``bb_parameter``
          - ``bb_parameter_dimensions``

        If any of these fields are missing, the spec is skipped with a warning.

        Processors whose ``demand_grid`` is in the configured ``exclude_grids`` list
        are also skipped.

        Returns
        -------
        list[dict]
            List of enriched processor specifications.
        """
        specs: list[dict] = []
        timeseries_specs: dict = self.config["timeseries_specs"]
        exclude_grids: list[str] = self.config["exclude_grids"]
        processors_base = Path("src/processors")

        for human_name, spec in timeseries_specs.items():
            processor_name: str | None = spec.get("processor_name")
            bb_parameter: str | None = spec.get("bb_parameter")
            bb_parameter_dimensions: list | None = spec.get("bb_parameter_dimensions")

            if not processor_name or not bb_parameter or not bb_parameter_dimensions:
                self.logger.log_status(
                    f"Timeseries spec '{human_name}' is incomplete (missing processor_name, bb_parameter, or bb_parameter_dimensions). Skipping.",
                    level="warn"
                )
                continue

            demand_grid: str | None = spec.get("demand_grid")
            if demand_grid and demand_grid in exclude_grids:
                self.logger.log_status(
                    f"Skipping {processor_name} due to excluded demand grid: {demand_grid}",
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


    def _create_other_demands(
        self, df_annual_demands: pd.DataFrame, other_demands: set[str]
    ) -> pd.DataFrame:
        """
        Generate hourly demand timeseries for grids not covered by explicit processors.

        For each (grid, node) combination in ``df_annual_demands`` where the grid
        (case-insensitive) is in the set ``other_demands``, this function creates
        8760 hourly rows with the following columns:

          - ``grid``  : grid identifier (copied from input)
          - ``node``  : node identifier (copied from input)
          - ``f``     : fixed to 'f00'
          - ``t``     : hourly time index from 't000001' to 't008760'
          - ``value`` : hourly demand in MWh (negative), computed as
                        (TWh/year * 1e6 / 8760), rounded to two decimals

        Notes
        -----
        - Each input annual demand is assumed to be given in TWh/year.
        - The resulting ``value`` column is negative to represent demand.
        - If required input columns are missing, a warning is logged and an
          empty DataFrame with the correct schema is returned.

        Parameters
        ----------
        df_annual_demands : pd.DataFrame
            Input DataFrame containing at least the columns:
              * ``grid`` (str)
              * ``node`` (str)
              * ``twh/year`` (float)
        other_demands : set[str]
            Set of grid names (lowercased) for which hourly demand timeseries
            should be generated.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [``grid``, ``node``, ``f``, ``t``, ``value``]
            containing 8760 rows per (grid, node) in ``other_demands``.
            If no rows match, or required input columns are missing, returns
            an empty DataFrame with the same columns.
        """
        required_cols = {"grid", "twh/year", "node"}
        missing_cols = required_cols - set(df_annual_demands.columns)

        if missing_cols:
            self.logger.log_status(
                f"[TimeseriesPipeline] Cannot create other demands – missing required columns: {missing_cols}",
                level="warn",
            )
            return pd.DataFrame(columns=["grid", "node", "f", "t", "value"])

        # Filter for rows with unprocessed grid values (case-insensitive)
        df_filtered = df_annual_demands[
            df_annual_demands["grid"].str.lower().isin(other_demands)
        ]

        if df_filtered.empty:
            return pd.DataFrame(columns=["grid", "node", "f", "t", "value"])

        # Create t-index for a full year (8760 hours)
        t_index = [f"t{str(i).zfill(6)}" for i in range(1, 8760 + 1)]

        rows: list[pd.DataFrame] = []
        for _, row in df_filtered.iterrows():
            try:
                # Calculate hourly value (negative demand in MWh)
                hourly_value = round(row["twh/year"] * 1e6 / 8760 * -1, 2)
            except Exception as e:
                self.logger.log_status(
                    f"[TimeseriesPipeline] Failed to calculate hourly demand for row {row.to_dict()}: {e}",
                    level="warn",
                )
                continue

            row_ts = pd.DataFrame(
                {
                    "grid": row["grid"],
                    "node": row["node"],
                    "f": "f00",
                    "t": t_index,
                    "value": hourly_value,
                }
            )
            rows.append(row_ts)

        if rows:
            return pd.concat(rows, ignore_index=True)

        return pd.DataFrame(columns=["grid", "node", "f", "t", "value"])

    def _get_unprocessed_demand_grids(self) -> set[str]:
        """
        Identify demand grids that are not covered by any explicit processor.

        Compares all grids present in ``df_annual_demands`` against the
        ``demand_grid`` fields declared in ``timeseries_specs``.  Grids in
        ``exclude_grids`` are not filtered here; they are excluded upstream in
        ``_load_all_processor_specs``.

        Returns
        -------
        set[str]
            Lowercased grid names present in demand data but not claimed by any
            processor spec.  Empty set if ``df_annual_demands`` is absent or has
            no ``grid`` column.
        """
        # Get all demand grids from annual demands
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

        # Get grids processed by explicit processors
        timeseries_specs = self.config["timeseries_specs"]

        processed_grids = set()
        for spec in timeseries_specs.values():
            demand_grid = spec.get("demand_grid", "").lower()
            if demand_grid:
                processed_grids.add(demand_grid)

        return all_demand_grids - processed_grids


    def _copy_processor_from_reference(self, processor_spec: dict) -> dict:
        """
        Copy outputs of an input-data-independent processor from a reference folder.

        Used when a processor is marked ``is_input_data_dependent: false`` in the
        config and a ``reference_ts_folder`` is configured.  Instead of re-running
        the processor, its pre-built artifacts are reused:

        1. GDX files matching ``{bb_parameter}_{gdx_name_suffix}*.gdx`` are copied
           to ``output_folder`` and registered in ``import_timeseries.inc``.
        2. If ``calculate_average_year`` is set, the ``forecasts`` entry is also
           registered in ``import_timeseries.inc``.
        3. If the spec declares a ``secondary_output_name``, the corresponding
           pickle is copied to the current cache and its content is loaded.
        4. Per-processor domain cache (``processor_domains_{name}.json``) is read
           from the reference cache and re-saved to the current cache.
        5. The processor's hash is copied from the reference cache so the local
           cache manager treats it as up-to-date.

        Parameters
        ----------
        processor_spec : dict
            Enriched processor specification as produced by
            ``_load_all_processor_specs``.

        Returns
        -------
        dict
            Keys:
              * ``secondary_result`` -- loaded secondary output or ``None``.
              * ``ts_domains`` -- domain dict read from the reference cache.
              * ``ts_domain_pairs`` -- domain-pair dict read from the reference cache.
        """
        spec = processor_spec["spec"]
        processor_name = processor_spec["name"]
        human_name = processor_spec["human_name"]
        bb_parameter = spec.get("bb_parameter")
        gdx_name_suffix = spec.get("gdx_name_suffix", "")

        self.logger.log_status(f"{human_name}", section_start_length=45)

        ref_folder = Path(self.reference_ts_folder)

        # Safety check: reference folder must exist
        if not ref_folder.exists():
            self.logger.log_status(
                f"Reference folder {ref_folder} does not exist. Cannot copy.",
                level="warn"
            )
            return {"secondary_result": None, "ts_domains": {}, "ts_domain_pairs": {}}

        # 1. Copy GDX files
        fname_base = f"{bb_parameter}_{gdx_name_suffix}" if gdx_name_suffix else f"{bb_parameter}"
        pattern = str(ref_folder / f"{fname_base}*.gdx")
        gdx_files = glob_module.glob(pattern)

        copied_count = 0
        for gdx_file in gdx_files:
            dest = Path(self.output_folder) / Path(gdx_file).name
            shutil.copy2(gdx_file, dest)
            copied_count += 1

        if copied_count:
            self.logger.log_status(f"Copied {copied_count} GDX file(s) from reference folder", level="info")
        else:
            self.logger.log_status(f"No GDX files found matching {fname_base}*.gdx in reference folder", level="warn")

        # 2. Update import_timeseries.inc for this processor
        bb_kwargs = {"bb_parameter": bb_parameter, "gdx_name_suffix": gdx_name_suffix}
        update_import_timeseries_inc(self.output_folder, **bb_kwargs)

        if spec.get("calculate_average_year", False):
            update_import_timeseries_inc(self.output_folder, file_suffix="forecasts", **bb_kwargs)

        # 3. Copy secondary results (pickle files) if processor has secondary_output_name
        secondary_result = None
        secondary_output_name = spec.get("secondary_output_name")
        if secondary_output_name:
            ref_pkl = ref_folder / "cache" / "secondary_results" / f"{processor_name}.pkl"
            if ref_pkl.exists():
                dest_pkl = self.cache_manager.secondary_results_folder / f"{processor_name}.pkl"
                dest_pkl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ref_pkl, dest_pkl)
                self.logger.log_status(f"Copied secondary result: {processor_name}.pkl", level="info")
                with open(ref_pkl, "rb") as f:
                    pkl_data = pickle.load(f)
                secondary_result = pkl_data.get(secondary_output_name)

        # 4. Load domain data from reference folder's cache
        ref_domain_file = ref_folder / "cache" / f"processor_domains_{processor_name}.json"
        ts_domains = {}
        ts_domain_pairs = {}

        if ref_domain_file.exists():
            domain_cache = json_exchange.load_json(ref_domain_file)
            raw_domains = domain_cache.get("ts_domains", {})
            for key, vals in raw_domains.items():
                ts_domains[key] = set(vals) if isinstance(vals, list) else vals
            raw_pairs = domain_cache.get("ts_domain_pairs", {})
            for key, vals in raw_pairs.items():
                if isinstance(vals, list):
                    ts_domain_pairs[key] = set(tuple(v) for v in vals)
                else:
                    ts_domain_pairs[key] = vals
        else:
            self.logger.log_status(f"No domain cache found at {ref_domain_file}", level="warn")

        # 5. Copy processor hash from reference for cache consistency
        ref_hash_file = ref_folder / "cache" / "processor_hashes.json"
        if ref_hash_file.exists():
            ref_hashes = json_exchange.load_json(ref_hash_file)
            if processor_name in ref_hashes:
                self.cache_manager.save_processor_hash(processor_name, ref_hashes[processor_name])

        # Save domain cache to current output folder's cache as well
        if ts_domains or ts_domain_pairs:
            domain_cache_data = {
                "ts_domains": {k: list(v) for k, v in ts_domains.items()},
                "ts_domain_pairs": {k: [list(t) for t in v] for k, v in ts_domain_pairs.items()}
            }
            domain_file = Path(self.cache_manager.cache_folder) / f"processor_domains_{processor_name}.json"
            json_exchange.save_json(domain_file, domain_cache_data)

        return {
            "secondary_result": secondary_result,
            "ts_domains": ts_domains,
            "ts_domain_pairs": ts_domain_pairs,
        }


    def run(self) -> TimeseriesRunResult:
        """
        Execute the full timeseries processing pipeline.

        Workflow
        --------
        1. Initialization
           - If a full rerun is requested, remove any existing
             ``import_timeseries.inc`` file.
           - Log the start of timeseries processing.

        2. Determine processors to rerun
           - Build a set of processors to run

        3. Run selected processors
           - For each processor to rerun:
             * Execute it with :class:`ProcessorRunner`.
             * Collect secondary results, timeseries domains,
               and domain pairs.
           - For processors marked ``is_input_data_dependent: false`` and a
             ``reference_ts_folder`` is set, copy GDX files and cache data
             from the reference folder via
             :meth:`_copy_processor_from_reference` instead of re-running.

        4. Process other demands
           - Identify demand grids present in ``df_annual_demands`` but not
             covered by explicit processors.
           - Generate hourly timeseries for these with
             :meth:`_create_other_demands`.
           - Write optional CSV/GDX outputs and update
             ``import_timeseries.inc``.

        5. Cache management
           - Merge newly discovered domains and domain pairs into cache.
           - Reload merged results from cache for consistency.
           - Optionally reload secondary results from cache if
             rebuilding Backbone Excel.

        Returns
        -------
        TimeseriesRunResult
            Dataclass containing:
              * ``secondary_results`` : dict
                Secondary outputs from processors or cache.
              * ``ts_domains`` : dict[str, list]
                Mapping of domain → sorted list of values.
              * ``ts_domain_pairs`` : dict[str, list[tuple]]
                Mapping of domain-pair → sorted list of tuples.
        """
        # --- 1. Initialization ---
        # If full rerun, remove import_timeseries.inc
        if self.cache_manager.full_rerun:
            p = Path(self.output_folder) / "import_timeseries.inc"
            p.unlink(missing_ok=True)

        # --- 2. Determine processors to run ---
        self.logger.log_status(
            "Checking the status of timeseries processors",
            level="none",
            add_empty_line_before=True
        )

        # Build set of processors to run
        processors_to_rerun = set()
        self.processors = self._load_all_processor_specs()

        # Get processors marked for rerun by cache manager
        # (includes config changes, code changes, and full rerun flag)
        for proc in self.processors:
            human_name = proc["human_name"]
            needs_rerun = (
                self.cache_manager.full_rerun
                or self.cache_manager.timeseries_changed.get(human_name, False)
            )

            if needs_rerun:
                processors_to_rerun.add(human_name)

        # Separate input-data-independent processors for copying from reference folder
        processors_to_copy = set()
        if self.reference_ts_folder and Path(self.reference_ts_folder) != Path(self.output_folder):
            timeseries_specs_raw = self.config["timeseries_specs"]
            for human_name in list(processors_to_rerun):
                spec = timeseries_specs_raw.get(human_name, {})
                if not spec.get('is_input_data_dependent', True):
                    processors_to_copy.add(human_name)
                    processors_to_rerun.discard(human_name)

        # Log what will run and what will be copied
        self.logger.log_status(
            f"Need to run {len(processors_to_rerun)} timeseries processor(s): "
            f"{', '.join(sorted(processors_to_rerun)) if processors_to_rerun else 'none'}",
            level="info"
        )
        if processors_to_copy:
            self.logger.log_status(
                f"{len(processors_to_copy)} processor(s) will be copied from reference folder: "
                f"{', '.join(sorted(processors_to_copy))}",
                level="info"
            )

        # --- 3. Run selected processors ---
        all_ts_domains = {}
        all_ts_domain_pairs = {}

        if processors_to_rerun:
            processor_iter = (p for p in self.processors if p['human_name'] in processors_to_rerun)

            for processor in processor_iter:
                # --- run processor ---
                runner = ProcessorRunner(
                    processor_spec=processor,
                    config=self.config,
                    input_folder=self.input_folder,
                    output_folder=self.output_folder,
                    source_excel_data_pipeline=self.source_excel_data_pipeline,
                    cache_manager=self.cache_manager,
                    scenario_year=self.scenario_year,
                    logger=self.logger
                )
                self.logger.log_status(f"Running: {processor['name']}", level="run", add_empty_line_before=True)

                # Get structured result
                result = runner.run()

                # Process outputs
                self.secondary_results[result.processor_name] = result.secondary_result

                for dom, vals in result.ts_domains.items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in result.ts_domain_pairs.items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

        # --- 3b. Copy input-data-independent processors from reference folder ---
        if processors_to_copy:
            copy_iter = (p for p in self.processors if p['human_name'] in processors_to_copy)

            for processor in copy_iter:
                self.logger.log_status(f"Copying: {processor['name']}", level="run", add_empty_line_before=True)

                copy_result = self._copy_processor_from_reference(processor)

                self.secondary_results[processor["name"]] = copy_result["secondary_result"]

                for dom, vals in copy_result["ts_domains"].items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in copy_result["ts_domain_pairs"].items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)


        # --- 4. Process Other Demands ---
        self.logger.log_status(f"Remaining timeseries actions", level="run", section_start_length=45, add_empty_line_before=True)

        unprocessed_grids = self._get_unprocessed_demand_grids()

        if unprocessed_grids:
            self.logger.log_status("Processing other demands", level="none")
            for grid in sorted(unprocessed_grids):
                self.logger.log_status(f" .. {grid}", level="none")

            # Create timeseries for other demands
            df_other_demands = self._create_other_demands(self.df_annual_demands, unprocessed_grids)

            # Collect domain info
            other_domains = utils.collect_domains_for_cache(df_other_demands, ['grid', 'node'])
            other_domain_pairs = utils.collect_domain_pairs_for_cache(df_other_demands, [['grid', 'node']])

            for dom, vals in other_domains.items():
                all_ts_domains.setdefault(dom, set()).update(vals)

            for pair_key, tuples in other_domain_pairs.items():
                all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

            if self.config["write_csv_files"]:
                df_other_demands.to_csv(self.output_folder / "Other_demands_1h_MWh.csv")

            # Write gdx file, update GAMS import instructions
            output_file_other = self.output_folder / "ts_influx_other_demands.gdx"
            write_BB_gdx(df_other_demands, str(output_file_other), self.logger,
                         bb_parameter="ts_influx",
                         bb_parameter_dimensions=["grid", "node", "f", "t"])
            update_import_timeseries_inc(self.output_folder, bb_parameter="ts_influx", gdx_name_suffix="other_demands")


        # --- 5. Cache management ---

        # Merge domain data into cache
        self.cache_manager.merge_dict_to_cache(all_ts_domains, "all_ts_domains.json")
        all_ts_domains = self.cache_manager.load_dict_from_cache("all_ts_domains.json")

        self.cache_manager.merge_dict_to_cache(all_ts_domain_pairs, "all_ts_domain_pairs.json")
        all_ts_domain_pairs = self.cache_manager.load_dict_from_cache("all_ts_domain_pairs.json")

        # Load all secondary results (from both current run and cache)
        # If we're rebuilding BB Excel, we need ALL secondary results, not just from this run
        if self.cache_manager.rebuild_bb_excel:
            self.logger.log_status("Loading all secondary results from cache.", level="none")
            all_secondary_results = self.cache_manager.load_all_secondary_results()
            # Merge with results from current run (current run takes precedence)
            all_secondary_results.update(self.secondary_results)
            self.secondary_results = all_secondary_results

        # Returning TimeseriesRunResult dataclass
        return TimeseriesRunResult(
            secondary_results=self.secondary_results,
            ts_domains={k: sorted(v) for k, v in all_ts_domains.items()},
            ts_domain_pairs={k: sorted(v) for k, v in all_ts_domain_pairs.items()},
        )
