# src/timeseries_pipeline.py

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
from src.utils import log_status
from src.utils import collect_domains, collect_domain_pairs
from src.GDX_exchange import write_BB_gdx, update_import_timeseries_inc
import src.json_exchange as json_exchange

@dataclass
class TimeseriesRunResult:
    """Results from the complete timeseries pipeline execution."""
    secondary_results: dict
    ts_domains: dict[str, list]
    ts_domain_pairs: dict[str, list[tuple]]
    logs: list[str]

class TimeseriesPipeline:
    """
    Orchestrates the execution of timeseries processors based on configuration.
    """
    
    def __init__(self, config: dict, input_folder: Path, output_folder: Path,
                 cache_manager: CacheManager, source_excel_data_pipeline: SourceExcelDataPipeline,
                 reference_ts_folder: Path = None, scenario_year: int = None):
        self.config = config
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cache_manager = cache_manager
        self.source_excel_data_pipeline = source_excel_data_pipeline
        self.reference_ts_folder = reference_ts_folder
        self.scenario_year = scenario_year
        self.secondary_results = {}
        self.logs = []
        self.df_annual_demands = source_excel_data_pipeline.df_demanddata



    def _load_all_processor_specs(self) -> tuple[list[dict], list[str]]:
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
        tuple[list[dict], list[str]]
            A tuple containing:
              * list of enriched processor specifications
              * list of log messages generated during spec validation
        """
        specs: list[dict] = []
        specs_logs: list[str] = []
        timeseries_specs: dict = self.config.get("timeseries_specs", {})
        exclude_grids: list[str] = self.config.get("exclude_grids", [])
        processors_base = Path("src/processors")

        for human_name, spec in timeseries_specs.items():
            processor_name: str | None = spec.get("processor_name")
            bb_parameter: str | None = spec.get("bb_parameter")
            bb_parameter_dimensions: list | None = spec.get("bb_parameter_dimensions")

            if not processor_name or not bb_parameter or not bb_parameter_dimensions:
                log_status(f"Timeseries spec '{human_name}' is incomplete (missing processor_name, bb_parameter, or bb_parameter_dimensions). Skipping.",
                           specs_logs, level="warn")
                continue

            demand_grid: str | None = spec.get("demand_grid")
            if demand_grid and demand_grid in exclude_grids:
                log_status(f"Skipping {processor_name} due to excluded demand grid: {demand_grid}", 
                           specs_logs, level="warn")
                continue
            
            if spec.get("disabled", False):
                log_status(f"Skipping '{processor_name}' due 'disable' flag in timeseries_specs in config file", 
                           specs_logs, level="info")

            processor_file = processors_base / f"{processor_name}.py"

            enriched_spec: dict = {
                "name": processor_name,
                "file": str(processor_file),
                "spec": spec,
                "human_name": human_name,
                "disabled": bool(spec.get("disabled", False)),
                "reserve_grid_when_disabled": bool(spec.get("reserve_grid_when_disabled", True)),
            }
            specs.append(enriched_spec)

        return specs, specs_logs


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
            log_status(
                f"[TimeseriesPipeline] Cannot create other demands – missing required columns: {missing_cols}",
                self.logs,
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
                log_status(
                    f"[TimeseriesPipeline] Failed to calculate hourly demand for row {row.to_dict()}: {e}",
                    self.logs,
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
        Identify demand grids that aren't covered by explicit processors.
        
        Returns:
            Set of lowercased grid names that need default timeseries generation
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
        
        # Get grids processed by enabled processors
        timeseries_specs = self.config.get("timeseries_specs", {})
        disable_all = self.config.get('disable_all_ts_processors', False)
        
        processed_grids = set()
        for spec in timeseries_specs.values():
            demand_grid = spec.get("demand_grid", "").lower()
            if not demand_grid:
                continue
            
            is_disabled = spec.get("disabled", False) or disable_all
            should_reserve = spec.get("reserve_grid_when_disabled", True)
            
            # Grid is "processed" if either:
            # 1. Processor is enabled, OR
            # 2. Processor is disabled but reserves the grid
            if not is_disabled or (is_disabled and should_reserve):
                processed_grids.add(demand_grid)
        
        return all_demand_grids - processed_grids


    def _copy_processor_from_reference(self, processor_spec: dict) -> dict:
        """
        Copy GDX files and cache data for a single input-data-independent processor
        from the reference folder instead of re-running it.

        Returns a dict with keys: secondary_result, ts_domains, ts_domain_pairs, log_messages
        """
        spec = processor_spec["spec"]
        processor_name = processor_spec["name"]
        human_name = processor_spec["human_name"]
        bb_parameter = spec.get("bb_parameter")
        gdx_name_suffix = spec.get("gdx_name_suffix", "")

        copy_logs = []
        log_status(f"{human_name}", copy_logs, section_start_length=45)

        ref_folder = Path(self.reference_ts_folder)

        # Safety check: reference folder must exist
        if not ref_folder.exists():
            log_status(
                f"Reference folder {ref_folder} does not exist. Cannot copy.",
                copy_logs, level="warn"
            )
            return {"secondary_result": None, "ts_domains": {}, "ts_domain_pairs": {}, "log_messages": copy_logs}

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
            log_status(f"Copied {copied_count} GDX file(s) from reference folder", copy_logs, level="info")
        else:
            log_status(f"No GDX files found matching {fname_base}*.gdx in reference folder", copy_logs, level="warn")

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
                log_status(f"Copied secondary result: {processor_name}.pkl", copy_logs, level="info")
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
            log_status(f"No domain cache found at {ref_domain_file}", copy_logs, level="warn")

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
            "log_messages": copy_logs
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
           - Check if user has disabled all timeseries processors. If not,
           - Build a set of processors to run

        3. Run selected processors
           - For each processor to rerun:
             * Execute it with :class:`ProcessorRunner`.
             * Collect secondary results, timeseries domains,
               domain pairs, and logs.
             * Normalize outputs to avoid type inconsistencies.

        4. Process other demands
           - Check that user has not disable other demand timeseries
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
              * ``logs`` : list[str]  
                Aggregated log messages from the entire run.
        """
        # --- 1. Initialization ---
        # If full rerun, remove import_timeseries.inc
        if self.cache_manager.full_rerun:
            p = Path(self.output_folder) / "import_timeseries.inc"
            p.unlink(missing_ok=True)

        # --- 2. Determine processors to run ---
        log_status(
            "Checking the status of timeseries processors",
            self.logs,
            level="run",
            add_empty_line_before=True
        )

        # Check if user has disabled all timeseries processors
        disable_all_ts_processors = self.config.get('disable_all_ts_processors', False)
        if disable_all_ts_processors:
            log_status(
                "User has disabled all timeseries processors in the config file",
                self.logs,
                level="info",
                add_empty_line_before=True
            )

        # Build set of processors to run
        processors_to_rerun = set()
        if not disable_all_ts_processors:
            # Load processor specs
            self.processors, specs_logs = self._load_all_processor_specs()
            self.logs.extend(specs_logs)

            # Get processors marked for rerun by cache manager
            # (includes config changes, code changes, and full rerun flag)
            for proc in self.processors:
                human_name = proc["human_name"]
                is_disabled = proc.get('disabled', False)
                needs_rerun = (
                    self.cache_manager.full_rerun 
                    or self.cache_manager.timeseries_changed.get(human_name, False)
                )

                if needs_rerun and not is_disabled:
                    processors_to_rerun.add(human_name)

        # Separate input-data-independent processors for copying from reference folder
        processors_to_copy = set()
        if self.reference_ts_folder and Path(self.reference_ts_folder) != Path(self.output_folder):
            timeseries_specs_raw = self.config.get("timeseries_specs", {})
            for human_name in list(processors_to_rerun):
                spec = timeseries_specs_raw.get(human_name, {})
                if not spec.get('is_input_data_dependent', True):
                    processors_to_copy.add(human_name)
                    processors_to_rerun.discard(human_name)

        # Log what will run and what will be copied
        log_status(
            f"{len(processors_to_rerun)} timeseries processor(s) need to be run: "
            f"{', '.join(sorted(processors_to_rerun)) if processors_to_rerun else 'none'}",
            self.logs,
            level="info"
        )
        if processors_to_copy:
            log_status(
                f"{len(processors_to_copy)} processor(s) will be copied from reference folder: "
                f"{', '.join(sorted(processors_to_copy))}",
                self.logs,
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
                    scenario_year=self.scenario_year
                )
                log_status(f"Running: {processor['name']}", self.logs, level="run", add_empty_line_before=True)

                # Get structured result
                result = runner.run()
    
                # Process outputs - much cleaner!
                self.secondary_results[result.processor_name] = result.secondary_result
    
                for dom, vals in result.ts_domains.items():
                    all_ts_domains.setdefault(dom, set()).update(vals)
    
                for pair_key, tuples in result.ts_domain_pairs.items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)
    
                self.logs.extend(result.log_messages)

        # --- 3b. Copy input-data-independent processors from reference folder ---
        if processors_to_copy:
            copy_iter = (p for p in self.processors if p['human_name'] in processors_to_copy)

            for processor in copy_iter:
                log_status(f"Copying: {processor['name']}", self.logs, level="run", add_empty_line_before=True)

                copy_result = self._copy_processor_from_reference(processor)

                self.secondary_results[processor["name"]] = copy_result["secondary_result"]

                for dom, vals in copy_result["ts_domains"].items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in copy_result["ts_domain_pairs"].items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

                self.logs.extend(copy_result["log_messages"])


        # --- 4. Process Other Demands ---
        log_status(f"Remaining timeseries actions", self.logs, section_start_length=45, add_empty_line_before=True)
            
        disable_other_demand_ts = self.config.get('disable_other_demand_ts', False)
        if disable_other_demand_ts:
            log_status("User has disabled all 'other demand' timeseries in the config file.", self.logs, level="info", add_empty_line_before=True)

        if not disable_other_demand_ts:
            unprocessed_grids = self._get_unprocessed_demand_grids()

            if unprocessed_grids:
                log_status("Processing other demands", self.logs, level="run")
                for grid in sorted(unprocessed_grids):
                    log_status(f" .. {grid}", self.logs, level="None")

                # Create timeseries for other demands
                df_other_demands = self._create_other_demands(self.df_annual_demands, unprocessed_grids)

                # Collect domain info
                other_domains = collect_domains(df_other_demands, ['grid', 'node'])
                other_domain_pairs = collect_domain_pairs(df_other_demands, [['grid', 'node']])

                for dom, vals in other_domains.items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in other_domain_pairs.items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

                if self.config.get("write_csv_files", False):
                    df_other_demands.to_csv(self.output_folder / "Other_demands_1h_MWh.csv")

                # Write gdx file, update GAMS import instructions
                output_file_other = self.output_folder / "ts_influx_other_demands.gdx"
                write_BB_gdx(df_other_demands, str(output_file_other), self.logs,
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
            log_status("Loading all secondary results from cache.", self.logs, level="run")
            all_secondary_results = self.cache_manager.load_all_secondary_results()
            # Merge with results from current run (current run takes precedence)
            all_secondary_results.update(self.secondary_results)
            self.secondary_results = all_secondary_results

        # Returning TimeseriesRunResult dataclass
        return TimeseriesRunResult(
            secondary_results=self.secondary_results,
            ts_domains={k: sorted(v) for k, v in all_ts_domains.items()},
            ts_domain_pairs={k: sorted(v) for k, v in all_ts_domain_pairs.items()},
            logs=self.logs
        )

