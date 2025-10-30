from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from src.pipeline.timeseries_processor import ProcessorRunner
from src.utils import log_status
from src.hash_utils import compute_processor_code_hash
from src.utils import collect_domains, collect_domain_pairs
from gdxpds import to_gdx
from src.GDX_exchange import update_import_timeseries_inc

@dataclass
class TimeseriesRunResult:
    secondary_results: dict
    ts_domains: dict[str, list]
    ts_domain_pairs: set[tuple]
    logs: str

class TimeseriesPipeline:
    """
    Orchestrates the execution of timeseries processors based on configuration.
    """
    
    def __init__(self, config: dict, input_folder: Path, output_folder: Path,
                 cache_manager: CacheManager, source_excel_data_pipeline: SourceExcelDataPipeline):
        self.config = config
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cache_manager = cache_manager
        self.source_excel_data_pipeline = source_excel_data_pipeline
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
                log_status(f"Warning! {human_name} spec is incomplete. Skipping.", 
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



    def _determine_processors_to_rerun(self, ts_full_rerun: bool) -> set[str]:
        """
        Determine which timeseries processors need to be rerun.
          - Processors are skipped if ``disabled`` is True.
          - If ``ts_full_rerun`` is True, all enabled processors are marked for rerun.
          - Otherwise, enabled processors are rerun only if their current code hash
            differs from the stored hash in the cache.

        Parameters
        ----------
        ts_full_rerun : bool
            If True, force rerun of all enabled processors regardless of code changes.

        Returns
        -------
        set[str]
            Set of processor ``human_name`` identifiers that should be rerun.
        """
        rerun: set[str] = set()
        processor_hashes: dict[str, str] = self.cache_manager.load_processor_hashes()

        for processor in self.processors:
            if processor.get("disabled", False):
                continue  # Skip disabled processors

            human_name: str = processor["human_name"]
            name: str = processor["name"]
            path = Path(processor["file"])
            current_hash: str = compute_processor_code_hash(path)

            if ts_full_rerun or processor_hashes.get(name) != current_hash:
                rerun.add(human_name)

        return rerun



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
 



    def run(self) -> TimeseriesRunResult:
        """
        Execute the full timeseries processing pipeline.

        Workflow
        --------
        1. **Initialization**
           - If a full rerun is requested, remove any existing
             ``import_timeseries.inc`` file.
           - Log the start of timeseries processing.

        2. **Determine processors to rerun**
           - Check if user has disabled all timeseries processors. If not,
           - Compare processor specifications and cached code hashes.
           - Collect all processors that need rerun due to:
             * global full rerun,
             * changed configuration, or
             * changed processor code.
           - Check that single timeseries processor is not disabled

        3. **Run selected processors**
           - For each processor to rerun:
             * Execute it with :class:`ProcessorRunner`.
             * Collect secondary results, timeseries domains,
               domain pairs, and logs.
             * Normalize outputs to avoid type inconsistencies.

        4. **Process unhandled demands**
           - Check that user has not disable other demand timeseries
           - Identify demand grids present in ``df_annual_demands`` but not
             covered by explicit processors.
           - Generate hourly timeseries for these with
             :meth:`_create_other_demands`.
           - Write optional CSV/GDX outputs and update
             ``import_timeseries.inc``.

        5. **Cache management**
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
        # --- 1. **Initialization** ---
        # If full rerun, remove import_timeseries.inc
        if self.cache_manager.full_rerun:
            p = Path(self.output_folder) / "import_timeseries.inc"
            p.unlink(missing_ok=True)

        # --- 2. **Determine processors to rerun** ---
        log_status("Checking the status of timeseries processors", self.logs, level="run", add_empty_line_before=True)

        # Check is user has disabled all timeseries processors
        disable_all_ts_processors: bool = self.config.get('disable_all_ts_processors', False)
        if disable_all_ts_processors:
            log_status("User has disabled all timeseries processors in the config file", self.logs, level="info", add_empty_line_before=True)

        processors_to_rerun = set()
        if not disable_all_ts_processors:
            # Loading processor specs, storing logged messages        
            self.processors, specs_logs = self._load_all_processor_specs()
            self.logs.extend(specs_logs)

            # Get processors marked as changed by config
            spec_changes = self.cache_manager.timeseries_changed

            # Get processors marked for rerun based on code change
            code_reruns = self._determine_processors_to_rerun(self.cache_manager.full_rerun)

            # Merge rerun set by human_name
            for proc in self.processors:
                human_name = proc["human_name"]
                if self.cache_manager.full_rerun or spec_changes.get(human_name, False) or human_name in code_reruns:
                    if not proc.get('disabled', False):
                        processors_to_rerun.add(human_name)
            
        # --- 3. **Run selected processors** ---
        # Print processors that will be rerun
        log_status(
            f"{len(processors_to_rerun)} timeseries processors needs to be run: {', '.join(processors_to_rerun)}",
            self.logs,
            level="info"
        )

        # Initialize dictionaries for domains and domain pairs from timeseries processing
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
                    cache_manager=self.cache_manager
                )
                log_status(f"Running: {processor['name']}", self.logs, level="run", add_empty_line_before=True)
                name, secondary_result, ts_domains, ts_domain_pairs, processor_log = runner.run()

                # --- Normalize outputs from runner so merges don't explode ---
                # ts_domains: expect dict[str, Iterable]
                if not ts_domains:
                    ts_domains = {}
                elif not isinstance(ts_domains, dict):
                    ts_domains = {}

                # ts_domain_pairs: expect dict[str, Iterable[tuple]]
                if not ts_domain_pairs:
                    ts_domain_pairs = {}
                elif not isinstance(ts_domain_pairs, dict):
                    # Legacy or accidental set/other types → discard for safety
                    ts_domain_pairs = {}

                # processor_log: expect list[str]
                if not processor_log:
                    processor_log = []
                elif isinstance(processor_log, str):
                    processor_log = [processor_log]

                # --- process outputs ---
                self.secondary_results[name] = secondary_result

                for dom, vals in ts_domains.items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in ts_domain_pairs.items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

                self.logs.extend(processor_log)


        # --- 4. **Process unhandled demands** ---
        log_status(f"Remaining timeseries actions", self.logs, section_start_length=45, add_empty_line_before=True)
            
        # --- Process Other Demands Not Yet Processed 
        all_demand_grids = set()
        disable_other_demand_ts = self.config.get('disable_other_demand_ts', False)
        if disable_other_demand_ts:
            log_status("User has disabled all 'other demand' timeseries in the config file.", self.logs, level="info", add_empty_line_before=True)

        if not disable_other_demand_ts:
            if self.df_annual_demands is not None and not self.df_annual_demands.empty and "grid" in self.df_annual_demands:
                # pick unique demand grids while dropping NaN, converting to string, and converting to lower case.
                all_demand_grids = set(self.df_annual_demands["grid"].dropna().astype(str).str.lower().unique())

            specs_cfg: dict = self.config.get("timeseries_specs", {})

            processed_from_enabled = {
                (spec.get("demand_grid") or "").lower()
                for spec in specs_cfg.values()
                if spec.get("demand_grid") and not spec.get("disabled", False) and not disable_all_ts_processors
            }

            reserved_from_disabled = {
                (spec.get("demand_grid") or "").lower()
                for spec in specs_cfg.values()
                if spec.get("demand_grid")
                and (spec.get("disabled", False) or disable_all_ts_processors)
                and spec.get("reserve_grid_when_disabled", True)
            }

            processed_demand_grids = (processed_from_enabled | reserved_from_disabled) - {""}
            unprocessed_grids = all_demand_grids - processed_demand_grids

            if unprocessed_grids:
                log_status("Processing other demands", self.logs, level="run")
                for g in unprocessed_grids:
                    log_status(f" .. {g}", self.logs, level="None")

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
                to_gdx({"ts_influx": df_other_demands}, path=output_file_other)
                update_import_timeseries_inc(self.output_folder, bb_parameter="ts_influx", gdx_name_suffix="other_demands")


        # --- 5. **Cache management** ---

        # merge all_ts_domains to the ones in cache, load the merged dictionary
        self.cache_manager.merge_dict_to_cache(all_ts_domains, "all_ts_domains.json")
        all_ts_domains = self.cache_manager.load_dict_from_cache("all_ts_domains.json")
        # merge all_ts_domain_pairs to the ones in cache, load the merged dictionary
        self.cache_manager.merge_dict_to_cache(all_ts_domain_pairs, "all_ts_domain_pairs.json")        
        all_ts_domain_pairs = self.cache_manager.load_dict_from_cache("all_ts_domain_pairs.json")

        # Populating self.secondary_results if rebuilding bb excel
        if self.cache_manager.rebuild_bb_excel:
            log_status("Loading secondary results from cache.", self.logs, level="run")
            self.secondary_results = self.cache_manager.load_all_secondary_results()
        else:
            self.secondary_results = {}

        # Returning TimeseriesRunResult dataclass
        return TimeseriesRunResult(
            secondary_results=self.secondary_results,
            ts_domains={k: sorted(v) for k, v in all_ts_domains.items()},
            ts_domain_pairs={k: sorted(v) for k, v in all_ts_domain_pairs.items()},
            logs=self.logs
        )

