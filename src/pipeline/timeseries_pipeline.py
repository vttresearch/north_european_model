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

        # Loading processor specs, storing logged messages        
        self.processors, specs_logs = self._load_all_processor_specs()
        self.logs.extend(specs_logs)


    def _load_all_processor_specs(self) -> list:
        specs = []
        specs_logs = []
        timeseries_specs = self.config.get("timeseries_specs")
        exclude_grids = self.config.get("exclude_grids")
        processors_base = Path("src/processors")

        for human_name, spec in timeseries_specs.items():
            processor_name = spec.get("processor_name")
            bb_parameter = spec.get("bb_parameter")
            bb_parameter_dimensions = spec.get("bb_parameter_dimensions")

            if not processor_name or not bb_parameter or not bb_parameter_dimensions:
                log_status(f"   Warning! {human_name} spec is incomplete. Skipping.", specs_logs, level="warn")
                continue

            demand_grid = spec.get("demand_grid")
            if demand_grid in exclude_grids:
                log_status(f"   Skipping {processor_name} due to excluded demand grid: {demand_grid}", specs_logs, level="warn")
                continue

            processor_file = processors_base / f"{processor_name}.py"

            enriched_spec = {
                "name": processor_name,
                "file": str(processor_file),
                "spec": spec,
                "human_name": human_name
            }
            specs.append(enriched_spec)

        return specs, specs_logs


    def _determine_processors_to_rerun(self, ts_full_rerun: bool) -> set[str]:
        rerun = set()
        processor_hashes = self.cache_manager.load_processor_hashes()

        for processor in self.processors:
            human_name = processor["human_name"]
            name = processor["name"]
            path = Path(processor["file"])
            current_hash = compute_processor_code_hash(path)

            if ts_full_rerun or processor_hashes.get(name) != current_hash:
                rerun.add(human_name)

        return rerun


    def _create_other_demands(self, df_annual_demands, other_demands):
        """
        For each (grid, node) combination in df_annual_demands
        where the grid (case-insensitive) is in the set 'other_demands',
        create 8760 rows with columns [grid, node, f, t, value].      
        - 'f' is set to 'f00'.
        - 't' is a sequential time label from t000001 up to t008760.
        - 'value' is calculated as TWh/year * 1e6 / 8760.
        """
        # Filter for rows with unprocessed grid values (using lower-case for comparison)
        df_filtered = df_annual_demands[df_annual_demands["grid"].str.lower().isin(other_demands)]
        
        # Create t-index for a full year (8760 hours)
        t_index = [f"t{str(i).zfill(6)}" for i in range(1, 8760+1)]

        # Initialize rows, and empty list for result
        rows = []

        for _, row in df_filtered.iterrows():
            # Calculate hourly value (assume twh/year is numeric). 
            # Negative value for demands. Round to two decimals.
            hourly_value = round(row["twh/year"] * 1e6 / 8760 * -1, 2)
            row = pd.DataFrame({
                "grid": row["grid"],
                "node": row["node"],
                "f": "f00",
                "t": t_index,
                "value": hourly_value
            })
            rows.append(row)
        if rows:
            df_result = pd.concat(rows, ignore_index=True)
        else:
            df_result = pd.DataFrame(columns=["grid", "node", "f", "t", "value"])
        return df_result    



    def run(self) -> dict:

        # If full rerun, remove import_timeseries.inc
        if self.cache_manager.full_rerun:
            p = Path(self.output_folder) / "import_timeseries.inc"
            p.unlink(missing_ok=True)


        log_status("Checking the status of timeseries processors", self.logs, level="run", add_empty_line_before=True)

        # Get processors marked as changed by config
        spec_changes = self.cache_manager.timeseries_changed

        # Get processors marked for rerun based on code change
        code_reruns = self._determine_processors_to_rerun(self.cache_manager.full_rerun)

        # Merge rerun set by human_name
        processors_to_rerun = set()
        for proc in self.processors:
            human_name = proc["human_name"]
            if self.cache_manager.full_rerun or spec_changes.get(human_name, False) or human_name in code_reruns:
                processors_to_rerun.add(human_name)

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

                # secondary_result can be anything or None; no normalization needed

                # --- process outputs ---
                self.secondary_results[name] = secondary_result

                for dom, vals in ts_domains.items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in ts_domain_pairs.items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

                self.logs.extend(processor_log)

            # --- Process Other Demands Not Yet Processed -------------------------------
            log_status(f"Remaining timeseries actions", self.logs, section_start_length=45, add_empty_line_before=True)
            
            all_demand_grids = set()
            if not self.df_annual_demands.empty and "grid" in self.df_annual_demands:
                # pick unique demand grids while dropping NaN, converting to string, and converting to lower case.
                all_demand_grids = set(self.df_annual_demands["grid"].dropna().astype(str).str.lower().unique())

            processed_demand_grids = {
                spec.get("demand_grid").lower()
                for spec in self.config.get("timeseries_specs", {}).values()
                if spec.get("demand_grid")
            }
            unprocessed_grids = all_demand_grids - processed_demand_grids

            if unprocessed_grids:
                log_status("Processing other demands", self.logs, level="run")
                for g in unprocessed_grids:
                    log_status(f" .. {g}", self.logs, level="None")

                # Create timeseries for other demands
                df_other_demands = self._create_other_demands(self.df_annual_demands, unprocessed_grids)

                # Collect new domain info
                domains = ['grid', 'node', 'flow', 'group']
                domain_pairs = [['grid', 'node'], ['flow', 'node']]

                other_domains = collect_domains(df_other_demands, domains)
                other_domain_pairs = collect_domain_pairs(df_other_demands, domain_pairs)

                for dom, vals in other_domains.items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in other_domain_pairs.items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

                if self.config.get("write_csv_files", False):
                    df_other_demands.to_csv(self.output_folder / "Other_demands_1h_MWh.csv")

                output_file_other = self.output_folder / "ts_influx_other_demands.gdx"
                to_gdx({"ts_influx": df_other_demands}, path=output_file_other)

                update_import_timeseries_inc(self.output_folder, bb_parameter="ts_influx", gdx_name_suffix="other_demands")

        # merge all_ts_domains to the ones in cache, load the merged dictionary
        self.cache_manager.merge_dict_to_cache(all_ts_domains, "all_ts_domains.json")
        all_ts_domains = self.cache_manager.load_dict_from_cache("all_ts_domains.json")
        # all_ts_domain_pairs
        self.cache_manager.merge_dict_to_cache(all_ts_domain_pairs, "all_ts_domain_pairs.json")        
        all_ts_domain_pairs = self.cache_manager.load_dict_from_cache("all_ts_domain_pairs.json")

        # Populating self.secondary_results if rebuilding bb excel
        if self.cache_manager.rebuild_bb_excel:
            log_status("Loading secondary results from cache.", self.logs, level="run")
            self.secondary_results = self.cache_manager.load_all_secondary_results()
        else:
            self.secondary_results = {}

        # populating all_ts_domains and all_ts_domain_pairs if rebuilding bb excel

        # Returning TimeseriesRunResult dataclass
        return TimeseriesRunResult(
            secondary_results=self.secondary_results,
            ts_domains={k: sorted(v) for k, v in all_ts_domains.items()},
            ts_domain_pairs={k: sorted(v) for k, v in all_ts_domain_pairs.items()},
            logs=self.logs
        )

