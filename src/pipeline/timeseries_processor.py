# src/pipeline/timeseries_processor.py

import os
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import pandas as pd
from src.utils import log_status
from src.hash_utils import compute_file_hash
from src.utils import trim_df, collect_domains, collect_domain_pairs
import src.GDX_exchange as GDX_exchange 
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from datetime import datetime
from typing import Optional, Any


@dataclass
class ProcessorResult:
    """Results from a single processor execution."""
    main_result: pd.DataFrame
    secondary_result: Optional[Any] = None
    log_messages: list[str] = field(default_factory=list)
    
    def add_log(self, message: str, level: str = "info"):
        """Add a log message."""
        self.log_messages.append(f"[{level.upper()}] {message}")

@dataclass
class ProcessorRunnerResult:
    """
    Results from ProcessorRunner execution.
    
    This dataclass encapsulates all outputs from running a single
    timeseries processor, including the processed data's domain
    information and any secondary outputs.
    
    Attributes
    ----------
    processor_name : str
        Name of the processor that was executed
    secondary_result : Any | None
        Optional secondary output (e.g., metadata, statistics)
        that will be cached for use in other pipeline stages
    ts_domains : dict[str, set]
        Mapping of domain names to sets of values found in the
        processed data (e.g., {'grid': {'FI', 'SE'}, 'node': {...}})
    ts_domain_pairs : dict[str, set[tuple]]
        Mapping of domain pair keys to sets of tuples representing
        relationships (e.g., {('grid', 'node'): {('FI', 'N1'), ...}})
    log_messages : list[str]
        Accumulated log messages from processor execution
    """
    processor_name: str
    secondary_result: Optional[Any]
    ts_domains: dict[str, set]
    ts_domain_pairs: dict[str, set[tuple]]
    log_messages: list[str]


@dataclass
class ProcessorRunner:
    processor_spec: dict
    config: dict
    input_folder: Path
    output_folder: Path
    cache_manager: CacheManager
    source_excel_data_pipeline: SourceExcelDataPipeline

    def _update_processor_hash(self, processor_file: Path, processor_name: str):
        """
        Update the cached hash for this processor.

        This is called after processor execution (successful or skipped) to mark
        the processor code as "seen" at this version. This prevents unnecessary
        reruns when the processor hasn't changed.

        Note: This is separate from CacheManager._validate_processor_code_changes()
        which only READS hashes to determine what needs to run. The update happens
        here to ensure we only mark processors as "up-to-date" after they've 
        actually executed successfully.
        """
        hash_value = compute_file_hash(processor_file)
        self.cache_manager.save_processor_hash(processor_name, hash_value)

    def _write_csv_output(
        self,
        df: pd.DataFrame,
        bb_parameter: str,
        gdx_name_suffix: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        single_year: bool,
        log_messages: list[str]
    ):
        """Write CSV output file."""
        if single_year:
            csv_file = f"{bb_parameter}_{gdx_name_suffix}.csv"
        else:
            csv_file = f"{bb_parameter}_{gdx_name_suffix}_{start_date.year}-{end_date.year}.csv"
        
        csv_path = os.path.join(self.output_folder, csv_file)
        df.to_csv(csv_path)
        log_status(f"Summary CSV written to '{csv_path}'", log_messages, level="info")

    def _write_annual_gdx(self, df, bb_kwargs, log_messages):
        """Write annual GDX files with fallback to gdxpds."""
        log_status("Preparing annual GDX files...", log_messages)
        try:
            GDX_exchange.write_BB_gdx_annual(
                df, self.output_folder, log_messages, 
                use_gams_transfer=True, **bb_kwargs
            )
        except Exception as e:
            log_status(f"Falling back to gdxpds: {e}", log_messages, level="warn")
            GDX_exchange.write_BB_gdx_annual(
                df, self.output_folder, log_messages, 
                use_gams_transfer=False, **bb_kwargs
            )
        GDX_exchange.update_import_timeseries_inc(self.output_folder, **bb_kwargs)

    def _write_average_year_output(
        self,
        df: pd.DataFrame,
        bb_parameter: str,
        gdx_name_suffix: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        single_year: bool,
        rounding_precision: int,
        write_csv: bool,
        bb_kwargs: dict,
        log_messages: list[str]
    ):
        """Write average year CSV and GDX outputs."""
        avg_df = GDX_exchange.calculate_average_year_df(
            df, round_precision=rounding_precision, **bb_kwargs
        )

        if write_csv:
            if single_year:
                avg_csv = f"{bb_parameter}_{gdx_name_suffix}_average_year.csv"
            else:
                avg_csv = f"{bb_parameter}_{gdx_name_suffix}_average_year_from_{start_date.year}-{end_date.year}.csv"

            avg_csv_path = os.path.join(self.output_folder, avg_csv)
            avg_df.to_csv(avg_csv_path)
            log_status(f"Average year CSV written to '{avg_csv_path}'", log_messages, level="info")

        # Write average year GDX file
        log_status("Writing average year GDX file...", log_messages)
        forecast_gdx_path = os.path.join(
            self.output_folder,
            f"{bb_parameter}_{gdx_name_suffix}_forecasts.gdx"
        )

        try:
            # writing with gams.transfer
            GDX_exchange.write_BB_gdx(
                avg_df, 
                forecast_gdx_path, 
                log_messages, 
                use_gams_transfer=True,
                **bb_kwargs
            )
        except Exception as e:
            log_status(
                f"GDX writing with GAMS Transfer failed ({e}), falling back to gdxpds.",
                log_messages,
                level="warn"
            )
            # Fallback to gdxpds
            GDX_exchange.write_BB_gdx(
                avg_df, 
                forecast_gdx_path, 
                log_messages,  # ← FIXED: This was missing in the original!
                use_gams_transfer=False,
                **bb_kwargs
            )

        GDX_exchange.update_import_timeseries_inc(
            self.output_folder, file_suffix="forecasts", **bb_kwargs
        )
        log_status(
            f"Average year GDX for Backbone written to '{self.output_folder}'",
            log_messages,
            level="info"
        )


    def run(self) -> ProcessorRunnerResult:
        """
        Run a single processor and return structured results.
        """
        spec = self.processor_spec["spec"]
        processor_name = self.processor_spec["name"]
        human_name = self.processor_spec["human_name"]
        processor_file = Path(self.processor_spec["file"])
        
        log_messages = []
        log_status(f"{human_name}", log_messages, section_start_length=45)

        # Extract config values
        start_date = pd.to_datetime(self.config.get("start_date"))
        end_date = pd.to_datetime(self.config.get("end_date"))
        country_codes = self.config.get("country_codes", [])
        rounding_precision = spec.get("rounding_precision", 0)
        bb_parameter = spec.get("bb_parameter")
        gdx_name_suffix = spec.get("gdx_name_suffix", "")
        process_only_single_year = spec.get("process_only_single_year", False)
        write_csv_files = spec.get("write_csv_files", False)

        # Load processor module
        module_spec = importlib.util.spec_from_file_location(processor_name, processor_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        if not hasattr(module, processor_name):
            raise AttributeError(f"Processor module {processor_name} is missing class {processor_name}.")

        ProcessorClass = getattr(module, processor_name)

        # Prepare processor kwargs
        processor_kwargs = {
            "input_folder": os.path.join(self.input_folder, "timeseries"),
            "input_file": spec.get("input_file", ""),
            "country_codes": country_codes,
            "start_date": self.config.get("start_date"),
            "end_date": self.config.get("end_date"),
            "scenario_year": self.config.get("scenario_years", [None])[0],
            "exclude_nodes": self.config.get("exclude_nodes", []),
            "attached_grid": spec.get("attached_grid"),
            **spec
        }

        # Handle demand data
        demand_grid = spec.get("demand_grid")
        if demand_grid:
            df_annual_demands = self.source_excel_data_pipeline.df_demanddata

            # Filter to the specific grid
            df_filtered = df_annual_demands[
                df_annual_demands["grid"].str.lower() == demand_grid.lower()
            ]

            if df_filtered.empty:
                log_status(
                    f"No demand data found for grid '{demand_grid}'. Skipping processor '{human_name}'.",
                    log_messages,
                    level="warn",
                )
                self._update_processor_hash(processor_file, processor_name)
                return ProcessorRunnerResult(
                    processor_name=processor_name,
                    secondary_result=None,
                    ts_domains={},
                    ts_domain_pairs={},
                    log_messages=log_messages
                )

            processor_kwargs["df_annual_demands"] = df_filtered

        # Prepare BB Conversion kwargs
        bb_conversion_kwargs = {
            "processor_name": processor_name,
            "bb_parameter": bb_parameter,
            "bb_parameter_dimensions": spec.get("bb_parameter_dimensions"),
            "custom_column_value": spec.get("custom_column_value"),
            "quantile_map": spec.get("quantile_map", {0.5: "f01", 0.1: "f02", 0.9: "f03"}),
            "gdx_name_suffix": gdx_name_suffix,
            "is_demand": bool(demand_grid)
        }

        # Instantiate and run processor
        processor_instance = ProcessorClass(**processor_kwargs)
        processor_result = processor_instance.run_processor()

        # Extract results from ProcessorResult dataclass
        main_result = processor_result.main_result
        secondary_result = processor_result.secondary_result
        log_messages.extend(processor_result.log_messages)

        # Trim + convert to BB
        trimmed_result = trim_df(main_result, rounding_precision)
        main_result_bb = GDX_exchange.prepare_BB_df(
            trimmed_result, start_date, country_codes, **bb_conversion_kwargs
        )

        # Write CSV if requested
        if write_csv_files:
            self._write_csv_output(
                trimmed_result, bb_parameter, gdx_name_suffix,
                start_date, end_date, process_only_single_year, log_messages
            )

        # Write annual GDX files
        self._write_annual_gdx(main_result_bb, bb_conversion_kwargs, log_messages)

        # Average year processing
        if spec.get("calculate_average_year", False):
            self._write_average_year_output(
                main_result_bb, bb_parameter, gdx_name_suffix,
                start_date, end_date, process_only_single_year,
                rounding_precision, write_csv_files,
                bb_conversion_kwargs, log_messages
            )

        # Save secondary result to cache
        if secondary_result is not None:
            secondary_output_name = spec.get("secondary_output_name")
            self.cache_manager.save_secondary_result(
                processor_name, secondary_result, secondary_output_name
            )

        # Collect domains and domain pairs
        domains = ['grid', 'node', 'flow', 'group']
        domain_pairs = [['grid', 'node'], ['flow', 'node']]
        local_ts_domains = collect_domains(main_result_bb, domains)
        local_ts_domain_pairs = collect_domain_pairs(main_result_bb, domain_pairs)

        # Save processor hash
        self._update_processor_hash(processor_file, processor_name)

        # Return structured result
        return ProcessorRunnerResult(
            processor_name=processor_name,
            secondary_result=secondary_result,
            ts_domains=local_ts_domains,
            ts_domain_pairs=local_ts_domain_pairs,
            log_messages=log_messages
        )



