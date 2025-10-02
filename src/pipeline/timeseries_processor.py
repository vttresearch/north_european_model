# src/pipeline/processor.py

import os
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import pandas as pd
from src.utils import log_status
from src.hash_utils import compute_processor_code_hash
from src.utils import trim_df, collect_domains, collect_domain_pairs
from src.GDX_exchange import (
    prepare_BB_df,
    write_BB_gdx,
    write_BB_gdx_gt,
    calculate_average_year_df,
    write_BB_gdx_annual,
    write_BB_gdx_annual_gt,      
    update_import_timeseries_inc
)
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from datetime import datetime


@dataclass
class ProcessorRunner:
    processor_spec: dict
    config: dict
    input_folder: Path
    output_folder: Path
    cache_manager: CacheManager
    source_excel_data_pipeline: SourceExcelDataPipeline


    def _parse_processor_result(self, result) -> tuple[pd.DataFrame, object, str]:
        """
        Parse the result returned from a processor's run method.

        Returns:
            tuple: (main_result, secondary_result, processor_log)
        """
        main_result = None
        secondary_result = None
        processor_log = ""   # Note: processor log is a string, because then we can distinct it from secondary results

        if isinstance(result, pd.DataFrame):
            main_result = result

        elif isinstance(result, tuple):
            if len(result) == 2:
                main_result, second = result

                if isinstance(second, str):
                    processor_log = second
                    secondary_result = None
                else:
                    secondary_result = second
                    processor_log = ""

            elif len(result) == 3:
                main_result, secondary_result, processor_log = result

                if not isinstance(processor_log, str):
                    print("⚠️ Warning: Expected a string for processor_log, got", type(processor_log))
                    processor_log = str(processor_log)

            else:
                raise ValueError(f"Unexpected number of values returned from processor: {len(result)}")

        else:
            raise TypeError(f"Processor returned unsupported result type: {type(result)}")

        if not isinstance(main_result, pd.DataFrame):
            raise TypeError("Processor must return a pandas DataFrame as the first result.")

        return main_result, secondary_result, processor_log


    def run(self) -> tuple[str, object, dict, set]:
        """
        Run a single processor and return its secondary result.

        Returns:
            tuple: (processor_name, secondary_result, local_ts_domains, local_ts_domain_pairs, log_messages)
        """
        spec = self.processor_spec["spec"]
        processor_name = self.processor_spec["name"]
        human_name = self.processor_spec["human_name"]
        processor_file = Path(self.processor_spec["file"])
        
        log_messages = []
        log_status(f"{human_name}", log_messages, section_start_length=45)

        # Helper function updating processor hash
        def update_processor_hash():
            hash_value = compute_processor_code_hash(processor_file)
            self.cache_manager.save_processor_hash(processor_name, hash_value)

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
        # Demand data
        demand_grid = spec.get("demand_grid")
        if demand_grid:
            df_annual_demands = self.source_excel_data_pipeline.df_demanddata
            if df_annual_demands.empty:
                log_status(
                    f"All annual demands empty. Skipping processor '{human_name}'.",
                    log_messages,
                    level="warn",
                )
                # Keep hash bookkeeping consistent even when skipping
                update_processor_hash()
                # Return: (processor_name, secondary_result, local_ts_domains, local_ts_domain_pairs, log_messages)
                return processor_name, None, {}, {}, log_messages
            else: 
                df_filtered = df_annual_demands[df_annual_demands["grid"].str.lower() == demand_grid.lower()]

            if df_filtered.empty:
                log_status(
                    f"No annual demand rows found for grid '{demand_grid}'. Skipping processor '{human_name}'.",
                    log_messages,
                    level="warn",
                )
                # Keep hash bookkeeping consistent even when skipping
                update_processor_hash()
                # Return: (processor_name, secondary_result, local_ts_domains, local_ts_domain_pairs, log_messages)
                return processor_name, None, {}, {}, log_messages

            # Only pass demands onward if non-empty
            processor_kwargs["df_annual_demands"] = df_filtered
        ## Demand data
        #demand_grid = spec.get("demand_grid")
        #if demand_grid:
        #    df_annual_demands = self.source_excel_data_pipeline.df_demanddata
        #    df_filtered = df_annual_demands[df_annual_demands["grid"].str.lower() == demand_grid.lower()]
        #    processor_kwargs["df_annual_demands"] = df_filtered

        # Prepare BB Conversion kwargs
        bb_conversion_kwargs = {
            "processor_name": processor_name,
            "bb_parameter": bb_parameter,
            "bb_parameter_dimensions": spec.get("bb_parameter_dimensions"),
            "custom_column_value": spec.get("custom_column_value"),
            "quantile_map": spec.get("quantile_map", {0.5: "f01", 0.1: "f02", 0.9: "f03"}),
            "gdx_name_suffix": gdx_name_suffix
        }
        if demand_grid:
            bb_conversion_kwargs["is_demand"] = True

        # Instantiate and run
        processor_instance = ProcessorClass(**processor_kwargs)
        result = processor_instance.run_processor()
        main_result, secondary_result, processor_log = self._parse_processor_result(result)

        # Trim + convert to BB
        trimmed_result = trim_df(main_result, rounding_precision)
        main_result_bb = prepare_BB_df(trimmed_result, start_date, country_codes, **bb_conversion_kwargs)

        # Add processor log to logs
        # Note: appending instead of extending, because processor log is a string
        if processor_log != "":
            log_messages.append(processor_log)

        # Write CSV
        if write_csv_files:
            if process_only_single_year:
                csv_file = f"{bb_parameter}_{gdx_name_suffix}.csv"
            else:
                csv_file = f"{bb_parameter}_{gdx_name_suffix}_{start_date.year}-{end_date.year}.csv"
            csv_path = os.path.join(self.output_folder, csv_file)
            trimmed_result.to_csv(csv_path)
            log_status(f"Summary CSV written to '{csv_path}'", log_messages, level="info")

        # Write annual GDX files
        log_status(f"Writing annual GDX files...", log_messages)
        try:
            write_BB_gdx_annual_gt(main_result_bb, self.output_folder, log_messages, **bb_conversion_kwargs)
        except:
            log_status("GDX writing with GAMS Transfer failed, falling back to gdxpds.", log_messages, level="warn")
            write_BB_gdx_annual(main_result_bb, self.output_folder, log_messages, **bb_conversion_kwargs)   

        log_status(f"Annual GDX files for Backbone written to '{self.output_folder}'", log_messages, level="info")
        # Update import_timeseries.inc
        update_import_timeseries_inc(self.output_folder, **bb_conversion_kwargs)

        # Average year processing
        if spec.get("calculate_average_year", False):
            avg_df = calculate_average_year_df(main_result_bb, round_precision=rounding_precision, **bb_conversion_kwargs)
            if write_csv_files:
                if process_only_single_year:
                    avg_csv = f"{bb_parameter}_{gdx_name_suffix}_average_year.csv"
                else:
                    avg_csv = f"{bb_parameter}_{gdx_name_suffix}_average_year_from_{start_date.year}-{end_date.year}.csv"
                avg_csv_path = os.path.join(self.output_folder, avg_csv)
                avg_df.to_csv(avg_csv_path)
                log_status(f"Average year CSV written to '{avg_csv_path}'", log_messages, level="info")

            # Write average year GDX file
            log_status(f"Writing average year GDX file...", log_messages)
            forecast_gdx_path = os.path.join(self.output_folder, f"{bb_parameter}_{gdx_name_suffix}_forecasts.gdx")
            try:
                write_BB_gdx_gt(avg_df, forecast_gdx_path, log_messages, **bb_conversion_kwargs)
            except:
                log_status("GDX writing with GAMS Transfer failed, falling back to gdxpds.", log_messages, level="warn")
                write_BB_gdx(avg_df, forecast_gdx_path, **bb_conversion_kwargs)

            # Update import_timeseries.inc
            update_import_timeseries_inc(self.output_folder, file_suffix="forecasts", **bb_conversion_kwargs)
            log_status(f"Average year GDX for Backbone written to '{self.output_folder}'", log_messages, level="info")

        # Save secondary result to cache
        if secondary_result is not None:
            secondary_output_name = spec.get("secondary_output_name")
            self.cache_manager.save_secondary_result(processor_name, secondary_result, secondary_output_name)

        # collect domains and domain pairs
        domains = ['grid', 'node', 'flow', 'group']
        domain_pairs = [['grid', 'node'], ['flow', 'node']]
        local_ts_domains = collect_domains(main_result_bb, domains)
        local_ts_domain_pairs = collect_domain_pairs(main_result_bb, domain_pairs)


        # Save processor hash
        update_processor_hash()

        # Return: (processor_name, secondary_result, local_ts_domains, local_ts_domain_pairs, log_messages)
        return processor_name, secondary_result, local_ts_domains, local_ts_domain_pairs, log_messages



