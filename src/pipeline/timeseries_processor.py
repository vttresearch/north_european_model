# src/pipeline/timeseries_processor.py
"""
Timeseries processor runner -- dynamic loading and execution of individual processors.

Purpose
-------
This module provides the glue between the orchestrating ``TimeseriesPipeline``
and the individual processor classes that live in ``src/processors/``.
``ProcessorRunner`` dynamically loads a processor by name, injects a standard
set of kwargs, calls ``run_processor()``, normalizes the returned DataFrame,
converts it to Backbone long format, and writes GDX output files.

Data interface -- processor contract
-------------------------------------
Every processor class must implement ``run_processor()`` and return a
:class:`ProcessorResult` whose ``main_result`` is a **wide-format**
``pd.DataFrame``.  The layout varies by processor (e.g. dates as row index,
node/country codes as columns), but the following dtype contract must hold
or ``ProcessorRunner`` will not be able to normalize the data reliably:

- Numeric data must be representable as ``Float64`` (pandas nullable float).
  Plain Python ``float``, ``int``, and NumPy numeric dtypes are all accepted;
  ``ProcessorRunner`` casts them during normalization.
- String / categorical columns must be ``object`` dtype.
- Missing values should be ``pd.NA``, ``np.nan``, or ``None``; string
  ``'NaN'`` is also handled.  Do **not** use sentinel values such as ``-9999``
  to represent missing data.

If ``main_result`` is not a ``pd.DataFrame``, or is empty, ``ProcessorRunner``
logs a warning and continues with an empty DataFrame -- no GDX is written for
that processor.

Normalization applied by ProcessorRunner (before BB conversion)
---------------------------------------------------------------
After ``run_processor()`` returns, ``ProcessorRunner`` applies the same
normalization pipeline that ``SourceExcelDataPipeline`` applies to Excel
source data, so downstream code can make the same assumptions about both:

1. **Drop empty columns** -- columns where all values are NA or zero
   (numeric) / blank (string) are removed via ``utils.is_col_empty``.
2. **Standardize dtypes** -- ``utils.standardize_df_dtypes`` converts numeric
   columns to ``Float64`` and preserves ``pd.NA`` for missing values.
3. **Fill numeric NA** -- ``utils.fill_numeric_na`` replaces remaining
   ``pd.NA`` in ``Float64`` columns with ``0``.
4. **Round** -- values are rounded to ``rounding_precision`` (default 0)
   from the processor spec.

Processors must not rely on this normalization for correctness; they should
return clean data.  The normalization is a safety net and a compatibility
layer, not a substitute for proper output.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import src.hash_utils as hash_utils
import src.utils as utils
import src.GDX_exchange as GDX_exchange
import src.json_exchange as json_exchange
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from typing import Optional, Any


@dataclass
class ProcessorResult:
    """
    Results returned by a processor's ``run_processor()`` method.

    Attributes
    ----------
    main_result : pd.DataFrame
        Wide-format DataFrame in the processor's own output convention,
        before any BB-format conversion or GDX writing.  Column layout
        varies by processor (e.g. dates as index, countries as columns).
    secondary_result : Any or None
        Optional secondary output produced alongside the main timeseries
        (e.g. annual totals, scaling factors) for use in later pipeline
        stages such as ``BuildInputExcel``.  ``None`` if the processor
        does not produce a secondary output.
    """
    main_result: pd.DataFrame
    secondary_result: Optional[Any] = None


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
        relationships (e.g., {"grid,node": {("elec", "FI00_elec"), ...}})
    """
    processor_name: str
    secondary_result: Optional[Any]
    ts_domains: dict[str, set]
    ts_domain_pairs: dict[str, set[tuple]]


@dataclass
class ProcessorRunner:
    """
    Executes a single timeseries processor and writes its outputs.

    Dynamically loads the processor class from ``src/processors/`` (the module
    file and the class inside it must share the same name), instantiates it with
    a standardised set of kwargs derived from the config and the enriched
    processor spec, and calls ``run_processor()``.

    After the processor returns a :class:`ProcessorResult`, this class:

    - Cleans and converts ``main_result`` to Backbone long format via
      ``GDX_exchange.prepare_BB_df``.
    - Writes one GDX file per calendar year (``write_BB_gdx_annual``) and
      updates ``import_timeseries.inc``.
    - Optionally computes an average-year GDX when ``calculate_average_year``
      is set in the spec.
    - Persists ``secondary_result`` and per-processor domain data to the cache.
    - Records a hash of the processor file so the cache manager can detect
      code changes on the next run.
    """
    processor_spec: dict
    config: dict
    input_folder: Path
    output_folder: Path
    cache_manager: CacheManager
    source_excel_data_pipeline: SourceExcelDataPipeline
    scenario_year: Optional[int] = None
    logger: Optional[Any] = None

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

        The function is thin, but the purpose is hopefully easier to catch 
        witht this docstring.
        """
        hash_value = hash_utils.compute_file_hash(processor_file)
        self.cache_manager.save_processor_hash(processor_name, hash_value)


    def run(self) -> ProcessorRunnerResult:
        """
        Execute the processor and return all structured outputs.

        Workflow
        --------
        1. Initialisation
           - Extract common config values (dates, country codes, rounding).
           - Dynamically load the processor module and class from
             ``src/processors/{processor_name}.py``.

        2. Demand data handling
           - If the spec declares a ``demand_grid``, filter ``df_demanddata``
             to that grid and pass it to the processor as ``df_annual_demands``.
           - If no matching demand rows are found, log a warning and return an
             empty :class:`ProcessorRunnerResult` (hash still updated).

        3. Run and convert
           - Instantiate the processor class and call ``run_processor()``.
           - Drop empty columns, standardise dtypes, and round to
             ``rounding_precision``.
           - Convert to Backbone long format via ``GDX_exchange.prepare_BB_df``
             (adds ``f`` and ``t`` columns, applies ``country_codes`` mapping).
           - Write one GDX per calendar year and update ``import_timeseries.inc``.
           - Optionally write a pre-conversion summary CSV (wide format).

        4. Average-year processing (optional)
           - If ``calculate_average_year`` is set in the spec, compute a
             single-year average DataFrame and write it as a separate
             ``_forecasts.gdx`` file, also registered in
             ``import_timeseries.inc``.

        5. Post-processing
           - Save ``secondary_result`` to the cache if present.
           - Collect domain values and domain pairs from the converted DataFrame
             and save them to a per-processor JSON cache file.
           - Update the processor file hash.

        Returns
        -------
        ProcessorRunnerResult
            Contains ``processor_name``, ``secondary_result``, ``ts_domains``,
            and ``ts_domain_pairs`` for the executed processor.
        """
        # --- Initialization ---
        spec = self.processor_spec["spec"]
        processor_name = self.processor_spec["name"]
        human_name = self.processor_spec["human_name"]
        processor_file = Path(self.processor_spec["file"])

        self.logger.log_status(f"{human_name}", section_start_length=45)

        # Extract config values
        start_date = pd.to_datetime(self.config["start_date"])
        end_date = pd.to_datetime(self.config["end_date"])
        country_codes = self.config["country_codes"]
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
            self.logger.log_status(
                f"Processor module '{processor_name}' is missing a class named '{processor_name}'. "
                f"No GDX output will be written.",
                level="warn",
            )
            self._update_processor_hash(processor_file, processor_name)
            return ProcessorRunnerResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )

        ProcessorClass = getattr(module, processor_name)

        # Prepare processor kwargs
        processor_kwargs = {
            "input_folder": os.path.join(self.input_folder, "timeseries"),
            "input_file": spec.get("input_file", ""),
            "country_codes": country_codes,
            "start_date": start_date,
            "end_date": end_date,
            "scenario_year": self.scenario_year,
            "exclude_nodes": self.config["exclude_nodes"],
            "attached_grid": spec.get("attached_grid"),
            "logger": self.logger,
            **spec
        }


        # --- Handle demand data ---
        demand_grid = spec.get("demand_grid")
        if demand_grid:
            df_annual_demands = self.source_excel_data_pipeline.df_demanddata

            # Filter to the specific grid
            df_filtered = df_annual_demands[
                df_annual_demands["grid"].str.lower() == demand_grid.lower()
            ]

            if df_filtered.empty:
                self.logger.log_status(
                    f"No demand data found for grid '{demand_grid}'. Skipping processor '{human_name}'.",
                    level="warn",
                )
                self._update_processor_hash(processor_file, processor_name)
                return ProcessorRunnerResult(
                    processor_name=processor_name,
                    secondary_result=None,
                    ts_domains={},
                    ts_domain_pairs={},
                )

            processor_kwargs["df_annual_demands"] = df_filtered


        # --- Prepare and run processor ---
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
        try:
            processor_instance = ProcessorClass(**processor_kwargs)
            processor_result = processor_instance.run_processor()
        except Exception as e:
            self.logger.log_status(
                f"Processor '{processor_name}' raised an exception during execution: {e}. "
                f"No GDX output will be written.",
                level="warn",
            )
            self._update_processor_hash(processor_file, processor_name)
            return ProcessorRunnerResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )

        # Extract results from ProcessorResult dataclass
        main_result = processor_result.main_result
        secondary_result = processor_result.secondary_result

        # Guard: processors must return a DataFrame (see module docstring)
        if not isinstance(main_result, pd.DataFrame):
            self.logger.log_status(
                f"Processor '{processor_name}' returned main_result of type "
                f"'{type(main_result).__name__}', expected pd.DataFrame.  "
                f"Check that run_processor() returns a ProcessorResult.  "
                f"No GDX output will be written.",
                level="warn",
            )
            main_result = pd.DataFrame()
        if main_result.empty:
            self.logger.log_status(
                f"Processor '{processor_name}' returned an empty DataFrame.  "
                f"No GDX output will be written.",
                level="warn",
            )

        # --- Convert results to BB format and write them ---
        # drop empty columns, convert dTypes, and round
        main_result = main_result.loc[:, ~main_result.apply(utils.is_col_empty)]
        main_result = utils.fill_numeric_na(utils.standardize_df_dtypes(main_result))
        main_result = main_result.round(rounding_precision)

        main_result_bb = pd.DataFrame()
        try:
            # Convert to BB format
            main_result_bb = GDX_exchange.prepare_BB_df(
                main_result, start_date, country_codes, **bb_conversion_kwargs
            )

            # Write trimmed results (wide format) to CSV if requested
            if write_csv_files:
                if process_only_single_year:
                    csv_file = f"{bb_parameter}_{gdx_name_suffix}.csv"
                else:
                    csv_file = f"{bb_parameter}_{gdx_name_suffix}_{start_date.year}-{end_date.year}.csv"

                csv_path = os.path.join(self.output_folder, csv_file)
                main_result.to_csv(csv_path)
                self.logger.log_status(f"Summary CSV written to '{csv_path}'", level="info")

            # Write results (BB format) to annual GDX files
            self.logger.log_status("Preparing annual GDX files...")
            GDX_exchange.write_BB_gdx_annual(main_result_bb, self.output_folder, self.logger, **bb_conversion_kwargs)
            GDX_exchange.update_import_timeseries_inc(self.output_folder, **bb_conversion_kwargs)

        except Exception as e:
            self.logger.log_status(
                f"Processor '{processor_name}': BB conversion or GDX writing failed: {e}",
                level="warn",
            )

        # --- Average year processing ---
        if spec.get("calculate_average_year", False):
            try:
                self.logger.log_status("Calculating average year GDX file...")
                avg_df = GDX_exchange.calculate_average_year_df(
                    main_result_bb,
                    round_precision=rounding_precision,
                    **bb_conversion_kwargs
                )

                # Write CSV if requested
                if write_csv_files:
                    self.logger.log_status("Writing average year csv file...")
                    if process_only_single_year:
                        avg_csv = f"{bb_parameter}_{gdx_name_suffix}_average_year.csv"
                    else:
                        avg_csv = f"{bb_parameter}_{gdx_name_suffix}_average_year_from_{start_date.year}-{end_date.year}.csv"

                    avg_csv_path = os.path.join(self.output_folder, avg_csv)
                    avg_df.to_csv(avg_csv_path)

                # Write average year GDX file
                self.logger.log_status("Writing average year GDX file...")
                forecast_gdx_path = os.path.join(
                    self.output_folder,
                    f"{bb_parameter}_{gdx_name_suffix}_forecasts.gdx"
                )
                GDX_exchange.write_BB_gdx(avg_df, forecast_gdx_path, self.logger, **bb_conversion_kwargs)

                # Update import file
                GDX_exchange.update_import_timeseries_inc(
                    self.output_folder, file_suffix="forecasts", **bb_conversion_kwargs
                )

            except Exception as e:
                self.logger.log_status(
                    f"Processor '{processor_name}': Average year calculation or GDX writing failed: {e}",
                    level="warn",
                )

        # --- Post-processing activities ---
        # Save secondary result to cache
        if secondary_result is not None:
            secondary_output_name = spec.get("secondary_output_name")
            self.cache_manager.save_secondary_result(
                processor_name, secondary_result, secondary_output_name
            )

        # Collect domains and domain pairs
        domains = ['grid', 'node', 'flow', 'group']
        domain_pairs = [['grid', 'node'], ['flow', 'node']]
        local_ts_domains = utils.collect_domains_for_cache(main_result_bb, domains)
        local_ts_domain_pairs = utils.collect_domain_pairs_for_cache(main_result_bb, domain_pairs)

        # Save per-processor domain data for copy optimization
        domain_cache_data = {
            "ts_domains": {k: list(v) for k, v in local_ts_domains.items()},
            "ts_domain_pairs": {k: [list(t) for t in v] for k, v in local_ts_domain_pairs.items()}
        }
        domain_file = Path(self.cache_manager.cache_folder) / f"processor_domains_{processor_name}.json"
        json_exchange.save_json(domain_file, domain_cache_data)

        # Save processor hash
        self._update_processor_hash(processor_file, processor_name)

        self.logger.log_status("Processing completed.", level="info")

        # Return structured result
        return ProcessorRunnerResult(
            processor_name=processor_name,
            secondary_result=secondary_result,
            ts_domains=local_ts_domains,
            ts_domain_pairs=local_ts_domain_pairs,
        )
