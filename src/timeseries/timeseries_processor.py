"""
Timeseries processor runner -- dynamic loading and execution of individual processors.

Purpose
-------
This module provides the glue between the orchestrating ``TimeseriesPipeline``
and the individual processor classes that live in ``src/timeseries/processors/``.
``ProcessorRunner`` dynamically loads a processor by name, injects a standard
set of kwargs, calls ``run_processor()``, validates the returned DataFrame,
and writes GDX output files.

Data interface -- processor contract
-------------------------------------
Every processor class must implement ``run_processor()`` and return a
:class:`ProcessorOutput` whose ``main_result`` is a **long-format**
``pd.DataFrame`` with exactly the following columns:

    bb_parameter_dimensions (excluding 't' and 'f')  +  ['time', 'value']

For example, if ``bb_parameter_dimensions = ['grid', 'node', 'f', 't']``
the processor must return columns ``['grid', 'node', 'time', 'value']`` --
nothing more, nothing less.  The ``time`` column must contain datetime values.
The ``t`` and ``f`` dimensions are absent from the processor output.
``split_timeseries_to_climate_windows`` assigns ``t`` and inserts ``f00``
as the realized-weather branch.  ``calculate_climatological_forecasts``
computes the remaining forecast branches (f01, f02, …) from climatological
quantiles.

Processors must cover the full date range from the start of ``start_year``
to the end of ``end_year`` (i.e. ``{end_year}-12-31 23:00``).  Climate-window
slicing is handled entirely by the runner; processors must not filter to a
particular window or timeseries length.

If ``main_result`` is not a ``pd.DataFrame``, is empty, or does not have the
required columns, ``ProcessorRunner`` logs a warning/error and continues --
no GDX is written for that processor.

Post-processing applied by ProcessorRunner
------------------------------------------
After ``run_processor()`` returns and the interface is validated:

1. **Standardize dtypes** -- ``utils.standardize_df_dtypes`` converts numeric
   columns to ``Float64``.
2. **Fill numeric NA** -- ``utils.fill_all_na`` replaces ``pd.NA`` with ``0``.
3. **Round** -- ``value`` is rounded to ``rounding_precision`` (default 0).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import pandas as pd
import src.hash_utils as hash_utils
import src.utils as utils
import src.GDX_exchange as GDX_exchange
import src.json_exchange as json_exchange
from src.infrastructure.cache_manager import CacheManager
from src.source_data.source_data_pipeline import SourceDataPipeline
from src.timeseries.timeseries_helpers import (
    collect_domains_for_cache,
    collect_domain_pairs_for_cache,
    update_import_timeseries_inc,
    split_timeseries_to_climate_windows,
    calculate_climatological_forecasts,
)
from src.timeseries.timeseries_results import ProcessorOutput, ProcessorRunResult
from src.infrastructure.logger import IterationLogger
from typing import Optional


@dataclass
class ProcessorRunner:
    """
    Executes a single timeseries processor and writes its outputs.

    Dynamically loads the processor class from ``src/timeseries/processors/`` (the module
    file and the class inside it must share the same name), instantiates it with
    a standardised set of kwargs derived from the config and the enriched
    processor spec, and calls ``run_processor()``.

    After the processor returns a :class:`ProcessorOutput`, this class:

    - Cleans and converts ``main_result`` to long format via ``prepare_BB_df``
      (dimension columns added; ``t`` assigned later by downstream functions).
    - Writes one GDX file per climate window (``write_climate_window_GDX_files``) and
      updates ``import_timeseries.inc``.
    - Computes climatological forecast GDX when the spec dimensions include
      both ``f`` and ``t`` and at least one additional grouping dimension.
    - Persists ``secondary_result`` and per-processor domain data to the cache.
    - Records a hash of the processor file so the cache manager can detect
      code changes on the next run.
    """
    processor_spec: dict
    config: dict
    input_folder: Path
    output_folder: Path
    cache_manager: CacheManager
    source_data_pipeline: SourceDataPipeline
    logger: IterationLogger
    scenario_year: Optional[int] = None

    def _update_processor_hash(self, processor_file: Path, processor_name: str):
        """
        Update the cached hash for this processor.

        This is called after processor execution (successful or skipped) to mark
        the processor code as "seen" at this version. This prevents unnecessary
        reruns when the processor hasn't changed.

        Note: This is separate from CacheManager._detect_processor_code_changes()
        which only READS hashes to determine what needs to run. The update happens
        here to ensure we only mark processors as "up-to-date" after they've
        actually executed successfully.

        The function is thin, but the purpose is hopefully easier to catch 
        witht this docstring.
        """
        hash_value = hash_utils.compute_file_hash(processor_file)
        self.cache_manager.save_processor_hash(processor_name, hash_value)


    def run(self) -> ProcessorRunResult:
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
             empty :class:`ProcessorRunResult` (hash still updated).

        3. Run and convert
           - Instantiate the processor class and call ``run_processor()``.
           - Drop empty columns, standardise dtypes, and round to
             ``rounding_precision``.
           - Validate that the returned DataFrame has exactly the required columns.
           - Standardize dtypes, fill NA, and round.
           - Write one GDX per calendar year and update ``import_timeseries.inc``.

        4. Climatological forecasts
           - When the spec dimensions include both ``f`` and ``t`` and at least
             one additional grouping dimension, compute quantile-based forecast
             branches via ``calculate_climatological_forecasts`` and write them
             as a separate ``_forecasts.gdx`` file, also registered in
             ``import_timeseries.inc``.

        5. Post-processing
           - Save ``secondary_result`` to the cache if present.
           - Collect domain values and domain pairs from the converted DataFrame
             and save them to a per-processor JSON cache file.
           - Update the processor file hash.

        Returns
        -------
        ProcessorRunResult
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
        start_year = self.config["start_year"]
        end_year   = self.config["end_year"]
        bb_ts_start  = self.config.get("bb_timeseries_start")
        bb_ts_length = self.config.get("bb_timeseries_length")
        country_codes = self.config["country_codes"]
        rounding_precision = spec.get("rounding_precision")
        bb_parameter = spec.get("bb_parameter")
        gdx_name_suffix = spec.get("gdx_name_suffix")

        # Determine which climate years have a complete window within the available data.
        data_end = pd.Timestamp(f"{end_year}-12-31 23:00")
        valid_climate_years = [
            yr for yr in range(start_year, end_year + 1)
            if pd.Timestamp(f"{yr}-{bb_ts_start}") + pd.Timedelta(hours=bb_ts_length * 24 - 1) <= data_end
        ]

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
            return ProcessorRunResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )

        # Prepare processor kwargs
        # input_folder is pre-joined with the spec's input_sub_folder so that
        # processors receive a single ready-to-use path and need no knowledge of the
        # base timeseries directory.
        ts_base = os.path.join(self.input_folder, "timeseries")
        processor_kwargs = {
            "input_folder": os.path.join(ts_base, spec.get("input_sub_folder") or ""),
            "country_codes": country_codes,
            "start_year": start_year,
            "end_year": end_year,
            "scenario_year": self.scenario_year,
            "exclude_nodes": self.config["exclude_nodes"],
            "logger": self.logger,
            **{k: v for k, v in spec.items() if k != "input_sub_folder"},
        }

        # Add demand data to processor_kwargs if demand grid
        demand_grid = spec.get("demand_grid")
        if demand_grid:
            df_annual_demands = self.source_data_pipeline.df_demanddata
            if "grid" not in df_annual_demands.columns:
                self.logger.log_status(
                    f"Demand data has not been loaded (df_demanddata has no columns). "
                    f"Cannot run processor '{human_name}'. Re-run to trigger source excel import.",
                    level="warn",
                )
                self._update_processor_hash(processor_file, processor_name)
                return ProcessorRunResult(
                    processor_name=processor_name,
                    secondary_result=None,
                    ts_domains={},
                    ts_domain_pairs={},
                )
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
                return ProcessorRunResult(
                    processor_name=processor_name,
                    secondary_result=None,
                    ts_domains={},
                    ts_domain_pairs={},
                )
            processor_kwargs["df_annual_demands"] = df_filtered


        # Instantiate and run processor
        ProcessorClass = getattr(module, processor_name)
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
            return ProcessorRunResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )

        # Extract results from ProcessorOutput dataclass
        main_result = processor_result.main_result
        secondary_result = processor_result.secondary_result


        # --- Validate processor interface ---
        self.logger.log_status(
            f"Validating and curing processor output...",
            level="none",
        )
        # Guard: processors must return a DataFrame (see module docstring)
        if not isinstance(main_result, pd.DataFrame):
            self.logger.log_status(
                f"Processor '{processor_name}' returned main_result of type "
                f"'{type(main_result).__name__}', expected pd.DataFrame.  "
                f"Check that run_processor() returns a ProcessorOutput.  "
                f"No GDX output will be written.",
                level="error",
            )
            main_result = pd.DataFrame()
        if main_result.empty:
            self.logger.log_status(
                f"Processor '{processor_name}' returned an empty DataFrame.  "
                f"No GDX output will be written.",
                level="warn",
            )
        # Processors must return exactly bb_parameter_dimensions (excluding 't' and 'f') + ['time', 'value'].
        expected_dims = [d for d in spec.get("bb_parameter_dimensions") if d not in ('t', 'f')]
        expected_cols = set(expected_dims + ['time', 'value'])
        actual_cols = set(main_result.columns)
        if actual_cols != expected_cols:
            self.logger.log_status(
                f"Processor '{processor_name}' returned unexpected columns. "
                f"Expected {sorted(expected_cols)}, got {sorted(actual_cols)}. "
                f"No GDX output will be written.",
                level="error",
            )
            self._update_processor_hash(processor_file, processor_name)
            return ProcessorRunResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )
        # Processors must return at most one row per (group, time) combination.
        # Duplicates indicate a processor bug (e.g. duplicate timestamps or sub-hourly data)
        # and would cause silent data corruption during t-label assignment.
        key_cols = expected_dims + ['time']
        if main_result.duplicated(subset=key_cols).any():
            dup_count = main_result.duplicated(subset=key_cols).sum()
            self.logger.log_status(
                f"Processor '{processor_name}' returned {dup_count} duplicate rows "
                f"(same group+time combination). This causes incorrect t-label assignment. "
                f"No GDX output will be written.",
                level="error",
            )
            self._update_processor_hash(processor_file, processor_name)
            return ProcessorRunResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )

        # --- cure and standardize main results ---

        # fill NA only if needed (avoids a full DataFrame copy when processors return clean data)
        if main_result.isnull().values.any():
            main_result = utils.fill_all_na(main_result)
        # ensure time is datetime only if needed (processors should already return datetime)
        if not pd.api.types.is_datetime64_any_dtype(main_result['time']):
            main_result['time'] = pd.to_datetime(main_result['time'])

        # Categorize grouping dimension columns (bb_parameter_dimensions excluding t and f).
        # Categorical dtype reduces memory use and speeds up groupby in downstream functions.
        group_dim_cols = [d for d in spec.get("bb_parameter_dimensions") if d not in ("t", "f")]
        for col in group_dim_cols:
            if col in main_result.columns:
                main_result[col] = main_result[col].astype("category")

        # Round
        main_result = main_result.round(rounding_precision)

        # --- Slice and write climate windows' data ---
        # Split into climate windows
        self.logger.log_status("Preparing annual GDX files...")
        annual_dfs = split_timeseries_to_climate_windows(
            main_result,
            bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
            bb_ts_start=bb_ts_start,
            bb_ts_length=bb_ts_length,
            valid_climate_years=valid_climate_years,
        )
        # Write climate windows' GDX files
        GDX_exchange.write_climate_window_GDX_files(
            annual_dfs, self.output_folder, self.logger,
            bb_parameter=bb_parameter,
            bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
            gdx_name_suffix=gdx_name_suffix,
        )
        # Update Backbone ts import instructions file
        update_import_timeseries_inc(
            self.output_folder,
            bb_parameter=bb_parameter,
            gdx_name_suffix=gdx_name_suffix,
        )

        # --- Climatological forecasts ---
        # Automatically calculate when dimensions include 'f', 't', and at least one grouping dim.
        dims = spec.get("bb_parameter_dimensions", [])
        calculate_forecasts = "f" in dims and "t" in dims and any(d not in ("f", "t") for d in dims)

        # Guard: requires multi-year data.
        if calculate_forecasts:
            unique_years = main_result["time"].dt.year.unique()
            if len(unique_years) <= 1:
                self.logger.log_status(
                    f"Processor '{processor_name}': data covers only {len(unique_years)} year(s); "
                    "cannot calculate climatological forecasts.",
                    level="warn",
                )
                calculate_forecasts = False

        if calculate_forecasts:
            self.logger.log_status("Calculating climatological forecasts...")
            forecast_df = calculate_climatological_forecasts(
                main_result,
                bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
                forecast_quantiles=self.config["forecast_quantiles"],
                bb_ts_start=bb_ts_start,
                bb_ts_length=bb_ts_length,
                round_precision=rounding_precision,
            )

            # Write forecast data GDX file
            forecast_gdx_path = os.path.join(
                self.output_folder,
                f"{bb_parameter}_{gdx_name_suffix}_forecasts.gdx"
            )
            GDX_exchange.write_df_to_gdx(
                forecast_df, forecast_gdx_path, self.logger,
                parameter_name=bb_parameter,
                parameter_dimensions=spec.get("bb_parameter_dimensions"),
            )
            self.logger.log_status(f"Forecast data GDX written to {forecast_gdx_path}", level="info")                    

            # Update times series Backbone import instructions file
            update_import_timeseries_inc(
                self.output_folder,
                file_suffix="forecasts",
                bb_parameter=bb_parameter,
                gdx_name_suffix=gdx_name_suffix,
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
        local_ts_domains = collect_domains_for_cache(main_result, domains)
        local_ts_domain_pairs = collect_domain_pairs_for_cache(main_result, domain_pairs)

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
        return ProcessorRunResult(
            processor_name=processor_name,
            secondary_result=secondary_result,
            ts_domains=local_ts_domains,
            ts_domain_pairs=local_ts_domain_pairs,
        )

