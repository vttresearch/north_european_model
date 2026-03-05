# src/pipeline/timeseries_processor.py
"""
Timeseries processor runner -- dynamic loading and execution of individual processors.

Purpose
-------
This module provides the glue between the orchestrating ``TimeseriesPipeline``
and the individual processor classes that live in ``src/processors/``.
``ProcessorRunner`` dynamically loads a processor by name, injects a standard
set of kwargs, calls ``run_processor()``, validates the returned DataFrame,
and writes GDX output files.

Data interface -- processor contract
-------------------------------------
Every processor class must implement ``run_processor()`` and return a
:class:`ProcessorResult` whose ``main_result`` is a **long-format**
``pd.DataFrame`` with exactly the following columns:

    bb_parameter_dimensions (excluding 't')  +  ['time', 'value']

For example, if ``bb_parameter_dimensions = ['grid', 'node', 'f', 't']``
the processor must return columns ``['grid', 'node', 'f', 'time', 'value']`` --
nothing more, nothing less.  The ``time`` column must contain datetime values.
The ``t`` dimension is absent from the processor output; it is assigned
downstream by ``_split_timeseries_to_climate_windows`` and
``_calculate_climatological_forecasts``.

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
import numpy as np
import pandas as pd
import src.hash_utils as hash_utils
import src.utils as utils
import src.GDX_exchange as GDX_exchange
import src.json_exchange as json_exchange
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from typing import Dict, List, Optional, Sequence, Any



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

    - Cleans and converts ``main_result`` to long format via ``prepare_BB_df``
      (dimension columns added; ``t`` assigned later by downstream functions).
    - Writes one GDX file per calendar year (``write_BB_gdx_annual``) and
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
    source_excel_data_pipeline: SourceExcelDataPipeline
    scenario_year: Optional[int] = None
    logger: Optional[Any] = None

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
           - Validate that the returned DataFrame has exactly the required columns.
           - Standardize dtypes, fill NA, and round.
           - Write one GDX per calendar year and update ``import_timeseries.inc``.

        4. Climatological forecasts
           - When the spec dimensions include both ``f`` and ``t`` and at least
             one additional grouping dimension, compute quantile-based forecast
             branches via ``_calculate_climatological_forecasts`` and write them
             as a separate ``_forecasts.gdx`` file, also registered in
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
            return ProcessorRunnerResult(
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
            return ProcessorRunnerResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )

        # Extract results from ProcessorResult dataclass
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
                f"Check that run_processor() returns a ProcessorResult.  "
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
        # Processors must return exactly bb_parameter_dimensions (excluding 't') + ['time', 'value'].
        expected_dims = [d for d in spec.get("bb_parameter_dimensions") if d != 't']
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
            return ProcessorRunnerResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )

        # Standardize dtype, fill NA, ensure time is datetime, and round
        #main_result = utils.standardize_df_dtypes(main_result)
        main_result = utils.fill_all_na(main_result)
        main_result['time'] = pd.to_datetime(main_result['time'])
        main_result = main_result.round(rounding_precision)


        # --- Slice and write climate windows' data ---
        # Split into climate windows
        self.logger.log_status("Preparing annual GDX files...")
        annual_dfs = _split_timeseries_to_climate_windows(
            main_result,
            bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
            bb_ts_start=bb_ts_start,
            bb_ts_length=bb_ts_length,
            valid_climate_years=valid_climate_years,
        )
        # Write climate windows' GDX files
        GDX_exchange.write_BB_gdx_annual(
            annual_dfs, self.output_folder, self.logger,
            bb_parameter=bb_parameter,
            bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
            gdx_name_suffix=gdx_name_suffix,
        )
        # Update Backbone ts import instructions file        
        GDX_exchange.update_import_timeseries_inc(
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
            forecast_df = _calculate_climatological_forecasts(
                main_result,
                bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
                quantile_map=spec.get("quantile_map"),
                bb_ts_start=bb_ts_start,
                bb_ts_length=bb_ts_length,
                round_precision=rounding_precision,
            )

            # Write forecast data GDX file
            forecast_gdx_path = os.path.join(
                self.output_folder,
                f"{bb_parameter}_{gdx_name_suffix}_forecasts.gdx"
            )
            GDX_exchange.write_BB_gdx(
                forecast_df, forecast_gdx_path, self.logger,
                bb_parameter=bb_parameter,
                bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
            )
            self.logger.log_status(f"Forecast data GDX written to {forecast_gdx_path}", level="info")                    

            # Update times series Backbone import instructions file
            GDX_exchange.update_import_timeseries_inc(
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
        local_ts_domains = utils.collect_domains_for_cache(main_result, domains)
        local_ts_domain_pairs = utils.collect_domain_pairs_for_cache(main_result, domain_pairs)

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
    




def _split_timeseries_to_climate_windows(
    df: pd.DataFrame,
    *,
    bb_parameter_dimensions: Sequence[str],
    bb_ts_start: str,
    bb_ts_length: int,
    valid_climate_years: List[int],
    ) -> Dict[int, pd.DataFrame]:
    """
    Split a multi-year timeseries DataFrame into per-year climate window chunks
    and assign Backbone t-labels.

    A climate window for year Y starts at {Y}-{bb_ts_start} 00:00 and spans
    bb_ts_length * 24 consecutive hours.  One output DataFrame is produced for
    every year in valid_climate_years for which the data covers a complete window.
    valid_climate_years is computed in run() from the config start/end years and
    bb_ts_start/bb_ts_length, so only years that can start a full window with the 
    available data are included.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format input with columns from bb_parameter_dimensions (excluding 't',
        which is absent from the intermediate format), 'time' (datetime), and 'value'.
    bb_parameter_dimensions : sequence of str
        Backbone dimension names for the output (must include 't').
    bb_ts_start : str
        Window start within each year in "MM-DD" format (e.g. "01-01").
    bb_ts_length : int
        Window length in days.
    valid_climate_years : list of int
        Years for which to extract windows.

    Returns
    -------
    dict[int, pd.DataFrame]
        Keys are climate years; values are DataFrames with columns
        bb_parameter_dimensions + ['value'] and t-labels t000001..t{bb_ts_length*24}.
    """
    dims = list(bb_parameter_dimensions)

    if df["value"].dtype != np.float64:
        df = df.copy()
        df["value"] = df["value"].astype(np.float64)

    # Grouping dimensions exclude f and t
    group_dims = [c for c in dims if c not in {"f", "t"}]

    max_hours = bb_ts_length * 24
    t_labels = np.array(['t' + str(i).zfill(6) for i in range(1, max_hours + 1)])
    final_cols = dims + ["value"]
    out: Dict[int, pd.DataFrame] = {}

    for yr in valid_climate_years:
        window_start = pd.Timestamp(f"{yr}-{bb_ts_start}")
        window_end   = window_start + pd.Timedelta(hours=max_hours - 1)
        mask = (df["time"] >= window_start) & (df["time"] <= window_end)
        df_yr = df[mask].copy()

        # Skipping start years for which there is not enough data for the whole climate window
        if len(df_yr) == 0:
            continue

        if group_dims:
            sort_cols = group_dims + ["time"]
            df_yr = df_yr.sort_values(sort_cols, kind="mergesort")

            group_ids = df_yr.groupby(group_dims, observed=True, sort=False).ngroup()

            # Fast row numbering within groups
            group_changes = np.diff(group_ids.values, prepend=-1) != 0
            row_nums = np.arange(len(df_yr))
            row_nums -= np.repeat(
                row_nums[group_changes],
                np.diff(np.append(np.where(group_changes)[0], len(df_yr))),
            )
            df_yr['_row_num'] = row_nums
        else:
            df_yr['_row_num'] = np.arange(len(df_yr))

        # Safety: truncate if a group somehow has more rows than the window requires
        df_yr = df_yr[df_yr['_row_num'] < max_hours]

        row_nums_filtered = df_yr['_row_num'].values
        df_yr['t'] = t_labels[row_nums_filtered]

        frame = df_yr[final_cols].reset_index(drop=True)
        out[int(yr)] = frame

    return out


def _calculate_climatological_forecasts(
    input_df: pd.DataFrame,
    *,
    bb_parameter_dimensions,
    quantile_map,
    bb_ts_start: str,
    bb_ts_length: int,
    round_precision: int = 0,
    ) -> pd.DataFrame:
    """
    Build stochastic forecast timeseries for Backbone from long-term climatological statistics.

    Backbone can represent uncertainty via multiple forecast branches (f-index).  This
    function creates one such set of branches by computing quantiles of the input timeseries
    across all available climate years, so each branch reflects a different statistical
    outcome drawn from the historical record -- e.g. a wet/dry/median hydro year or a
    warm/cold/normal temperature year.

    The caller controls how many forecasts to create and which quantile each represents via
    the ``quantile_map`` kwarg (default: ``{0.5: 'f01', 0.1: 'f02', 0.9: 'f03'}``).
    Keys are quantile probabilities (0..1) and values are the Backbone f-labels to use.
    A quantile of 0.5 gives the median climate year; 0.1 gives a low-end year; 0.9 a
    high-end year.

    To save disk space, these values are calculated and stored only once. They are the 
    same for every climate_window.

    Algorithm
    ---------
    For every combination of the non-f/t dimension columns (e.g. grid, node):

    1. Compute the requested quantiles across all years at each hour-of-year position
       (1..8760).  Leap-day hours are excluded so that the statistics are always aligned
       on a common 8760-hour calendar.
    2. Map the resulting quantile values onto the output time window
       (``bb_ts_start`` + ``bb_ts_length`` * 24 hours), using hour-of-year as the key.
       Windows longer than one calendar year are tiled correctly.
    3. Assign Backbone t-labels (t000001..) and f-labels from ``quantile_map``.

    Input requirements
    ------------------
    - Long-format DataFrame with the dimension columns from ``bb_parameter_dimensions``
      (excluding 't', which is absent from the intermediate format), plus ``time``
      (datetime) and ``value``.
    - Data must cover more than one climate year (checked before calling this function).
    - ``bb_parameter_dimensions`` must include both ``'f'`` and ``'t'`` and at least one
      other dimension to group by (validated in config_reader).

    Returns
    -------
    pd.DataFrame
        Single-year long-format DataFrame with columns ``bb_parameter_dimensions + ['value']``
        and Backbone t/f labels ready for GDX output.
    """

    # ---- Dimension handling ----
    # Dimension columns are everything in bb_parameter_dimensions except 'f' and 't'.
    dim_cols = [col for col in bb_parameter_dimensions if col not in ("f", "t")]

    # Restrict columns early to reduce memory use in heavy operations.
    # 'f' and 't' are not in the intermediate format; they are assigned here.
    cols_to_keep = dim_cols + ["time", "value"]
    input_df = input_df[cols_to_keep].copy()

    # ---- Create helper columns ----
    # Fast hour_of_year: avoid datetime arithmetic, use dayofyear + hour.
    time = input_df["time"]
    day_of_year = time.dt.dayofyear.to_numpy()
    hour = time.dt.hour.to_numpy()
    hour_of_year = (day_of_year - 1) * 24 + hour + 1

    input_df["hour_of_year"] = hour_of_year.astype(np.int32)

    # Only process hours up to 8760 (ignore extra hours from leap years)
    input_df = input_df[input_df["hour_of_year"] <= 8760]

    # ---- Quantile computation ----
    # Vectorized quantile computation:
    # Group by the additional dimensions and 'hour_of_year' then compute the quantiles.
    # Always computed over the full 8760-hour calendar year regardless of bb_ts_length.
    q_values = list(quantile_map.keys())

    df_quant = (
        input_df
        .groupby(dim_cols + ["hour_of_year"], observed=True)["value"]
        .quantile(q_values)
        # quantile with sequence -> MultiIndex with a 'quantile' level
        .rename_axis(index=dim_cols + ["hour_of_year", "quantile"])
        .reset_index()
    )
    # df_quant now has columns: dim_cols..., 'hour_of_year', 'quantile', 'value'

    # ---- Build window reference sequence ----
    # Generate the sequence of hour_of_year positions (1..8760) that correspond
    # to each hour in the output window.  Use a fixed non-leap reference year (2001)
    # so that the sequence wraps correctly across calendar year boundaries.
    ref_start = pd.Timestamp(f"2001-{bb_ts_start}")
    ref_times = pd.date_range(ref_start, periods=bb_ts_length * 24, freq='h')
    ref_hoy = ((ref_times.dayofyear - 1) * 24 + ref_times.hour + 1).astype(np.int32)
    # Safety clip (should not be needed for non-leap 2001/2002, but guards edge cases)
    ref_hoy = np.clip(ref_hoy, 1, 8760)

    # t-labels for the full window length
    t_labels_arr = np.array(['t' + str(i + 1).zfill(6) for i in range(bb_ts_length * 24)])

    # Window dimension DataFrame: one row per window position
    window_df = pd.DataFrame({
        "hour_of_year": ref_hoy,
        "t": t_labels_arr,
    })

    # ---- Build full grid (Cartesian product) ----
    # Unique combinations of all dimension columns
    unique_dims = input_df[dim_cols].drop_duplicates()

    # Quantiles as in quantile_map (order preserved)
    quantiles_df = pd.DataFrame({"quantile": q_values})

    # Cross join: unique_dims × window_df × quantiles_df
    full_grid = (
        unique_dims
        .merge(window_df, how="cross")
        .merge(quantiles_df, how="cross")
    )

    # Merge the computed quantile results using hour_of_year as the lookup key.
    # Multiple window positions can share the same hour_of_year (e.g. when tiling
    # the average year for bb_ts_length > 365).
    df_full = full_grid.merge(
        df_quant,
        on=dim_cols + ["hour_of_year", "quantile"],
        how="left",
    )

    # ---- Prepare final DataFrame ----
    # 't' is already set from window_df
    df_full["t"] = df_full["t"].astype("category")

    # Map quantile -> f
    df_full["f"] = df_full["quantile"].map(quantile_map)
    df_full["f"] = df_full["f"].astype("category")

    # Fill missing quantile values with 0, then round
    df_full["value"] = df_full["value"].fillna(0)
    if round_precision is not None:
        df_full["value"] = df_full["value"].round(round_precision)

    # Reorder columns to match bb_parameter_dimensions plus 'value'
    df_final = df_full[bb_parameter_dimensions + ["value"]]

    return df_final

