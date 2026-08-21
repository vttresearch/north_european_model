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

Validation
----------
``main_result`` is rejected -- logged, and no GDX written for that processor --
when it is not a ``pd.DataFrame``, is empty, does not have exactly the required
columns, contains duplicate ``(dimensions, time)`` rows, has a missing value in
any *dimension* column, has a non-numeric ``value`` column, or has a ``time``
column that cannot be read as datetime.

Missing values in ``value`` are **not** a rejection: they mean "no data", and
they keep that meaning all the way to the GDX gate.

NA and zero
-----------
GAMS has no NaN, and a plain ``0`` *is* empty.  That convention begins at the
GDX boundary and nowhere earlier.  ``GDX_exchange.prepare_values_for_gdx`` is
the single place NaN becomes 0, and it logs how many entries it converted.

The distinction matters beyond tidiness: ``calculate_climatological_forecasts``
computes quantiles, and pandas' ``quantile`` skips NaN.  Filling gaps with 0
before that point makes a missing hour count as a genuine zero and biases the
whole climatology downward.

Post-processing applied by ProcessorRunner
------------------------------------------
After ``run_processor()`` returns and the interface is validated:

1. **Coerce ``time``** to datetime if the processor did not already.
2. **Categorise** dimension columns (memory + groupby speed).
3. **Round** ``value`` to ``rounding_precision`` (default 0).
4. **Apply ``cutoff_below``** -- small magnitudes to 0, leaving NaN untouched.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import pandas as pd
import src.hash_utils as hash_utils
import src.GDX_exchange as GDX_exchange
import src.json_exchange as json_exchange
from src.infrastructure.cache_manager import CacheManager
from src.source_data.source_data_pipeline import SourceDataPipeline
from src.timeseries.timeseries_helpers import (
    collect_domains_for_cache,
    collect_domain_pairs_for_cache,
    find_incomplete_climate_windows,
    find_time_axis_defects,
    order_timeseries_for_labelling,
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


    def _warn_on_declaration_breaches(
        self, processor_class, values: pd.Series, processor_name: str
    ) -> None:
        """
        Check a processor's own declarations against what it actually produced.

        ``value_range`` and ``value_sign`` are optional class attributes on the
        processor (see ``BaseProcessor``). They are read with ``getattr`` and
        never required: ``ProcessorRunner`` loads a processor by name and imposes
        no base class, so demanding the attribute would break every processor
        that does not inherit from one -- including the test fakes.

        Breaches are warnings. A value outside its declared range is content, and
        content can be legitimately surprising; a broken time axis is form, and
        cannot be. So this does not stop the GDX write.
        """
        present = values.dropna()
        if present.empty:
            return
        lowest, highest = float(present.min()), float(present.max())

        declared = getattr(processor_class, "value_range", (None, None))
        try:
            low, high = declared
        except (TypeError, ValueError):
            self.logger.log_status(
                f"Processor '{processor_name}' declares value_range={declared!r}, "
                f"which is not a (minimum, maximum) pair. Ignoring it.",
                level="warn",
            )
            low = high = None

        if low is not None and lowest < low:
            self.logger.log_status(
                f"Processor '{processor_name}' declares values of at least {low} "
                f"but produced {lowest}. Either the data is wrong or the "
                f"declaration is out of date.",
                level="warn",
            )
        if high is not None and highest > high:
            self.logger.log_status(
                f"Processor '{processor_name}' declares values of at most {high} "
                f"but produced {highest}. Either the data is wrong or the "
                f"declaration is out of date.",
                level="warn",
            )

        sign = getattr(processor_class, "value_sign", "any")
        if sign == "non_negative" and lowest < 0:
            self.logger.log_status(
                f"Processor '{processor_name}' declares non-negative values but "
                f"produced {lowest}.",
                level="warn",
            )
        elif sign == "non_positive" and highest > 0:
            self.logger.log_status(
                f"Processor '{processor_name}' declares non-positive values but "
                f"produced {highest}.",
                level="warn",
            )
        elif sign not in ("any", "non_negative", "non_positive"):
            self.logger.log_status(
                f"Processor '{processor_name}' declares value_sign={sign!r}, which "
                f"is not one of 'any', 'non_negative', 'non_positive'. Ignoring it.",
                level="warn",
            )


    @staticmethod
    def _describe_time_axis_defect(
        report, processor_name: str, ordered_result: pd.DataFrame, group_dims: list
    ) -> str:
        """
        Turn a TimeAxisReport into the message its author needs.

        This text is the whole specification of the rule for most people who hit
        it: someone adding a processor reads the error, not the test suite. So it
        has to say what is wrong, where, and -- the part that is not guessable --
        why it corrupts results rather than merely being untidy.
        """
        prefix = f"Processor '{processor_name}'"
        tail = "No GDX output will be written."

        def group_at(index):
            if index is None or not group_dims:
                return None
            row = ordered_result.iloc[index]
            return tuple(row[d] for d in group_dims)

        if report.n_missing_timestamps:
            return (
                f"{prefix} returned {report.n_missing_timestamps} row(s) with a "
                f"missing timestamp. Every row has to sit at a known hour before "
                f"it can be given a t-label. {tail}"
            )

        if report.n_duplicate_or_finer_than_step:
            return (
                f"{prefix} returned {report.n_duplicate_or_finer_than_step} "
                f"duplicate rows: two or more rows fall on the same hour for the "
                f"same group, either repeated timestamps or data finer than "
                f"hourly. First at {report.first_defect_time} in group "
                f"{group_at(report.first_defect_index)}. This causes incorrect "
                f"t-label assignment. {tail}"
            )

        if report.n_gaps:
            return (
                f"{prefix} has {report.n_gaps} gap(s) in its hourly time axis, "
                f"first at {report.first_defect_time} in group "
                f"{group_at(report.first_defect_index)}. t-labels are assigned by "
                f"row position, so a gap does not leave a hole -- it pulls every "
                f"later hour of that group one label earlier, for the rest of the "
                f"window. {tail}"
            )

        first_lo, first_hi = report.group_first_range
        last_lo, last_hi = report.group_last_range
        return (
            f"{prefix} returned groups that do not cover the same hours: they "
            f"start between {first_lo} and {first_hi}, and end between {last_lo} "
            f"and {last_hi}. Each group may be complete on its own and they still "
            f"disagree about which real hour a given t-label names, for the whole "
            f"window. {tail}"
        )


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
        bb_ts_start  = self.config["bb_timeseries_start"]
        bb_ts_length = self.config["bb_timeseries_length"]
        country_codes = self.config["country_codes"]
        rounding_precision = spec["rounding_precision"]
        cutoff_below = spec["cutoff_below"]
        bb_parameter = spec["bb_parameter"]
        gdx_name_suffix = spec["gdx_name_suffix"]

        # Determine which climate years have a complete window within the available data.
        data_end = pd.Timestamp(f"{end_year}-12-31 23:00")
        valid_climate_years = [
            yr for yr in range(start_year, end_year + 1)
            if pd.Timestamp(f"{yr}-{bb_ts_start}") + pd.Timedelta(bb_ts_length * 24 - 1, unit="h") <= data_end
        ]

        # Load processor module
        module_spec = importlib.util.spec_from_file_location(processor_name, processor_file)
        if module_spec is None or module_spec.loader is None:
            self.logger.log_status(
                f"Could not load processor module '{processor_name}' from '{processor_file}'.",
                level="warn",
            )
            self._update_processor_hash(processor_file, processor_name)
            return ProcessorRunResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )
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

        # Resolved here rather than just before instantiation because the class
        # carries requires_source_data, and that has to be read before the kwargs
        # are assembled.
        ProcessorClass = getattr(module, processor_name)

        # Record what this processor asks for, so the next run's CacheManager can
        # rerun it when one of those source files changes. It cannot import
        # processor modules itself, and reading the declaration back from cache is
        # cheaper than teaching it to.
        required_source_data = tuple(getattr(ProcessorClass, "requires_source_data", ()) or ())
        self.cache_manager.save_processor_requirements(processor_name, required_source_data)

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

        # Add the merged source-data frames the processor declared.
        #
        # SourceDataPipeline.run() is conditional in build_input_data.py, so the
        # object can arrive with every frame still at its empty default. An empty
        # frame here means the source excels were skipped this run, not that the
        # user has no data -- so refuse rather than hand the processor an empty
        # frame it would silently build nothing from. CacheManager forces
        # reimport_source_excels for a declaring processor, so this should be
        # unreachable; it is the backstop for when that wiring is wrong.
        for source_name in required_source_data:
            frame = getattr(self.source_data_pipeline, f"df_{source_name}", None)
            if frame is None or frame.empty:
                self.logger.log_status(
                    f"Source data '{source_name}' has not been loaded, which processor "
                    f"'{human_name}' declares it needs. Cannot run it. Re-run to trigger "
                    f"source excel import.",
                    level="warn",
                )
                self._update_processor_hash(processor_file, processor_name)
                return ProcessorRunResult(
                    processor_name=processor_name,
                    secondary_result=None,
                    ts_domains={},
                    ts_domain_pairs={},
                )
            processor_kwargs[f"df_{source_name}"] = frame

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
            # Return rather than fall through: the message above promised no GDX
            # output, but execution used to continue into the curing block and
            # on towards the writers.
            self._update_processor_hash(processor_file, processor_name)
            return ProcessorRunResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
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
        # The time axis -- one complete grid per group, and the same grid -- is
        # checked further down, once 'time' has actually been converted to
        # datetime and the ordering it needs has been computed. It subsumes the
        # duplicate check that used to live here, and costs a fortieth as much.

        # Dimension values become GAMS set elements, so a missing one is not a
        # value problem -- it is a broken key. Caught here, where the message can
        # name the processor, rather than at the GDX gate where it can only name
        # a filename.
        for dim in expected_dims:
            if main_result[dim].isna().any():
                n_missing = int(main_result[dim].isna().sum())
                self.logger.log_status(
                    f"Processor '{processor_name}' returned {n_missing} row(s) with a "
                    f"missing '{dim}' value. Dimension values become GAMS set elements "
                    f"and cannot be blank. No GDX output will be written.",
                    level="error",
                )
                self._update_processor_hash(processor_file, processor_name)
                return ProcessorRunResult(
                    processor_name=processor_name,
                    secondary_result=None,
                    ts_domains={},
                    ts_domain_pairs={},
                )

        if not pd.api.types.is_numeric_dtype(main_result["value"]):
            self.logger.log_status(
                f"Processor '{processor_name}' returned a non-numeric 'value' column "
                f"(dtype {main_result['value'].dtype}). No GDX output will be written.",
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

        # NOTE: missing 'value' entries are deliberately NOT filled here.
        # NaN means "no data" all the way to the GDX gate, where
        # GDX_exchange.prepare_values_for_gdx converts it to 0 and logs how many.
        # Filling at this point silently erased the difference between "no wind"
        # and "no data", and -- worse -- fed those zeros into
        # calculate_climatological_forecasts, which computes quantiles: a gap in
        # the source data was counted as a genuine zero and dragged the whole
        # climatology down. pandas' quantile skips NaN, so leaving it alone is
        # both more honest and more correct.

        # ensure time is datetime only if needed (processors should already return datetime)
        if not pd.api.types.is_datetime64_any_dtype(main_result['time']):
            try:
                main_result['time'] = pd.to_datetime(main_result['time'])
            except (ValueError, TypeError) as e:
                self.logger.log_status(
                    f"Processor '{processor_name}' returned a 'time' column that is not "
                    f"datetime and could not be converted ({e}). "
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
            self.logger.log_status(
                f"Processor '{processor_name}' returned 'time' as "
                f"{main_result['time'].dtype}; converted to datetime. Processors are "
                f"expected to return datetime directly.",
                level="warn",
            )

        # Categorize grouping dimension columns (bb_parameter_dimensions excluding t and f).
        # Categorical dtype reduces memory use and speeds up groupby in downstream functions.
        group_dim_cols = [d for d in spec.get("bb_parameter_dimensions") if d not in ("t", "f")]
        for col in group_dim_cols:
            if col in main_result.columns:
                main_result[col] = main_result[col].astype("category")

        # Round
        main_result = main_result.round(rounding_precision)

        # Drop near-zero values to avoid tiny LP coefficients.
        # The isna() term keeps missing data missing: without it, `NaN >= cutoff`
        # is False and every gap would be quietly rewritten as a real 0 here,
        # before the GDX gate ever gets to count and report it.
        if cutoff_below is not None:
            values = main_result['value']
            main_result['value'] = values.where(
                values.isna() | (values.abs() >= cutoff_below), 0
            )

        # --- Verify the time axis ---
        # Ordering by (group dimensions, time) is what t-label assignment means,
        # so it is computed once here and handed to the splitter rather than
        # thrown away and redone. Checking it costs a few tens of milliseconds
        # on top; the duplicate check this replaced cost about 1.5 s.
        #
        # Deliberately after the missing-dimension guard above: ngroup() gives -1
        # to rows whose key is missing and sort_values puts them last, so a frame
        # with blank dimensions would arrive here as one bogus trailing group.
        # "Blank is not a GAMS set element" is the more useful message, and it
        # has already fired by this point.
        ordered_result, group_ids = order_timeseries_for_labelling(
            main_result, group_dims=group_dim_cols
        )
        time_axis = find_time_axis_defects(ordered_result, group_ids)
        if not time_axis.ok:
            self.logger.log_status(
                self._describe_time_axis_defect(
                    time_axis, processor_name, ordered_result, group_dim_cols
                ),
                level="error",
            )
            self._update_processor_hash(processor_file, processor_name)
            return ProcessorRunResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )

        # Values are checked against the processor's own declarations after the
        # rounding and cutoff above, so what is judged is what gets written.
        self._warn_on_declaration_breaches(
            ProcessorClass, ordered_result["value"], processor_name
        )

        # --- Slice and write climate windows' data ---
        # Split into climate windows.
        # main_result stays unsorted on purpose: the annual summary, the
        # climatological forecasts and the domain caches all read it below, and
        # reordering it would change the row order of the forecast GDX and the
        # element order of the domain JSON -- identical content, different bytes,
        # for no gain.
        self.logger.log_status("Preparing annual GDX files...")
        annual_dfs = split_timeseries_to_climate_windows(
            ordered_result,
            bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
            bb_ts_start=bb_ts_start,
            bb_ts_length=bb_ts_length,
            valid_climate_years=valid_climate_years,
            group_ids=group_ids,
        )
        # A complete, even axis can still stop before the window does. That is
        # tolerated -- valid_climate_years already drops years with no data at
        # all -- but it is worth saying, because a short window is not obvious
        # from anything else the build prints.
        short_windows = find_incomplete_climate_windows(
            annual_dfs, expected_rows=bb_ts_length * 24 * time_axis.n_groups
        )
        if short_windows:
            examples = ", ".join(
                f"{year} ({rows} of {bb_ts_length * 24 * time_axis.n_groups} rows)"
                for year, rows in list(short_windows.items())[:3]
            )
            self.logger.log_status(
                f"Processor '{processor_name}': {len(short_windows)} climate "
                f"window(s) do not cover the full {bb_ts_length}-day window "
                f"because the source data ends first: {examples}. The labels in "
                f"them are correct; the windows are simply shorter.",
                level="warn",
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

        # --- Annual summary CSV ---
        annual_summary = spec.get("annual_summary", "")
        if annual_summary:
            valid_methods = {'avg', 'sum'}
            if annual_summary not in valid_methods:
                self.logger.log_status(
                    f"Processor '{processor_name}': invalid annual_summary value "
                    f"'{annual_summary}'. Expected 'avg' or 'sum'. Skipping summary.",
                    level="warn",
                )
            else:
                self.logger.log_status("Writing annual summary CSV...")
                summary_df = main_result.copy()
                summary_df['year'] = summary_df['time'].dt.year

                group_cols = group_dim_cols + ['year']
                agg_func = 'mean' if annual_summary == 'avg' else 'sum'
                summary_df = (
                    summary_df
                    .groupby(group_cols, observed=True)['value']
                    .agg(agg_func)
                    .round(rounding_precision)
                    .reset_index()
                )
                summary_df['aggregation'] = annual_summary

                summary_filename = f"{bb_parameter}_{gdx_name_suffix}_summary.csv"
                summary_path = os.path.join(self.output_folder, summary_filename)
                summary_df.to_csv(summary_path, index=False)
                self.logger.log_status(
                    f"Annual summary ({annual_summary}) written to {summary_filename}",
                    level="info",
                )

        # --- Climatological forecasts ---
        # Automatically calculate when dimensions include 'f', 't', and at least one grouping dim.
        dims = spec.get("bb_parameter_dimensions", [])
        calculate_forecasts = "f" in dims and "t" in dims and any(d not in ("f", "t") for d in dims)

        # Deterministic mode: empty forecast_quantiles means no forecast branches to compute.
        if calculate_forecasts and not self.config["forecast_quantiles"]:
            calculate_forecasts = False

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

