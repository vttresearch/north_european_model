"""
Timeseries processor runner -- dynamic loading and execution of individual processors.

The glue between the orchestrating ``TimeseriesPipeline`` and the processor
classes in ``src/timeseries/processors/``. ``ProcessorRunner`` loads a processor
by name, injects a standard set of kwargs, calls ``run_processor()``, validates
the returned DataFrame, and writes GDX output files.

The processor contract
----------------------
``run_processor()`` returns a :class:`ProcessorOutput` whose ``main_result`` is
a **long-format** frame with exactly:

    bb_parameter_dimensions (excluding 't' and 'f')  +  ['time', 'value']

So ``['grid', 'node', 'f', 't']`` means the processor returns
``['grid', 'node', 'time', 'value']`` -- nothing more, nothing less, with
datetime in ``time``. The runner supplies ``t`` and ``f``:
``split_timeseries_to_climate_windows`` labels ``t`` and inserts ``f00`` as the
realized-weather branch, and ``calculate_climatological_forecasts`` computes
f01, f02, ... from climatological quantiles.

Processors cover the full range from the start of ``start_year`` to
``{end_year}-12-31 23:00``, and must not filter to a particular window or
timeseries length -- climate-window slicing belongs entirely to the runner.

Validation
----------
``main_result`` is rejected -- logged, and no GDX written for that processor --
when it is not a ``pd.DataFrame``, is empty, does not have exactly the required
columns, has a missing value in any *dimension* column, has a non-numeric
``value`` column, has a ``time`` column that cannot be read as datetime, or
fails the time-axis check.

A missing value in ``value`` is **not** a rejection: it means "no data" and
keeps that meaning to the GDX gate, where ``prepare_values_for_gdx`` is the
single place NaN becomes 0 and counts what it converted. The distinction matters
beyond tidiness -- ``calculate_climatological_forecasts`` takes quantiles and
pandas skips NaN, so filling a gap early makes it count as a genuine zero and
biases the whole climatology downward.

Post-processing applied by ProcessorRunner
------------------------------------------
After the interface is validated: coerce ``time`` to datetime if the processor
did not, categorise the dimension columns for memory and groupby speed, round
``value`` to ``rounding_precision``, and apply ``cutoff_below`` -- small
magnitudes to 0, leaving NaN untouched.
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

    Loads the processor class from ``src/timeseries/processors/`` -- the module
    file and the class inside it must share the same name -- instantiates it with
    a standardised set of kwargs derived from the config and the enriched
    processor spec, and calls ``run_processor()``.

    Then, from the returned :class:`ProcessorOutput`:

    - Validates and cures ``main_result`` against the contract above.
    - Writes one GDX per climate window and updates ``import_timeseries.inc``.
    - Computes forecast branches when the spec dimensions include both ``f`` and
      ``t`` and at least one grouping dimension.
    - Persists ``secondary_result`` and per-processor domain data to the cache.
    - Records a hash of the processor file, so the next run's cache manager can
      see that the code changed.
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
        Mark this processor's code as seen at this version.

        Called after every exit from `run`, so that an unchanged processor is not
        rerun next time. The write lives here rather than in CacheManager, which
        only reads hashes: a processor is marked up to date once it has actually
        run, not once someone has asked whether it should.
        """
        hash_value = hash_utils.compute_file_hash(processor_file)
        self.cache_manager.save_processor_hash(processor_name, hash_value)


    def _warn_on_declaration_breaches(
        self, processor_class, values: pd.Series, processor_name: str
    ) -> None:
        """
        Check a processor's own declarations against what it actually produced.

        ``value_range`` and ``value_sign`` are optional class attributes (see
        ``BaseProcessor``), read with ``getattr`` and never required:
        ``ProcessorRunner`` loads a processor by name and imposes no base class,
        so demanding them would break every processor that does not inherit one.

        Breaches are warnings and do not stop the GDX write. A value outside its
        declared range is content, and content can be legitimately surprising; a
        broken time axis is form, and cannot be.
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
                f"produced {lowest}. Either the data is wrong or the declaration is "
                f"out of date.",
                level="warn",
            )
        elif sign == "non_positive" and highest > 0:
            self.logger.log_status(
                f"Processor '{processor_name}' declares non-positive values but "
                f"produced {highest}. Either the data is wrong or the declaration is "
                f"out of date.",
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

        Someone adding a processor reads this error, not the test suite, so it is
        the whole specification of the rule for most people who hit it: what is
        wrong, where, and -- the part that is not guessable -- why it corrupts
        results rather than merely being untidy.
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

        Never raises. Every failure -- a module that will not load, a processor
        that throws, output that breaks the contract -- is logged, leaves no GDX
        for this processor, and returns an empty result. The file hash is updated
        on every one of those paths, so a rerun waits for the code to change.

        Returns
        -------
        ProcessorRunResult
            ``processor_name``, ``secondary_result``, ``ts_domains`` and
            ``ts_domain_pairs`` for the executed processor.
        """
        spec = self.processor_spec["spec"]
        processor_name = self.processor_spec["name"]
        human_name = self.processor_spec["human_name"]
        processor_file = Path(self.processor_spec["file"])

        self.logger.log_status(f"{human_name}", section_start_length=45)

        start_year = self.config["start_year"]
        end_year   = self.config["end_year"]
        bb_ts_start  = self.config["bb_timeseries_start"]
        bb_ts_length = self.config["bb_timeseries_length"]
        country_codes = self.config["country_codes"]
        rounding_precision = spec["rounding_precision"]
        cutoff_below = spec["cutoff_below"]
        bb_parameter = spec["bb_parameter"]
        gdx_name_suffix = spec["gdx_name_suffix"]

        # Which climate years can start a complete window inside the data.
        data_end = pd.Timestamp(f"{end_year}-12-31 23:00")
        valid_climate_years = [
            yr for yr in range(start_year, end_year + 1)
            if pd.Timestamp(f"{yr}-{bb_ts_start}") + pd.Timedelta(bb_ts_length * 24 - 1, unit="h") <= data_end
        ]

        module_spec = importlib.util.spec_from_file_location(processor_name, processor_file)
        if module_spec is None or module_spec.loader is None:
            self.logger.log_status(
                f"Could not load processor module '{processor_name}' from '{processor_file}'. "
                f"Check that the file exists and that timeseries_specs names it correctly. "
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

        # Resolved here rather than just before instantiation, because the class
        # carries requires_source_data and that is read before the kwargs are
        # assembled.
        ProcessorClass = getattr(module, processor_name)

        # Recorded so the next run's CacheManager can rerun this processor when
        # one of those source files changes: it cannot import processor modules
        # itself, and reading the declaration back from cache is cheaper than
        # teaching it to.
        required_source_data = tuple(getattr(ProcessorClass, "requires_source_data", ()) or ())
        self.cache_manager.save_processor_requirements(processor_name, required_source_data)

        # input_folder is pre-joined with the spec's input_sub_folder, so that a
        # processor receives one ready path and needs no knowledge of the base
        # timeseries directory.
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

        # The merged source-data frames the processor declared.
        #
        # SourceDataPipeline.run() is conditional in build_input_data.py, so the
        # object can arrive with every frame still at its empty default. An empty
        # frame means the source excels were skipped this run, not that the user
        # has no data -- so refuse rather than hand over a frame the processor
        # would silently build nothing from. CacheManager forces
        # reimport_source_excels for a declaring processor, making this the
        # backstop for when that wiring is wrong.
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

        main_result = processor_result.main_result
        secondary_result = processor_result.secondary_result

        # --- Validate processor interface ---
        self.logger.log_status(
            f"Validating and curing processor output...",
            level="none",
        )
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
            # Return rather than fall through, since the message above promised
            # no GDX output.
            self._update_processor_hash(processor_file, processor_name)
            return ProcessorRunResult(
                processor_name=processor_name,
                secondary_result=None,
                ts_domains={},
                ts_domain_pairs={},
            )
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
        # The time axis is checked further down, once 'time' is datetime and the
        # ordering it needs has been computed. It subsumes the duplicate-row
        # check that belongs here by subject, at a fortieth of the cost.

        # Dimension values become GAMS set elements, so a missing one is not a
        # value problem but a broken key. Caught here, where the message can name
        # the processor, rather than at the GDX gate, which can only name a file.
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
        #
        # Missing 'value' entries are deliberately NOT filled here -- see the
        # Validation section of the module docstring. This is the tempting place
        # to do it and the wrong one.

        # Processors are expected to return datetime already.
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

        # Categorical dtype for the grouping dimensions: less memory, and faster
        # groupby in everything downstream.
        group_dim_cols = [d for d in spec.get("bb_parameter_dimensions") if d not in ("t", "f")]
        for col in group_dim_cols:
            if col in main_result.columns:
                main_result[col] = main_result[col].astype("category")

        main_result = main_result.round(rounding_precision)

        # Near-zero values to 0, to avoid tiny LP coefficients. The isna() term
        # keeps missing data missing: without it `NaN >= cutoff` is False and
        # every gap would be rewritten as a real 0 here, before the GDX gate can
        # count and report it.
        if cutoff_below is not None:
            values = main_result['value']
            main_result['value'] = values.where(
                values.isna() | (values.abs() >= cutoff_below), 0
            )

        # --- Verify the time axis ---
        # Ordering by (group dimensions, time) is what t-label assignment means,
        # so it is computed once here and handed to the splitter rather than
        # thrown away and redone; checking it costs tens of milliseconds on top.
        #
        # Deliberately after the missing-dimension guard: ngroup() gives -1 to
        # rows whose key is missing and sort_values puts them last, so a frame
        # with blank dimensions would arrive here as one bogus trailing group,
        # and "blank is not a GAMS set element" is the more useful message.
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
        # main_result stays unsorted on purpose: the annual summary, the
        # climatological forecasts and the domain caches all read it below, and
        # reordering it would change the row order of the forecast GDX and the
        # element order of the domain JSON -- same content, different bytes.
        self.logger.log_status("Preparing annual GDX files...")
        annual_dfs = split_timeseries_to_climate_windows(
            ordered_result,
            bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
            bb_ts_start=bb_ts_start,
            bb_ts_length=bb_ts_length,
            valid_climate_years=valid_climate_years,
            group_ids=group_ids,
        )
        # A complete, even axis can still stop before the window does. Tolerated,
        # since valid_climate_years already drops years with no data at all, but
        # worth saying: nothing else the build prints makes it obvious.
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
        GDX_exchange.write_climate_window_GDX_files(
            annual_dfs, self.output_folder, self.logger,
            bb_parameter=bb_parameter,
            bb_parameter_dimensions=spec.get("bb_parameter_dimensions"),
            gdx_name_suffix=gdx_name_suffix,
        )
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
        # Built whenever the spec has 'f', 't' and something to group by.
        dims = spec.get("bb_parameter_dimensions", [])
        calculate_forecasts = "f" in dims and "t" in dims and any(d not in ("f", "t") for d in dims)

        # Empty forecast_quantiles is the deterministic mode: no branches at all.
        if calculate_forecasts and not self.config["forecast_quantiles"]:
            calculate_forecasts = False

        # A quantile across climate years needs more than one of them.
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

            update_import_timeseries_inc(
                self.output_folder,
                file_suffix="forecasts",
                bb_parameter=bb_parameter,
                gdx_name_suffix=gdx_name_suffix,
            )


        # --- Post-processing activities ---
        if secondary_result is not None:
            secondary_output_name = spec.get("secondary_output_name")
            self.cache_manager.save_secondary_result(
                processor_name, secondary_result, secondary_output_name
            )

        domains = ['grid', 'node', 'flow', 'group']
        domain_pairs = [['grid', 'node'], ['flow', 'node']]
        local_ts_domains = collect_domains_for_cache(main_result, domains)
        local_ts_domain_pairs = collect_domain_pairs_for_cache(main_result, domain_pairs)

        # Per processor rather than merged, so that a run which copies one
        # processor's output from a reference folder can copy its domains too.
        domain_cache_data = {
            "ts_domains": {k: list(v) for k, v in local_ts_domains.items()},
            "ts_domain_pairs": {k: [list(t) for t in v] for k, v in local_ts_domain_pairs.items()}
        }
        domain_file = Path(self.cache_manager.cache_folder) / f"processor_domains_{processor_name}.json"
        json_exchange.save_json(domain_file, domain_cache_data)

        self._update_processor_hash(processor_file, processor_name)

        self.logger.log_status("Processing completed.", level="info")

        return ProcessorRunResult(
            processor_name=processor_name,
            secondary_result=secondary_result,
            ts_domains=local_ts_domains,
            ts_domain_pairs=local_ts_domain_pairs,
        )

