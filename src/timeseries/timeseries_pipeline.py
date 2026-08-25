"""
Timeseries pipeline -- orchestration of timeseries processor execution.

``TimeseriesPipeline.run()`` decides which processors need to execute, runs or
copies them, handles demand grids that have no explicit processor, and writes
the GDX outputs and GAMS include directives Backbone needs. What a processor
must return, and what happens to it, is in ``timeseries_processor.py``.

What the pipeline accumulates across processors
-----------------------------------------------
``ts_domains`` maps a dimension column to the values seen in it, e.g.
``{"grid": {"elec", "dheat"}}``; ``ts_domain_pairs`` maps a compound key to the
combinations actually present, from ``['grid','node']`` and ``['flow','node']``
where both columns exist. Both are merged into shared cache files
(``all_ts_domains.json``, ``all_ts_domain_pairs.json``); ``BBExcelPipeline``
normalises the names downstream.

``secondary_results`` maps a processor's module name to whatever extra output it
produced, persisted as pickles in the cache so that a later phase can use it
without re-running the processor.

Output
------
Per executed processor, into ``output_folder``: one or more
``{bb_parameter}_{gdx_name_suffix}.gdx`` files, and a matching ``$gdxin``
directive appended to ``import_timeseries.inc``. Demand grids with no explicit
processor get ``ts_influx_other_demands.gdx``.
"""

from pathlib import Path
import importlib.util
import shutil
import glob as glob_module
import pickle
import pandas as pd
from src.infrastructure.cache_manager import CacheManager
from src.source_data.source_data_pipeline import SourceDataPipeline
from src.timeseries.timeseries_inputs import TimeseriesPipelineInputs
from src.timeseries.timeseries_processor import ProcessorRunner
from src.timeseries.timeseries_results import TimeseriesPipelineOutput
from src.timeseries.timeseries_helpers import (
    collect_domains_for_cache,
    collect_domain_pairs_for_cache,
    update_import_timeseries_inc,
)
import src.GDX_exchange as GDX_exchange
import src.json_exchange as json_exchange


class TimeseriesPipeline:
    """
    Orchestrates the execution of timeseries processors based on configuration.
    """

    def __init__(self, inputs: TimeseriesPipelineInputs):
        self.config = inputs.config
        self.input_folder = inputs.input_folder
        self.output_folder = inputs.output_folder
        self.cache_manager = inputs.cache_manager
        self.source_data_pipeline = inputs.source_data_pipeline
        self.reference_ts_folder = inputs.reference_ts_folder
        self.scenario_year = inputs.scenario_year
        self.secondary_results = {}
        self.logger = inputs.logger
        self.df_annual_demands = inputs.source_data_pipeline.df_demanddata



    def _create_enriched_processor_specs(self) -> list[dict]:
        """
        Build the list of enriched processor specs used throughout the pipeline.

        ``config_reader.py`` has already validated the mandatory fields and
        injected the defaults, so nothing is checked here. A processor whose
        ``demand_grid`` is in ``exclude_grids`` is skipped.

        Returns
        -------
        list[dict]
            Per processor: ``human_name`` (the config key), ``name``
            (``processor_name``, used for logging and result tracking), ``file``
            (path to the .py), and ``spec`` (the spec dict from config).
        """
        specs: list[dict] = []
        timeseries_specs: dict = self.config["timeseries_specs"]
        exclude_grids: list[str] = self.config["exclude_grids"]
        # Anchored to this file, not the working directory: the processors live
        # beside it and their location has nothing to do with where the build
        # was started from.
        processors_base = Path(__file__).resolve().parent / "processors"

        for human_name, spec in timeseries_specs.items():
            processor_name: str = spec["processor_name"]

            if spec["demand_grid"] and spec["demand_grid"] in exclude_grids:
                self.logger.log_status(
                    f"Skipping {processor_name}: its demand grid '{spec['demand_grid']}' is in "
                    f"exclude_grids.",
                    level="warn"
                )
                continue

            processor_file = processors_base / f"{processor_name}.py"

            enriched_spec: dict = {
                "name": processor_name,
                "file": str(processor_file),
                "spec": spec,
                "human_name": human_name,
            }
            specs.append(enriched_spec)

        return specs


    def _declared_source_data(self, processor_spec: dict) -> tuple[str, ...]:
        """
        Read ``requires_source_data`` off a processor class without running it.

        From the class rather than from the cache, because the answer decides
        whether the processor may be *skipped* -- and a cache written by an
        earlier run in a different output folder is the wrong authority for that.

        Any failure returns an empty tuple. A module that will not import is
        ProcessorRunner's to report, and it does so naming the file.
        """
        try:
            module_spec = importlib.util.spec_from_file_location(
                processor_spec["name"], Path(processor_spec["file"])
            )
            if module_spec is None or module_spec.loader is None:
                return ()
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            processor_cls = getattr(module, processor_spec["name"], None)
            return tuple(getattr(processor_cls, "requires_source_data", ()) or ())
        except Exception:
            return ()


    def _create_other_demands(
        self, df_annual_demands: pd.DataFrame, other_demands: set[str]
    ) -> pd.DataFrame:
        """
        Flat hourly demand for grids no explicit processor covers.

        Every (grid, node) whose grid is in ``other_demands`` gets the whole
        window at one value: the annual ``twh/year`` spread evenly, negative
        because it is demand. There is no profile to give it -- a grid reaches
        here precisely because nothing knows its shape.

        Parameters
        ----------
        df_annual_demands : pd.DataFrame
            Needs ``grid``, ``node`` and ``twh/year``. Anything less logs a
            warning and yields an empty frame of the right shape.
        other_demands : set[str]
            Lowercased grid names to build.

        Returns
        -------
        pd.DataFrame
            [``grid``, ``node``, ``f``, ``t``, ``value``], one row per window
            hour per (grid, node), or empty with those columns.
        """
        required_cols = {"grid", "twh/year", "node"}
        missing_cols = required_cols - set(df_annual_demands.columns)

        if missing_cols:
            self.logger.log_status(
                f"The demand table has no {', '.join(sorted(missing_cols))} column, so the "
                f"grids without a processor of their own cannot be built: "
                f"{', '.join(sorted(other_demands))}. Those nodes get no demand at all.",
                level="warn",
            )
            return pd.DataFrame(columns=["grid", "node", "f", "t", "value"])

        df_filtered = df_annual_demands[
            df_annual_demands["grid"].str.lower().isin(other_demands)
        ]

        if df_filtered.empty:
            return pd.DataFrame(columns=["grid", "node", "f", "t", "value"])

        bb_ts_length: int = self.config["bb_timeseries_length"]
        t_index = [f"t{str(i).zfill(6)}" for i in range(1, bb_ts_length * 24 + 1)]

        rows: list[pd.DataFrame] = []
        for _, row in df_filtered.iterrows():
            try:
                # Always a nominal 8760-hour year, whatever bb_ts_length is: the
                # rate is per hour, and a longer window repeats it rather than
                # dividing the same energy more thinly.
                hourly_value = round(row["twh/year"] * 1e6 / 8760 * -1, 2)
            except Exception as e:
                self.logger.log_status(
                    f"Node '{row.get('node')}' has a twh/year of {row.get('twh/year')!r}, "
                    f"which is not a number ({e}). It gets no demand at all.",
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
        Which demand grids no explicit processor covers.

        The grids in ``df_annual_demands`` against the ``demand_grid`` fields in
        ``timeseries_specs``. ``exclude_grids`` is not applied here -- those are
        dropped upstream, in ``_create_enriched_processor_specs``.

        Returns
        -------
        set[str]
            Lowercased grid names in the demand data that no processor spec
            claims. Empty if ``df_annual_demands`` is absent or has no ``grid``.
        """
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

        timeseries_specs = self.config["timeseries_specs"]

        processed_grids = set()
        for spec in timeseries_specs.values():
            demand_grid = spec.get("demand_grid", "").lower()
            if demand_grid:
                processed_grids.add(demand_grid)

        return all_demand_grids - processed_grids


    def _copy_processor_from_reference(self, processor_spec: dict) -> dict:
        """
        Reuse an input-data-independent processor's outputs instead of running it.

        Everything the processor would have produced has to come across, not just
        the GDX files: the secondary result, the domain caches and the processor
        hash all have consumers that cannot tell a copy from a run, and a
        half-copied processor looks to the next build like one that half failed.
        The numbered steps below are those five things.

        Missing pieces are warnings rather than errors, and the run continues on
        whatever did arrive.

        Returns
        -------
        dict
            ``secondary_result`` (or None), ``ts_domains`` and
            ``ts_domain_pairs`` as read from the reference cache.
        """
        spec = processor_spec["spec"]
        processor_name = processor_spec["name"]
        human_name = processor_spec["human_name"]
        bb_parameter = spec.get("bb_parameter")
        gdx_name_suffix = spec.get("gdx_name_suffix")

        self.logger.log_status(f"{human_name}", section_start_length=45)

        if self.reference_ts_folder is None:
            self.logger.log_status(
                "No reference folder configured. Cannot copy.",
                level="warn"
            )
            return {"secondary_result": None, "ts_domains": {}, "ts_domain_pairs": {}}

        ref_folder = Path(self.reference_ts_folder)

        if not ref_folder.exists():
            self.logger.log_status(
                f"Reference folder {ref_folder} does not exist. Cannot copy.",
                level="warn"
            )
            return {"secondary_result": None, "ts_domains": {}, "ts_domain_pairs": {}}

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
            self.logger.log_status(
                f"Copied {copied_count} GDX file(s) from the reference folder.", level="info"
            )
        else:
            self.logger.log_status(
                f"No {fname_base}*.gdx in the reference folder {ref_folder}, so there is "
                f"nothing to copy and this processor produces no output.",
                level="warn"
            )

        # 2. Update import_timeseries.inc for this processor
        bb_kwargs = {"bb_parameter": bb_parameter, "gdx_name_suffix": gdx_name_suffix}
        update_import_timeseries_inc(self.output_folder, **bb_kwargs)

        dims = spec.get("bb_parameter_dimensions", [])
        if "f" in dims and "t" in dims and any(d not in ("f", "t") for d in dims):
            update_import_timeseries_inc(self.output_folder, file_suffix="forecasts", **bb_kwargs)

        # 3. Copy the secondary result, if the spec names one
        secondary_result = None
        secondary_output_name = spec.get("secondary_output_name")
        if secondary_output_name:
            ref_pkl = ref_folder / "cache" / "secondary_results" / f"{processor_name}.pkl"
            if ref_pkl.exists():
                dest_pkl = self.cache_manager.secondary_results_folder / f"{processor_name}.pkl"
                dest_pkl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ref_pkl, dest_pkl)
                self.logger.log_status(f"Copied secondary result: {processor_name}.pkl", level="info")
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
            self.logger.log_status(
                f"No domain cache at {ref_domain_file}, so the copied GDX files are not "
                f"registered in the input Excel. Build the reference folder again to create it.",
                level="warn"
            )

        # 5. Copy the processor hash, so this cache agrees the copy is current
        ref_hash_file = ref_folder / "cache" / "processor_hashes.json"
        if ref_hash_file.exists():
            ref_hashes = json_exchange.load_json(ref_hash_file)
            if processor_name in ref_hashes:
                self.cache_manager.save_processor_hash(processor_name, ref_hashes[processor_name])

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
        }


    def run(self) -> TimeseriesPipelineOutput:
        """
        Execute the full timeseries processing pipeline.

        The numbered sections below are the workflow: decide what to run, run it,
        copy what can be copied from a reference folder instead, build flat
        series for demand grids no processor claims, and merge everything into
        the cache.

        Returns
        -------
        TimeseriesPipelineOutput
            ``secondary_results``, and ``ts_domains`` / ``ts_domain_pairs`` as
            sorted lists ready for ``BBExcelPipeline``.
        """
        # --- 1. Initialization ---
        # A full rerun starts the include file over rather than appending to a
        # file that still registers GDX files it is about to replace.
        if self.cache_manager.full_rerun:
            p = Path(self.output_folder) / "import_timeseries.inc"
            p.unlink(missing_ok=True)

        # --- 2. Determine processors to run ---
        self.logger.log_status(
            "Checking the status of timeseries processors...",
            level="none",
            add_empty_line_before=True
        )

        processors_to_rerun = set()
        self.processors = self._create_enriched_processor_specs()

        # timeseries_changed already folds in config changes and code changes.
        for proc in self.processors:
            human_name = proc["human_name"]
            needs_rerun = (
                self.cache_manager.full_rerun
                or self.cache_manager.timeseries_changed.get(human_name, False)
            )

            if needs_rerun:
                processors_to_rerun.add(human_name)

        # A processor that declares requires_source_data is input-data-dependent
        # by construction: its frames are whitelisted per scenario, year and
        # country, so a copy from another scenario's folder would be that
        # scenario's answer wearing this one's name. The declaration overrides
        # the config, not the other way round.
        processors_to_copy = set()
        if self.reference_ts_folder and Path(self.reference_ts_folder) != Path(self.output_folder):
            timeseries_specs_raw = self.config["timeseries_specs"]
            for proc in self.processors:
                human_name = proc["human_name"]
                if human_name not in processors_to_rerun:
                    continue
                spec = timeseries_specs_raw.get(human_name, {})
                if spec.get('is_input_data_dependent', True):
                    continue

                declared = self._declared_source_data(proc)
                if declared:
                    self.logger.log_status(
                        f"'{human_name}' is configured is_input_data_dependent: false, but "
                        f"{proc['name']} declares it needs source data ({', '.join(declared)}), "
                        f"which is scenario-specific. Running it instead of copying. "
                        f"Set is_input_data_dependent: true in the config to silence this.",
                        level="warn"
                    )
                    continue

                processors_to_copy.add(human_name)
                processors_to_rerun.discard(human_name)

        self.logger.log_status(
            f"Need to run {len(processors_to_rerun)} timeseries processor(s): "
            f"{', '.join(sorted(processors_to_rerun)) if processors_to_rerun else 'none'}",
            level="info"
        )
        if processors_to_copy:
            self.logger.log_status(
                f"{len(processors_to_copy)} processor(s) will be copied from reference folder: "
                f"{', '.join(sorted(processors_to_copy))}",
                level="info"
            )

        # Said once here rather than per processor: every processor reaches the
        # same answer, and a window longer than a year is the usual cause.
        start_year: int = self.config["start_year"]
        end_year: int = self.config["end_year"]
        bb_ts_start: str = self.config["bb_timeseries_start"]
        bb_ts_length: int = self.config["bb_timeseries_length"]
        data_end = pd.Timestamp(f"{end_year}-12-31 23:00")

        if processors_to_rerun:
            self.logger.log_status(
                f"Building a {bb_ts_length}-day window from {bb_ts_start} for every climate "
                f"year from {start_year} onwards...",
                level="none"
            )

            excluded_years = [
                yr for yr in range(start_year, end_year + 1)
                if pd.Timestamp(f"{yr}-{bb_ts_start}") + pd.Timedelta(bb_ts_length * 24 - 1, unit="h") > data_end
            ]
            if excluded_years:
                self.logger.log_status(
                    f"Climate years {excluded_years[0]} onwards are not built: a {bb_ts_length}-day "
                    f"window opening on {bb_ts_start} would run past the end of {end_year}.",
                    level="none"
                )

        # --- 3. Run selected processors ---
        all_ts_domains = {}
        all_ts_domain_pairs = {}

        if processors_to_rerun:
            processor_iter = (p for p in self.processors if p['human_name'] in processors_to_rerun)

            for processor in processor_iter:
                runner = ProcessorRunner(
                    processor_spec=processor,
                    config=self.config,
                    input_folder=self.input_folder,
                    output_folder=self.output_folder,
                    source_data_pipeline=self.source_data_pipeline,
                    cache_manager=self.cache_manager,
                    scenario_year=self.scenario_year,
                    logger=self.logger
                )
                self.logger.log_status(f"Running: {processor['name']}", level="run", add_empty_line_before=True)

                result = runner.run()

                self.secondary_results[result.processor_name] = result.secondary_result

                for dom, vals in result.ts_domains.items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in result.ts_domain_pairs.items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

        # --- 3b. Copy input-data-independent processors from reference folder ---
        if processors_to_copy:
            copy_iter = (p for p in self.processors if p['human_name'] in processors_to_copy)

            for processor in copy_iter:
                self.logger.log_status(f"Copying: {processor['name']}", level="run", add_empty_line_before=True)

                copy_result = self._copy_processor_from_reference(processor)

                self.secondary_results[processor["name"]] = copy_result["secondary_result"]

                for dom, vals in copy_result["ts_domains"].items():
                    all_ts_domains.setdefault(dom, set()).update(vals)

                for pair_key, tuples in copy_result["ts_domain_pairs"].items():
                    all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)


        # --- 4. Process Other Demands ---
        self.logger.log_status(f"Remaining timeseries actions", level="run", section_start_length=45, add_empty_line_before=True)

        unprocessed_grids = self._get_unprocessed_demand_grids()

        if unprocessed_grids:
            self.logger.log_status("Processing other demands", level="none")
            for grid in sorted(unprocessed_grids):
                self.logger.log_status(f" .. {grid}", level="none")

            df_other_demands = self._create_other_demands(self.df_annual_demands, unprocessed_grids)

            other_domains = collect_domains_for_cache(df_other_demands, ['grid', 'node'])
            other_domain_pairs = collect_domain_pairs_for_cache(df_other_demands, [['grid', 'node']])

            for dom, vals in other_domains.items():
                all_ts_domains.setdefault(dom, set()).update(vals)

            for pair_key, tuples in other_domain_pairs.items():
                all_ts_domain_pairs.setdefault(pair_key, set()).update(tuples)

            output_file_other = self.output_folder / "ts_influx_other_demands.gdx"
            GDX_exchange.write_df_to_gdx(df_other_demands, str(output_file_other), self.logger,
                            parameter_name="ts_influx",
                            parameter_dimensions=["grid", "node", "f", "t"])
            update_import_timeseries_inc(self.output_folder, bb_parameter="ts_influx", gdx_name_suffix="other_demands")


        # --- 5. Cache management ---

        # Merged in and read straight back out, so that what is returned includes
        # the processors that did not run this session.
        self.cache_manager.merge_dict_to_cache(all_ts_domains, "all_ts_domains.json")
        all_ts_domains = self.cache_manager.load_dict_from_cache("all_ts_domains.json")

        self.cache_manager.merge_dict_to_cache(all_ts_domain_pairs, "all_ts_domain_pairs.json")
        all_ts_domain_pairs = self.cache_manager.load_dict_from_cache("all_ts_domain_pairs.json")

        # Rebuilding the Excel needs every processor's secondary result, not only
        # this session's, so the cache fills in the rest -- with this run winning
        # wherever both have an answer.
        if self.cache_manager.rebuild_bb_excel:
            self.logger.log_status("Loading all secondary results from cache.", level="none")
            all_secondary_results = self.cache_manager.load_all_secondary_results()
            all_secondary_results.update(self.secondary_results)
            self.secondary_results = all_secondary_results

        return TimeseriesPipelineOutput(
            secondary_results=self.secondary_results,
            ts_domains={k: sorted(v) for k, v in all_ts_domains.items()},
            ts_domain_pairs={k: sorted(v) for k, v in all_ts_domain_pairs.items()},
        )
