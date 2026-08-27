"""
Purpose
-------
Cache manager -- allows partial reruns and watches which pipelines 
needs to be rerun

Backwards compatibility
----------------
The cache manager does not need to be backward compatible with prev_config
and previous cache, because any change in the cache manager code will 
cause a full rerun and flush the previous cache.

"""

import json
from pathlib import Path
import src.hash_utils as hash_utils
import src.json_exchange as json_exchange
import pickle
import shutil

#: Project root, derived from this file's location rather than the working
#: directory. The watched source paths below are relative to it, and resolving
#: them against the CWD made `python build_input_data.py ...` fail from anywhere
#: except the repo root -- with a bare FileNotFoundError naming a source file,
#: which reads like a broken installation rather than a wrong directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class CacheManager:
    """
    CacheManager handles the saving and loading of critical run information to enable
    partial pipeline execution and smart caching. It manages configuration hashes,
    input data hashes, processor-specific hashes, and the contributions the
    timeseries phase made to the source data tables.

    Attributes:
        cache_folder (Path): Directory where all cache files are stored.
        config_hash_file (Path): Path to store the hash of the config file.
        input_data_hash_file (Path): Path to store hashes of input Excel files.
        processor_hash_file (Path): Path to store hashes of processor modules.
        processor_frames_file (Path): Pickle holding each spec's contributions to
            the source data tables, exactly as its processor returned them.
    """

    # Source code file groups monitored for changes.
    # Changes in these groups trigger the corresponding pipeline phase to re-run.
    # Paths are relative to the project root (where build_input_data.py lives),
    # and are resolved against _REPO_ROOT rather than the working directory --
    # see the module-level constant. compute_file_hash opens the path directly
    # and raises FileNotFoundError, so a CWD-relative lookup meant the whole
    # build died on `./build_input_data.py` when run from anywhere else.
    _OVERALL_CODE_FILES = [
        Path("./build_input_data.py"),
        Path("./src/infrastructure/config_reader.py"),
        Path("./src/infrastructure/cache_manager.py"),
        Path("./src/utils.py"),
        Path("./src/hash_utils.py"),
        Path("./src/json_exchange.py"),
        # Every stage reads Backbone's parameter names and defaults from here,
        # so a change to one of them can move any output.
        Path("./src/backbone_params.py"),
        # And which table declares which dimension, which decides what the build
        # reports about a value nothing declares.
        Path("./src/source_workbook_shape.py"),
    ]
    _SOURCE_PIPELINE_FILES = [
        Path("./src/source_data/source_data_pipeline.py"),
        Path("./src/source_data/source_data_loader.py"),
        Path("./src/source_data/source_data_inputs.py"),
        Path("./src/source_data/source_data_contributions.py"),
    ]
    _TS_PIPELINE_FILES = [
        # Processors are hashed individually by ProcessorRunner, but only the
        # concrete file -- never the base they inherit from. So editing
        # base_processor.py changed every processor's behaviour while the cache
        # reported no change. It carries the declaration defaults now, which
        # makes that blind spot a live one.
        Path("./src/timeseries/processors/base_processor.py"),
        Path("./src/timeseries/timeseries_pipeline.py"),
        Path("./src/timeseries/timeseries_processor.py"),
        Path("./src/timeseries/timeseries_helpers.py"),
        Path("./src/timeseries/timeseries_inputs.py"),
        Path("./src/timeseries/timeseries_results.py"),
        Path("./src/GDX_exchange.py"),
    ]
    _BB_PIPELINE_FILES = [
        Path("./src/bb_excel/bb_excel_inputs.py"),
        Path("./src/bb_excel/bb_excel_pipeline.py"),
    ]

    @property
    def any_timeseries_changed(self) -> bool:
        """Check if any timeseries processor needs to be rerun."""
        return any(self.timeseries_changed.values())

    @property
    def needs_timeseries_run(self) -> bool:
        """
        Determine if timeseries pipeline needs to run.

        Returns True if:
        - Full rerun is requested
        - Any specific processor has changed
        - BB Excel needs to be rebuilt (it reads what the processors contributed)
        """
        return (
            self.full_rerun 
            or self.any_timeseries_changed
            or self.rebuild_bb_excel
        )

    def __init__(self, input_folder: Path, output_folder: Path, config: dict, logger):
        """
        Initialize the CacheManager.

        Args:
            output_folder (Path): The root folder where cache directory will be created.
            logger: IterationLogger instance for status messages.
        """
        self.cache_folder = output_folder / "cache"
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.config_hash_file = self.cache_folder / "config_hash.json"
        self.input_data_hash_file = self.cache_folder / "input_data_hashes.json"
        self.processor_hash_file = self.cache_folder / "processor_hashes.json"
        self.processor_requirements_file = self.cache_folder / "processor_requirements.json"
        self.processor_frames_file = self.cache_folder / "processor_frames.pkl"

        self.input_file_folder = Path(input_folder) / "data_files"
        self.config = config
        self.logger = logger

        # source code related rerun switches
        self.overall_code_files_updated = False
        self.source_data_pipeline_code_updated = False
        self.timeseries_pipeline_code_updated = False
        self.bb_excel_pipeline_code_updated = False

        # config file related rerun switches
        self.demand_files_changed = False
        self.other_input_files_changed = False
        self.timeseries_changed = {}

        # Storing general rerun switches
        self.full_rerun = False
        self.reimport_source_excels = False
        self.rebuild_bb_excel = False



    def _clean_cache_for_full_rerun(self):
        """
        Delete the cache subfolder (output_folder/cache/) and recreate its directory structure.

        This ensures a clean slate before a full rerun and avoids compatibility issues
        with stale hash files or cached results from a previous run.

        Note: only the cache/ subfolder is affected. Root-level output files
        (inputData.xlsx, GAMS files, logs, etc.) are cleaned separately in build_input_data.py.
        """
        if self.cache_folder.exists():
            shutil.rmtree(self.cache_folder, ignore_errors=True)

        self.cache_folder.mkdir(parents=True, exist_ok=True)




    def _check_source_code_changes(self, files: list[Path], cache_name: str) -> bool:
        """
        Check if any of the given source code files have changed since the last run.

        Computes the current hash of each file, compares it to the previously cached values,
        and saves updated hashes regardless. Returns True if any file has changed.

        Args:
            files (list[Path]): Source file paths to monitor for changes.
            cache_name (str): Filename within the cache folder to store the hash record.
        """
        current_hashes = {str(f): hash_utils.compute_file_hash(_REPO_ROOT / f) for f in files}

        hash_store_path = self.cache_folder / cache_name
        previous_hashes = json_exchange.load_json(hash_store_path)

        changed = any(previous_hashes.get(str(f)) != h for f, h in current_hashes.items())

        json_exchange.save_json(hash_store_path, current_hashes)

        return changed


    def _detect_processor_code_changes(self) -> dict:
        """
        Detect which timeseries processors have changed source code since the last run.

        Compares each processor's current file hash against the previously saved hash.
        Does not save updated hashes — processor hashes are saved by timeseries_processor.py
        only after a processor has actually run successfully.

        Returns:
            dict[str, bool]: processor human_name → True if the processor file changed.
        """
        timeseries_specs = self.config["timeseries_specs"]
        if not timeseries_specs:
            return {}

        processor_hashes = self.load_processor_hashes()
        processors_base = _REPO_ROOT / "src" / "timeseries" / "processors"
        result = {}
        changed_processors = []

        for human_name, spec in timeseries_specs.items():
            processor_name = spec.get("processor_name")
            if not processor_name:
                continue

            processor_file = processors_base / f"{processor_name}.py"
            if not processor_file.exists():
                self.logger.log_status(
                    f"Warning: Processor file not found: {processor_file}",
                    level="warn"
                )
                continue

            current_hash = hash_utils.compute_file_hash(processor_file)
            previous_hash = processor_hashes.get(processor_name)
            changed = previous_hash != current_hash

            result[human_name] = changed
            if changed:
                changed_processors.append(human_name)

        if changed_processors and not self.full_rerun:
            self.logger.log_status(
                f"Processor code changes detected: {', '.join(changed_processors)}",
                level="info"
            )

        return result


    def _detect_input_file_changes(self, config: dict, input_folder: Path) -> dict:
        """
        Detect which input Excel file categories have changed since the last run.

        Compares sheet-level hashes against the previous run for each category
        (e.g. 'demanddata_files', 'nodedata_files'). Only sheets matching the
        category's prefix are hashed, following the same logic as read_input_excels.
        Saves updated hashes for the next run regardless of whether changes were found.

        Args:
            config (dict): Parsed config containing file lists per category.
            input_folder (Path): Root folder for all input Excel files in the config.

        Returns:
            dict[str, bool]: category name → True if any sheet in that category changed.
        """
        # Load previous hashes with error handling
        try:
            prev_input_hashes = json_exchange.load_json(self.input_data_hash_file)
        except FileNotFoundError:
            self.logger.log_status("No previous hash file found, treating all files as new.",
                                   level="info")
            prev_input_hashes = {}
        except Exception as e:
            self.logger.log_status(f"Error loading hash file: {e}. Treating all files as changed.",
                                   level="warn")
            prev_input_hashes = {}

        # Map categories to their sheet prefixes (following read_input_excels logic)
        category_to_prefix = {
            "unittypedata_files": "unittype",
            "nodedata_files": "node",
            "emissiondata_files": "emission",
            "demanddata_files": "demand",
            "transferdata_files": "transfer",
            "unitdata_files": "unit",
            "userconstraintdata_files": "userconstraint"
        }

        category_status = {}
        all_hashes_to_save = {}

        for category, sheet_prefix in category_to_prefix.items():
            current_files = config[category]
            current_hashes = {}  # {filename: {sheetname: hash}}

            # Compute sheet-level hashes for each file
            for f in current_files:
                if f == '':
                    self.logger.log_status(f"Empty file name in config category '{category}', check config file.",
                                           level="error")
                    continue

                file_path = input_folder / f

                if not file_path.exists():
                    self.logger.log_status(f"File does not exist: {file_path}", level="error")
                    continue

                try:
                    sheet_hashes = hash_utils.compute_excel_sheets_hash(file_path, sheet_prefix)

                    if not sheet_hashes:
                        self.logger.log_status(
                            f"Did not find '{sheet_prefix}data' sheets from {file_path}",
                            level="warn"
                        )

                    current_hashes[f] = sheet_hashes

                except PermissionError:
                    self.logger.log_status(f"Permission denied reading file: {file_path}",
                                           level="error")
                    continue
                except Exception as e:
                    self.logger.log_status(f"Error computing hash for {file_path}: {e}",
                                           level="error")
                    continue

            prev_hashes = prev_input_hashes.get(category, {})
            changed = self._compare_sheet_hashes(current_hashes, prev_hashes, category)

            category_status[category] = changed
            all_hashes_to_save[category] = current_hashes

            if changed and not self.full_rerun:
                self.logger.log_status(
                    f"Input data changed in category '{category}', rerunning necessary steps.",
                    level="none"
                )

        # Save all current hashes
        try:
            json_exchange.save_json(self.input_data_hash_file, all_hashes_to_save)
        except Exception as e:
            self.logger.log_status(f"Warning: Could not save hash file: {e}", level="warn")

        return category_status


    def _compare_sheet_hashes(self, current: dict, previous: dict, category: str) -> bool:
        """
        Compare current and previous sheet-level hashes to detect changes.

        Args:
            current: {filename: {sheetname: hash}}
            previous: {filename: {sheetname: hash}}
            category: Category name for logging

        Returns:
            bool: True if any sheet changed, was added, or was removed
        """
        # Check if file lists differ
        if set(current.keys()) != set(previous.keys()):
            return True

        # Check each file's sheets
        for filename, curr_sheets in current.items():
            prev_sheets = previous.get(filename, {})

            # Check if sheet lists differ
            if set(curr_sheets.keys()) != set(prev_sheets.keys()):
                self.logger.log_status(
                    f"Sheet structure changed in '{filename}' for category '{category}'",
                    level="info"
                )
                return True

            # Check if any sheet content changed
            for sheet_name, curr_hash in curr_sheets.items():
                prev_hash = prev_sheets.get(sheet_name)
                if curr_hash != prev_hash:
                    self.logger.log_status(
                        f"Sheet '{sheet_name}' changed in '{filename}' for category '{category}'",
                        level="info"
                    )
                    return True

        return False


    def _detect_timeseries_spec_changes(self, config: dict, prev_config: dict,
                                        input_changes: dict | None = None) -> dict:
        """
        Detect which timeseries processors need to be rerun based on spec changes.

        Compares each processor's current spec against the previously cached spec,
        then adds the processors whose *source data* changed. Two ways a processor
        depends on source data:

        - ``demand_grid`` in its spec, which asks for df_demanddata;
        - ``requires_source_data`` on its class, recorded to the cache by
          ProcessorRunner on the previous run.

        The second is read from cache rather than from the class because this
        object would otherwise have to import every processor module to answer the
        question. A processor with no recorded requirement is treated
        conservatively -- changed if any input file changed -- which covers a first
        run, a cleared cache, and a processor that has never completed.

        Should only be called when prev_config exists and full_rerun is False — the caller
        is responsible for marking all processors True on a full rerun before calling this.

        Args:
            config: current config.
            prev_config: config structure cached on the previous run.
            input_changes: category name ('nodedata_files', ...) → whether it changed.

        Returns:
            dict[str, bool]: processor human_name → True if that processor needs to rerun.
        """
        curr_specs = config["timeseries_specs"]
        prev_specs = prev_config["timeseries_specs"]
        input_changes = input_changes or {}
        any_input_changed = any(input_changes.values())
        recorded_requirements = self.load_processor_requirements()
        result = {}

        for key, curr_spec in curr_specs.items():
            # Normalize curr_spec through a JSON round-trip so types match prev_specs,
            # which was loaded from JSON.
            curr_spec_normalized = json.loads(json.dumps(curr_spec))
            changed = (key not in prev_specs) or (prev_specs[key] != curr_spec_normalized)

            if input_changes.get("demanddata_files") and curr_spec.get("demand_grid"):
                changed = True

            processor_name = curr_spec.get("processor_name")
            if processor_name in recorded_requirements:
                for source_name in recorded_requirements[processor_name]:
                    if input_changes.get(f"{source_name}_files"):
                        changed = True
            elif any_input_changed:
                # Nothing recorded for this processor, so its requirements are
                # unknown rather than empty. Rerunning is the cheap mistake.
                changed = True

            result[key] = changed

        return result


    def _save_dict_to_cache(self, data: dict, filename: str):
        """
        Save a dictionary into a cache folder at the given filename.

        Args:
            dict (dict): The data to cache.
            filename (str):  name of the JSON file.
        """
        file_path = Path(self.cache_folder) / filename

        # convert any top-level sets into lists
        clean_data = {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in data.items()
        }

        json_exchange.save_json(file_path, clean_data)


    def load_dict_from_cache(self, filename: str):
        """
        Load a dictionary from the cache folder under the given filename.

        Args:
            filename (str): Name of the JSON file (e.g. "my_cache.json").

        Returns:
            dict: The parsed contents, or {} if the file is missing or unreadable.
        """
        file_path = Path(self.cache_folder) / filename
        try:
            return json_exchange.load_json(file_path)
        except Exception:
            return {}


    def merge_dict_to_cache(self, data: dict, filename: str):
        """
        Merge new data into the existing cache entry (if any), then save the result.

        Key by key: a set-valued key unions, anything else takes the new value.
        The only caller left is ``general_flags.json``, whose values are flags.

        Args:
            data (dict): New data to merge.
            filename (str): Name of the JSON cache file.
        """
        try:
            merged = self.load_dict_from_cache(filename)
        except (FileNotFoundError, json.JSONDecodeError):
            merged = {}

        for key, new_val in data.items():
            old_val = merged.get(key)
            if isinstance(old_val, set) and isinstance(new_val, set):
                merged[key] = old_val.union(new_val)
            else:
                merged[key] = new_val

        self._save_dict_to_cache(merged, filename)


    def save_processor_hash(self, processor_name: str, hash_value: str):
        """
        Save the hash of a specific processor.

        Args:
            processor_name (str): Name of the processor.
            hash_value (str): Hash of the processor file.
        """
        hashes = json_exchange.load_json(self.processor_hash_file)
        hashes[processor_name] = hash_value
        json_exchange.save_json(self.processor_hash_file, hashes)


    def load_processor_hashes(self):
        """
        Load previously saved processor hashes.

        Returns:
            dict: Processor names mapped to their hashes.
        """
        return json_exchange.load_json(self.processor_hash_file)


    def save_processor_requirements(self, processor_name: str, source_names):
        """
        Save which source-data frames a processor declared it needs.

        Recorded by ProcessorRunner from the processor class's
        ``requires_source_data``, and read back by
        ``_detect_timeseries_spec_changes`` so a change to one of those source
        workbooks reruns the processor. CacheManager cannot read the declaration
        itself without importing processor modules, which it has no business
        doing.

        Args:
            processor_name (str): Name of the processor.
            source_names: Iterable of source-data names, without the 'df_' prefix.
        """
        requirements = json_exchange.load_json(self.processor_requirements_file)
        requirements[processor_name] = sorted(source_names)
        json_exchange.save_json(self.processor_requirements_file, requirements)


    def load_processor_requirements(self) -> dict:
        """
        Load the source-data requirements recorded on the previous run.

        Returns:
            dict: Processor names mapped to a list of source-data names.
        """
        return json_exchange.load_json(self.processor_requirements_file)


    def load_processor_frames(self) -> dict:
        """
        Every spec's contributions to the source data tables, as last produced.

        Returns:
            dict: ``{human_name: {table name: DataFrame}}``. Empty if nothing has
            been cached yet.
        """
        if not self.processor_frames_file.exists():
            return {}
        with open(self.processor_frames_file, "rb") as f:
            return pickle.load(f)


    def save_processor_frames(self, human_name: str, frames: dict):
        """
        Record one spec's contributions, replacing whatever it said last time.

        Keyed by the spec's ``timeseries_specs`` name rather than by the
        processor's: three specs share ``VRE_PECD``, and a file named after the
        processor had them overwrite each other's answers.

        **What goes in is what the processor returned.** Never a merged or melted
        table: the merge into the source frames is recomputed from workbooks plus
        this file on every build, so a partial rerun -- where most specs did not
        run at all -- reads exactly the same as a full one. A cache holding
        half-merged tables could not offer that.

        Args:
            human_name (str): The spec's key in timeseries_specs.
            frames (dict): ``{table name: DataFrame}``, already validated.
        """
        stored = self.load_processor_frames()
        stored[human_name] = frames
        with open(self.processor_frames_file, "wb") as f:
            pickle.dump(stored, f)



    def _save_all_source_code_hashes(self):
        """
        Compute and save hashes for all source code file groups that can trigger a full rerun.

        Called in two situations:
        - During Phase 1 detection: to record current hashes and detect changes vs. previous run.
        - After a cache clear in Phase 2: to write fresh hashes into the newly-cleared cache,
          so the next run starts from a correct baseline.

        The BB pipeline files are excluded here because they never trigger a full rerun —
        they are checked separately in Phase 3.
        """
        self.overall_code_files_updated = self._check_source_code_changes(
            self._OVERALL_CODE_FILES, "overall_code_files_hashes.json"
        )
        self.source_data_pipeline_code_updated = self._check_source_code_changes(
            self._SOURCE_PIPELINE_FILES, "source_data_pipeline_hashes.json"
        )
        self.timeseries_pipeline_code_updated = self._check_source_code_changes(
            self._TS_PIPELINE_FILES, "timeseries_pipeline_hashes.json"
        )


    def run(self) -> None:
        """
        Determine what needs to be rerun based on changes since last execution.

        Flow:
        1. Check for full rerun causes (using existing cache to detect changes)
        2. If full rerun: clean cache, then set full rerun flags
        3. Always run remaining checks and regenerate caches for next run
        """

        # ========================================================================
        # PHASE 1: CHECK FOR FULL RERUN CAUSES (in priority order)
        # ========================================================================
        # We check using existing cache before updating, so we can detect changes

        full_rerun_reason = None  # Track why we're doing a full rerun

        # 1) Check config-based full rerun causes
        prev_config = json_exchange.load_json(self.cache_folder / "config_structural.json")

        # If previous config was never saved
        if not prev_config:
            full_rerun_reason = ("This is the first run or the config file cache has been removed. "
                                 "Starting a new run.")

        # User-requested full rerun
        if self.config['force_full_rerun']:
            full_rerun_reason = "User has requested a full rerun."

        # Topology changes
        if not full_rerun_reason:
            keys = ["country_codes", "exclude_grids", "exclude_nodes"]
            if any(prev_config[k] != self.config[k] for k in keys):
                full_rerun_reason = ("Config file topology, e.g. included countries, have changed. "
                                     "Starting a full rerun.")

        # Climate years or timeseries window changed
        if not full_rerun_reason:
            climate_keys = ("climate_data", "bb_timeseries_start", "bb_timeseries_length")
            changed = [k for k in climate_keys if prev_config.get(k) != self.config.get(k)]
            if changed:
                full_rerun_reason = f"Climate/timeseries config changed ({', '.join(changed)}), starting a full rerun."

        # Forecast structure changed (requires full rerun to recopy patched GAMS files and rerun timeseries)
        if not full_rerun_reason:
            forecast_keys = ("forecast_quantiles", "forecast_weights")
            changed = [k for k in forecast_keys if prev_config.get(k) != self.config.get(k)]
            if changed:
                full_rerun_reason = f"Forecast config changed ({', '.join(changed)}), starting a full rerun."

        # 2) Check source code based full reruns.
        # _save_all_source_code_hashes() checks and updates hashes for all three groups.
        # The guard stops at the first change found, but Phase 2 always regenerates
        # all three after the cache clear so they are correct for the next run.
        if not full_rerun_reason:
            self._save_all_source_code_hashes()
            if self.overall_code_files_updated:
                full_rerun_reason = ("Certain code files that orchestrate the overall workflow have been updated, "
                                     "starting a full rerun.")
            elif self.source_data_pipeline_code_updated:
                full_rerun_reason = "Source excel data pipeline code updated, starting a full rerun."
            elif self.timeseries_pipeline_code_updated:
                full_rerun_reason = "Timeseries pipeline code updated, starting a full rerun."

        # 3) Check if source excel or timeseries phases had errors in the previous run.
        # BB excel failures are handled separately via rebuild_bb_excel (Phase 3),
        # so they do not trigger a full rerun here.
        if not full_rerun_reason:
            general_flags = self.load_dict_from_cache("general_flags.json")
            source_excel_run_successfully = general_flags.get("source_excel_run_successfully", False)
            timeseries_run_successfully = general_flags.get("timeseries_run_successfully", False)
            if not source_excel_run_successfully:
                full_rerun_reason = "Source excel phase did not complete successfully in previous run. Starting a full rerun."
            elif not timeseries_run_successfully:
                full_rerun_reason = "Timeseries phase did not complete successfully in previous run. Starting a full rerun."


        # ========================================================================
        # PHASE 2: HANDLE FULL RERUN
        # ========================================================================

        if full_rerun_reason:
            self.full_rerun = True
            self.logger.log_status(full_rerun_reason, level="run", add_empty_line_before=True)

            # Clean the cache subfolder to ensure a fresh start
            self._clean_cache_for_full_rerun()
            self.logger.log_status("Cleared cache subfolder (output_folder/cache/).", level="info")

            # Mark all timeseries for rerun
            for key in self.config["timeseries_specs"].keys():
                self.timeseries_changed[key] = True

            # Regenerate source code hashes into the freshly-cleared cache.
            # Phase 1 may have short-circuited before checking all groups, or may
            # have checked against hashes that no longer exist after the clear.
            self._save_all_source_code_hashes()


        # ========================================================================
        # PHASE 3: RUN REMAINING CHECKS
        # Run always, for both full rerun and granular
        # These are fast and ensure all caches are up to date for next run
        # ========================================================================

        self.logger.log_status("Updating cache content", level="none")

        # Detect input file changes and update hashes for next run
        input_changes = self._detect_input_file_changes(self.config, self.input_file_folder)
        self.demand_files_changed = input_changes.get("demanddata_files", False)
        self.other_input_files_changed = any(
            v for k, v in input_changes.items() if k != "demanddata_files"
        )

        # Detect timeseries spec changes (granular only — full rerun already set all True in Phase 2)
        if prev_config and not self.full_rerun:
            ts_spec_changes = self._detect_timeseries_spec_changes(
                self.config, prev_config, input_changes
            )
            for key, changed in ts_spec_changes.items():
                self.timeseries_changed[key] = self.timeseries_changed.get(key, False) or changed

        # Detect processor code changes and merge into timeseries_changed
        proc_changes = self._detect_processor_code_changes()
        for human_name, changed in proc_changes.items():
            self.timeseries_changed[human_name] = self.timeseries_changed.get(human_name, False) or changed

        # Load flags for granular checks
        general_flags = self.load_dict_from_cache("general_flags.json")
        bb_excel_succesfully_built = general_flags.get("bb_excel_succesfully_built", False)

        # Check BB excel pipeline code — does not trigger a full rerun, only a bb excel rebuild
        self.bb_excel_pipeline_code_updated = self._check_source_code_changes(
            self._BB_PIPELINE_FILES, "bb_excel_pipeline_hashes.json"
        )
        if self.bb_excel_pipeline_code_updated and not self.full_rerun:
            self.logger.log_status("BB input excel pipeline code updated, generating new input excel for Backbone.",
                                   level="none")

        # Determine if BB input excel needs to be rebuilt.
        #
        # any_timeseries_changed belongs here because processor output reaches the
        # workbook, not just the GDX: a processor's contribution to the source
        # data tables drives create_p_gn, create_p_gnBoundaryPropertiesForStates
        # and add_storage_starts. Individual
        # processor files are hashed by ProcessorRunner and deliberately kept out of
        # _TS_PIPELINE_FILES, so editing one used to rerun the timeseries phase and
        # leave inputData.xlsx describing the previous run's series -- stale against
        # its own GDX, with nothing saying so.
        self.rebuild_bb_excel = (
            self.full_rerun
            or self.demand_files_changed
            or self.other_input_files_changed
            or self.bb_excel_pipeline_code_updated
            or not bb_excel_succesfully_built
            or self.any_timeseries_changed
        )

        # Determine if source excels should be re-imported.
        #
        # rebuild_bb_excel comes first because BBExcelPipeline reads every source
        # frame there is -- nodedata, unitdata, demanddata, transferdata,
        # emissiondata, userconstraintdata. SourceDataPipeline holds those as empty
        # DataFrames until run() fills them, so a rebuild against a skipped import
        # does not degrade, it dies on the first column it looks for. The two flags
        # once shared every term but rebuild_bb_excel, so nothing forced the import
        # for a processor that reads neither demanddata nor a declared workbook:
        # editing VRE_PECD alone rebuilt the workbook from nothing.
        #
        # The second clause is what stops a *processor* running against a
        # SourceDataPipeline whose run() was skipped: one that declares
        # requires_source_data would otherwise receive empty frames and refuse,
        # producing no GDX for a reason the user cannot see. It answers a different
        # question than the first and keeps its own name, even though a changed
        # processor implies a rebuild today.
        recorded_requirements = self.load_processor_requirements()
        any_source_consuming_processor_changed = any(
            self.timeseries_changed.get(human_name, False)
            and (spec.get("demand_grid")
                 or recorded_requirements.get(spec.get("processor_name"), []))
            for human_name, spec in self.config["timeseries_specs"].items()
        )
        self.reimport_source_excels = (
            self.rebuild_bb_excel
            or any_source_consuming_processor_changed
        )


        # ========================================================================
        # PHASE 4: FINALIZATION
        # ========================================================================

        # Save current config structure for next run
        relevant_keys = [
            "country_codes", "exclude_grids", "exclude_nodes",
            "climate_data", "bb_timeseries_start", "bb_timeseries_length",
            "forecast_quantiles", "forecast_weights", "timeseries_specs"
        ]
        data = {k: self.config[k] for k in relevant_keys if k in self.config}
        json_exchange.save_json(self.cache_folder / "config_structural.json", data)

        # Reset workflow_run_successfully flag
        # This will be set to True at the very end of the workflow if successful
        status_dict = {"workflow_run_successfully": False}
        self.merge_dict_to_cache(status_dict, "general_flags.json")