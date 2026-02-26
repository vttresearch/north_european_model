# src/cache_manager.py

import json
from pathlib import Path
import src.hash_utils as hash_utils
import src.json_exchange as json_exchange
import pickle
from datetime import datetime
import shutil

class CacheManager:
    """
    CacheManager handles the saving and loading of critical run information to enable
    partial pipeline execution and smart caching. It manages configuration hashes,
    input data hashes, processor-specific hashes, and secondary results generated during
    timeseries processing.

    Attributes:
        cache_folder (Path): Directory where all cache files are stored.
        config_hash_file (Path): Path to store the hash of the config file.
        input_data_hash_file (Path): Path to store hashes of input Excel files.
        processor_hash_file (Path): Path to store hashes of processor modules.
        secondary_results_folder (Path): Directory where secondary results are stored per processor.
    """

    # Source code file groups monitored for changes.
    # Changes in these groups trigger the corresponding pipeline phase to re-run.
    # Paths are relative to the project root (where build_input_data.py lives).
    _OVERALL_CODE_FILES = [
        Path("./build_input_data.py"),
        Path("./src/config_reader.py"),
        Path("./src/pipeline/cache_manager.py"),
        Path("./src/utils.py"),
        Path("./src/hash_utils.py"),
        Path("./src/json_exchange.py"),
    ]
    _SOURCE_PIPELINE_FILES = [
        Path("./src/pipeline/source_excel_data_pipeline.py"),
        Path("./src/data_loader.py"),
    ]
    _TS_PIPELINE_FILES = [
        Path("./src/pipeline/timeseries_pipeline.py"),
        Path("./src/pipeline/timeseries_processor.py"),
        Path("./src/GDX_exchange.py"),
    ]
    _BB_PIPELINE_FILES = [
        Path("./src/pipeline/bb_excel_context.py"),
        Path("./src/build_input_excel.py"),
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
        - BB Excel needs to be rebuilt (requires ts_domains data)
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
        self.secondary_results_folder = self.cache_folder / "secondary_results"
        self.secondary_results_folder.mkdir(exist_ok=True)

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
        self.secondary_results_folder.mkdir(parents=True, exist_ok=True)
  

    def _date_range_expanded(self, config: dict, prev: dict) -> bool:
        """
        Return True if the requested date range has expanded compared to the previous run.

        An expansion occurs when:
        - The new start_date is earlier than the previous start_date, OR
        - The new end_date is later than the previous end_date
        """
        def parse(dt):
            return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

        old_start = parse(prev["start_date"])
        old_end = parse(prev["end_date"])
        new_start = parse(config["start_date"])
        new_end = parse(config["end_date"])

        return new_start < old_start or new_end > old_end


    def _check_source_code_changes(self, files: list[Path], cache_name: str) -> bool:
        """
        Check if any of the given source code files have changed since the last run.

        Computes the current hash of each file, compares it to the previously cached values,
        and saves updated hashes regardless. Returns True if any file has changed.

        Args:
            files (list[Path]): Source file paths to monitor for changes.
            cache_name (str): Filename within the cache folder to store the hash record.
        """
        current_hashes = {str(f): hash_utils.compute_file_hash(f) for f in files}

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
        processors_base = Path("src/processors")
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
        (e.g. 'demanddata_files', 'fueldata_files'). Only sheets matching the
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
            "fueldata_files": "fuel",
            "emissiondata_files": "emission",
            "demanddata_files": "demand",
            "transferdata_files": "transfer",
            "unitdata_files": "unit",
            "storagedata_files": "storage",
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
                                        demand_files_changed: bool = False) -> dict:
        """
        Detect which timeseries processors need to be rerun based on spec changes.

        Compares each processor's current spec against the previously cached spec.
        If `demand_files_changed` is True, all processors that reference 'demand_grid'
        are also marked as changed.

        Should only be called when prev_config exists and full_rerun is False — the caller
        is responsible for marking all processors True on a full rerun before calling this.

        Returns:
            dict[str, bool]: processor human_name → True if that processor needs to rerun.
        """
        curr_specs = config["timeseries_specs"]
        prev_specs = prev_config["timeseries_specs"]
        result = {}

        for key, curr_spec in curr_specs.items():
            changed = (key not in prev_specs) or (prev_specs[key] != curr_spec)

            if demand_files_changed and "demand_grid" in curr_spec:
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
        Load a dictionary from the cache folder under the given filename,
        reconstructing any sets or list-of-tuples that were serialized as JSON arrays.
    
        Args:
            filename (str): Name of the JSON file (e.g. "my_cache.json").
    
        Returns:
            dict: The data with outer lists converted back into sets or list-of-tuples.
        """
        file_path = Path(self.cache_folder) / filename
        try:
            raw = json_exchange.load_json(file_path)  # returns a dict with JSON types
        except:
            return {}
    
        reconstructed = {}
        for key, val in raw.items():
            if isinstance(val, list):
                # case 1: list-of-2-lists → list-of-tuples
                if all(isinstance(item, list) and len(item) == 2 for item in val):
                    reconstructed[key] = [tuple(item) for item in val]
                else:
                    # assume list of scalars → set
                    reconstructed[key] = set(val)
            else:
                # leave anything else untouched
                reconstructed[key] = val
    
        return reconstructed


    def merge_dict_to_cache(self, data: dict, filename: str):
        """
        Merge new data into the existing cache entry (if any), then save the result.

        - For set-valued keys: union old and new.
        - For list-of-2-tuples keys: append any new tuples (preserving order).
        - For any other types or brand-new keys: overwrite/take new.

        Args:
            data (dict): New data to merge (values as sets or list-of-2-tuples).
            filename (str): Name of the JSON cache file.
        """
        # 1) Load existing (or get empty dict if none)
        try:
            merged = self.load_dict_from_cache(filename)
        except (FileNotFoundError, json.JSONDecodeError):      
            merged = {}

        # 2) Merge in new values
        for key, new_val in data.items():
            old_val = merged.get(key)
            if old_val is None:
                # brand new key
                merged[key] = new_val
            elif isinstance(old_val, set) and isinstance(new_val, set):
                merged[key] = old_val.union(new_val)
            elif (
                isinstance(old_val, list)
                and isinstance(new_val, list)
                and all(isinstance(t, tuple) and len(t) == 2 for t in old_val)
                and all(isinstance(t, tuple) and len(t) == 2 for t in new_val)
            ):
                # list-of-2-tuples: append uniques
                combined = list(old_val)
                for tup in new_val:
                    if tup not in combined:
                        combined.append(tup)
                merged[key] = combined
            else:
                # fallback: overwrite
                merged[key] = new_val

        # 3) Save back to cache
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


    def save_secondary_result(self, processor_name: str, data, secondary_result_name: str):
        """
        Save a secondary result under a named key inside a processor's result file.
    
        Args:
            processor_name (str): Name of the processor.
            data: Data to serialize and save.
            secondary_result_name (str): Key name for the secondary result.
        """
        path = self.secondary_results_folder / f"{processor_name}.pkl"

        # Save under the named secondary result
        processor_results = {}
        processor_results[secondary_result_name] = data
    
        with open(path, "wb") as f:
            pickle.dump(processor_results, f)


    def load_all_secondary_results(self) -> dict:
        """
        Load all secondary results from cache and flatten them into {secondary_result_name: result}.

        Returns:
            dict: {secondary_result_name: result}
        """
        secondary_results = {}
        if not self.secondary_results_folder.exists():
            return secondary_results

        for pkl_file in self.secondary_results_folder.glob("*.pkl"):
            with open(pkl_file, "rb") as f:
                processor_results = pickle.load(f)

            # Flatten: {secondary_result_name: result}
            for secondary_result_name, result in processor_results.items():
                secondary_results[secondary_result_name] = result

        return secondary_results
    

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

        # Date range expansion
        if not full_rerun_reason:
            if self._date_range_expanded(self.config, prev_config):
                full_rerun_reason = "Requested time range has expanded from previous run, starting a full rerun."

        # Check if csv writing was previously disabled, but is now enabled
        if not full_rerun_reason:
            prev_status = prev_config["write_csv_files"]
            curr_status = self.config["write_csv_files"]
            if curr_status and not prev_status:
                full_rerun_reason = "Config file now wants to print csv files, starting a full rerun."

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

        # 3) Check if previous workflow didn't complete successfully
        if not full_rerun_reason:
            general_flags = self.load_dict_from_cache("general_flags.json")
            workflow_run_successfully = general_flags.get("workflow_run_successfully", False)
            if not workflow_run_successfully:
                full_rerun_reason = "Previous workflow did not complete successfully. Starting a full rerun."


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
                self.config, prev_config, self.demand_files_changed
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

        # Determine if source excels should be re-imported.
        # Triggered by: full rerun, any input file change, or BB excel pipeline code change.
        # Pure timeseries spec/processor-code changes do NOT require re-importing source excels.
        self.reimport_source_excels = (
            self.full_rerun
            or self.demand_files_changed
            or self.other_input_files_changed
            or self.bb_excel_pipeline_code_updated
            or not bb_excel_succesfully_built
        )

        # Determine if BB input excel needs to be rebuilt
        self.rebuild_bb_excel = (
            self.full_rerun
            or self.demand_files_changed
            or self.other_input_files_changed
            or self.bb_excel_pipeline_code_updated
            or not bb_excel_succesfully_built
        )


        # ========================================================================
        # PHASE 4: FINALIZATION
        # ========================================================================

        # Save current config structure for next run
        relevant_keys = [
            "country_codes", "exclude_grids", "exclude_nodes",
            "start_date", "end_date", "write_csv_files",
            "timeseries_specs"
        ]
        data = {k: self.config[k] for k in relevant_keys if k in self.config}
        json_exchange.save_json(self.cache_folder / "config_structural.json", data)

        # Reset workflow_run_successfully flag
        # This will be set to True at the very end of the workflow if successful
        status_dict = {"workflow_run_successfully": False}
        self.merge_dict_to_cache(status_dict, "general_flags.json")