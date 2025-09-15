import json
from pathlib import Path
from src.hash_utils import compute_file_hash
from src.json_exchange import load_json, save_json
from src.utils import log_status
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

    def __init__(self, input_folder: Path, output_folder: Path, config: dict):
        """
        Initialize the CacheManager.

        Args:
            output_folder (Path): The root folder where cache directory will be created.
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

        # source code related rerun switches
        self.source_data_pipeline_code_updated = False
        self.timeseries_pipeline_code_updated = False
        self.bb_excel_pipeline_code_updated = False

        # config file related rerun switches
        self.topology_changed = False
        self.date_range_expanded = False
        self.csv_writer_requested = False
        self.demand_files_changed = False
        self.other_input_files_changed = False
        
        self.timeseries_changed = {}

        # Storting general rerun switches
        self.reimport_source_excels = False
        self.rerun_all_ts = False
        self.rebuild_bb_excel = False


    def run(self) -> list[str]:
        validation_log = []

        # --- Config file validation ---
        # Checking overall topology in config file
        log = self._validate_topology(self.config)
        validation_log.extend(log)

        log = self._validate_start_and_end(self.config)
        validation_log.extend(log)

        log = self._validate_csv_writer(self.config)
        validation_log.extend(log)
        
        # Checking input files in config file
        log = self._validate_input_files(self.config, self.input_file_folder)
        validation_log.extend(log)
      
        # Save current config
        self.save_structural_config(self.config)


        # --- Source code validation ---
        # Check changes in source excel data pipeline code files
        files = [
            Path("./src/pipeline/source_excel_data_pipeline.py"),
            Path("./src/data_loader.py")
        ]
        cache_name = "source_data_pipeline_hashes.json"
        self.source_data_pipeline_code_updated = self._validate_source_code_changes(files, cache_name)
        if self.source_data_pipeline_code_updated and not self.topology_changed:
            log_status('Source excel data pipeline code updated, rerunning all timeseries and generating new input excel for Backbone.', validation_log, level="run")

        # Check timeseries pipeline code files
        files = [
            Path("./src/pipeline/timeseries_pipeline.py"),
            Path("./src/pipeline/timeseries_processor.py"),
            Path("./src/GDX_exchange.py")
        ]
        cache_name = "timeseries_pipeline_hashes.json"
        self.timeseries_pipeline_code_updated = self._validate_source_code_changes(files, cache_name)
        if self.timeseries_pipeline_code_updated and not self.topology_changed:
            log_status('Timeseries pipeline code updated, rerunning all timeseries and generating new input excel for Backbone.', validation_log, level="run")

        # Check BB input excel pipeline code files
        files = [
            Path("./src/pipeline/bb_excel_context.py"),
            Path("./src/build_input_excel.py"),
            Path("./src/excel_exchange.py")
        ]
        cache_name = "bb_excel_pipeline_hashes.json"
        self.bb_excel_pipeline_code_updated = self._validate_source_code_changes(files, cache_name)
        if self.bb_excel_pipeline_code_updated and not self.topology_changed:
            log_status('BB input excel pipeline code updated, generating new input excel for Backbone.', validation_log, level="run")
        

        # --- General flags validation ---
        general_flags = self.load_dict_from_cache("general_flags.json")
        workflow_run_successfully = general_flags.get("workflow_run_successfully", False)
        bb_excel_succesfully_built = general_flags.get("bb_excel_succesfully_built", False)


        # --- Deciding what sections to rerun ---
        # checking if all timeseries need rerunning
        self.full_rerun = (self.config.get('force_full_rerun')
                           or self.topology_changed  
                           or self.date_range_expanded 
                           or self.csv_writer_requested
                           or self.source_data_pipeline_code_updated
                           or self.timeseries_pipeline_code_updated
                           or not workflow_run_successfully
                           )
        # checking if specific timeseries need rerunning
        log = self._validate_timeseries(self.config, self.full_rerun, self.demand_files_changed)
        validation_log.extend(log)

        # Checking if source excels should be re-imported
        self.reimport_source_excels = (self.full_rerun
                                       or self.demand_files_changed
                                       or self.other_input_files_changed
                                       or self.bb_excel_pipeline_code_updated
                                       or not bb_excel_succesfully_built
        )        

        # Checking if BB input excel needs to be rebuilt
        self.rebuild_bb_excel = self.reimport_source_excels


        # --- postprocessing ---
        # If full rerun, clear certain data from cache
        if self.full_rerun:    
            # General flags
            p = Path(self.cache_folder) / "general_flags.json"
            p.unlink(missing_ok=True)               
            # all_ts_domains
            p = Path(self.cache_folder) / "all_ts_domains.json"
            p.unlink(missing_ok=True)
            # all_ts_domain_pairs
            p = Path(self.cache_folder) / "all_ts_domain_pairs.json"
            p.unlink(missing_ok=True)
            # Secondary results
            shutil.rmtree(self.secondary_results_folder, ignore_errors=True)
            self.secondary_results_folder.mkdir(parents=True, exist_ok=True)            

        return validation_log


    def _validate_topology(self, config: dict) -> list[str]:
        prev = self.load_structural_config()

        # If previous config was never saved, treat it as a change
        if not prev:
            self.topology_changed = True
        else:
            keys = ["country_codes", "exclude_grids", "exclude_nodes"]
            self.topology_changed = any(prev.get(k) != config.get(k) for k in keys)
    
        # Printing to log
        validation_log = []
        if self.topology_changed:
            log_status('Config file topology, e.g. included countries, have changed or this is the first run. Starting a full rerun.', validation_log, level="run")

        return validation_log
    

    def _validate_start_and_end(self, config: dict) -> list[str]:
        prev = self.load_structural_config()

        # If previous config was never saved, treat it as a change
        if not prev:
            self.date_range_expanded = True

        def parse(dt): return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

        old_start = parse(prev.get("start_date", config["start_date"]))
        old_end = parse(prev.get("end_date", config["end_date"]))
        new_start = parse(config["start_date"])
        new_end = parse(config["end_date"])

        self.date_range_expanded = new_start != old_start or new_end != old_end

        # Printing to log
        validation_log = []
        if self.date_range_expanded:
            log_status('Requested time range has expanded from previous run, rerunning all timeseries.', validation_log, level="run")

        return validation_log


    def _validate_csv_writer(self, config: dict) -> list[str]:
        prev = self.load_structural_config()

        # If previous config was never saved, treat it as a change
        if not prev:
            return []

        self.csv_writer_requested = not prev.get("write_csv_files", False) and config.get("write_csv_files", False)

        # Printing to log
        validation_log = []
        if self.csv_writer_requested and not self.topology_changed:
            log_status('Config file now wants to print csv files, rerunning all timeseries.', validation_log, level="run")

        return validation_log


    def _validate_source_code_changes(self, files: list[Path], cache_name: str) -> bool:
        """
        Check if any of the given source code files have changed since the last run.

        This method computes the current hash of each specified file, compares it to the
        previously cached hash values (stored under a specified filename), and returns
        True if any differences are found. It updates the cached hashes regardless.

        Args:
            files (list[Path]): List of source file paths to monitor for changes.
            cache_name (str): Filename (within cache folder) to store the file hash record.

        Returns:
            bool: True if any file has changed since the last check, otherwise False.
        """
        # Compute current hashes for the specified files
        current_hashes = {str(f): compute_file_hash(f) for f in files}

        # Load previously stored hashes from cache
        hash_store_path = self.cache_folder / cache_name
        previous_hashes = load_json(hash_store_path)

        # Determine if any file has changed by comparing hashes
        changed = any(previous_hashes.get(str(f)) != h for f, h in current_hashes.items())

        # Update cache with current hashes
        save_json(hash_store_path, current_hashes)

        return changed


    def _validate_input_files(self, config: dict, input_folder: Path) -> list[str]:
        """
        Checks if input excel files have changed by comparing the file lists in config file and 
        with stored hashes of the included files.
    
        Args:
            config (dict): Parsed configuration dictionary.
            input_folder (Path): Folder containing all input excel files.
    
        Returns:
            dictionary: {'changed_status': {category: boolean}, 'log': str}
        """
        prev_input_hashes = load_json(self.input_data_hash_file)
        validation_log = []
        category_status = {}

        all_hashes_to_save = {}

        for category in [
            "unittypedata_files", "fueldata_files", "emissiondata_files",
            "demanddata_files", "transferdata_files", 
            "unitdata_files", "storagedata_files"
        ]:
            current_files = config.get(category, [])
            current_hashes = {f: compute_file_hash(input_folder / f) for f in current_files}
            prev_hashes = prev_input_hashes.get(category, {})

            changed = (
                current_hashes != prev_hashes or 
                set(current_files) != set(prev_hashes.keys())
            )

            category_status[category] = changed
            all_hashes_to_save[category] = current_hashes

            if changed and not self.topology_changed:
                log_status(f"Input data files changed in category '{category}', rerunning necessary steps.", validation_log, level="run")

        # Save all current hashes
        save_json(self.input_data_hash_file, all_hashes_to_save)

        # Check flags used in the main logic
        self.demand_files_changed = category_status['demanddata_files']
        self.other_input_files_changed = any(
            changed for key, changed in category_status.items() if key != 'demanddata_files'
        )

        # Return logs
        return validation_log


    def _validate_timeseries(self, config: dict, rerun_all_ts: bool = False, demand_files_changed: bool = False) -> list[str]:
        """
        Compare current timeseries specs to previously cached ones and detect changes.

        If `rerun_all_ts` is True, all processors are rerun.
        If `demand_files_changed` is True, all processors with 'demand_grid' are rerun.

        Returns:
            dict[str, bool]: Keys are human_name from timeseries_specs, values True if processor should rerun.
        """
        prev = self.load_structural_config()
        curr_specs = config.get("timeseries_specs", {})
        prev_specs = prev.get("timeseries_specs", {}) if prev else {}

        for key, curr_spec in curr_specs.items():
            # Default to rerun if key not in previous spec
            changed = key not in prev_specs or prev_specs[key] != curr_spec

            # Broader triggers
            if rerun_all_ts:
                changed = True
            elif demand_files_changed and "demand_grid" in curr_spec:
                changed = True

            self.timeseries_changed[key] = changed

        # Printing to log
        validation_log = []
        changed_keys = [k for k, v in self.timeseries_changed.items() if v]
        if changed_keys and not self.topology_changed:
            log_status(
                f'Rerunning following timeseries: {", ".join(changed_keys)}',
                validation_log,
                level="info"
            )

        return validation_log


    def load_structural_config(self) -> dict:
        return load_json(self.cache_folder / "config_structural.json")

    def save_structural_config(self, config: dict):
        relevant_keys = [
            "country_codes", "exclude_grids", "exclude_nodes",
            "start_date", "end_date", "write_csv_files", "timeseries_specs"
        ]
        data = {k: config[k] for k in relevant_keys if k in config}
        save_json(self.cache_folder / "config_structural.json", data)

    def save_dict_to_cache(self, data: dict, filename: str):
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

        save_json(file_path, clean_data)


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
            raw = load_json(file_path)  # returns a dict with JSON types
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
        self.save_dict_to_cache(merged, filename)


    def save_processor_hash(self, processor_name: str, hash_value: str):
        """
        Save the hash of a specific processor.

        Args:
            processor_name (str): Name of the processor.
            hash_value (str): Hash of the processor file.
        """
        hashes = load_json(self.processor_hash_file)
        hashes[processor_name] = hash_value
        save_json(self.processor_hash_file, hashes)

    def load_processor_hashes(self):
        """
        Load previously saved processor hashes.

        Returns:
            dict: Processor names mapped to their hashes.
        """
        return load_json(self.processor_hash_file)

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