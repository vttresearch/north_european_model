import json
from pathlib import Path
from src.hash_utils import compute_file_hash
from src.utils import log_status
import pickle
from datetime import datetime

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

        # storing specific rerun switches
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

        # Checking changes since previous run
        log = self.validate_topology(self.config)
        validation_log.extend(log)
        
        log = self.validate_input_files(self.config, self.input_file_folder)
        validation_log.extend(log)

        log = self.validate_start_and_end(self.config)
        validation_log.extend(log)

        log = self.validate_csv_writer(self.config)
        validation_log.extend(log)

        self.rerun_all_ts = self.topology_changed or self.date_range_expanded or self.csv_writer_requested
        log = self.validate_timeseries(self.config, self.rerun_all_ts, self.demand_files_changed)
        validation_log.extend(log)

        # Save current config
        self.save_structural_config(self.config)

        # Checking if source excels should be re-imported
        self.reimport_source_excels = (self.topology_changed
                                       or self.demand_files_changed
                                       or self.other_input_files_changed
                                       or self.rerun_all_ts
        )        

        # Checking if BB input excel needs to be rebuilt
        self.rebuild_bb_excel = (self.topology_changed
                                 or self.demand_files_changed
                                 or self.other_input_files_changed
                                 or self.rerun_all_ts
        )

        return validation_log


    def validate_topology(self, config: dict) -> list[str]:
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
    

    def validate_start_and_end(self, config: dict) -> list[str]:
        prev = self.load_structural_config()

        # If previous config was never saved, treat it as a change
        if not prev:
            self.date_range_expanded = True

        def parse(dt): return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

        old_start = parse(prev.get("start_date", config["start_date"]))
        old_end = parse(prev.get("end_date", config["end_date"]))
        new_start = parse(config["start_date"])
        new_end = parse(config["end_date"])

        self.date_range_expanded = new_start < old_start or new_end > old_end

        # Printing to log
        validation_log = []
        if self.date_range_expanded:
            log_status('Requested time range has expanded from previous run, rerunning all timeseries.', validation_log, level="run")

        return validation_log


    def validate_csv_writer(self, config: dict) -> list[str]:
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


    def validate_input_files(self, config: dict, input_folder: Path) -> list[str]:
        """
        Checks if input files have changed by comparing with stored hashes.
    
        Args:
            config (dict): Parsed configuration dictionary.
            input_folder (Path): Folder containing all input files.
    
        Returns:
            tuple: (demand_files_changed, other_input_files_changed)
        """
        # Extract file lists
        demand_files = config.get("demanddata_files", [])
        other_keys = [
            "unittypedata_files", "fueldata_files", "emissiondata_files",
            "transferdata_files", "unitdata_files", "storagedata_files"
        ]
        other_files = [f for key in other_keys for f in config.get(key, [])]
    
        # Compute current hashes
        demand_hashes = {f: compute_file_hash(input_folder / f) for f in demand_files}
        other_hashes = {f: compute_file_hash(input_folder / f) for f in other_files}
    
        # Load previous hashes
        previous_hashes = self.load_json(self.input_data_hash_file)
    
        # Compare
        self.demand_files_changed = any(previous_hashes.get(f) != h for f, h in demand_hashes.items())
        self.other_input_files_changed = any(previous_hashes.get(f) != h for f, h in other_hashes.items())
    
        # Save current combined hash set
        self.save_json(self.input_data_hash_file, {**demand_hashes, **other_hashes})
    
        # Printing to log
        validation_log = []
        if self.demand_files_changed and not self.topology_changed:
            log_status('Demand input data files changed, rerunning demand timeseries and input excel.', validation_log, level="run")

        if self.other_input_files_changed and not self.topology_changed:
            log_status('Other input data files changed, rerunning input excel.', validation_log, level="run")


        return validation_log


    def validate_timeseries(self, config: dict, rerun_all_ts: bool = False, demand_files_changed: bool = False) -> list[str]:
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
                f'Noticed changes in timeseries specifications, rerunning following timeseries: {", ".join(changed_keys)}',
                validation_log,
                level="info"
            )

        return validation_log



    def load_json(self, path: Path):
        """
        Load a JSON file from a given path.

        Args:
            path (Path): Path to the JSON file.

        Returns:
            dict: Parsed JSON content.
        """
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def save_json(self, path: Path, data: dict):
        """
        Save a dictionary to a JSON file.

        Args:
            path (Path): Path where JSON will be saved.
            data (dict): Data to save.
        """
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load_structural_config(self) -> dict:
        return self.load_json(self.cache_folder / "config_structural.json")

    def save_structural_config(self, config: dict):
        relevant_keys = [
            "country_codes", "exclude_grids", "exclude_nodes",
            "start_date", "end_date", "write_csv_files", "timeseries_specs"
        ]
        data = {k: config[k] for k in relevant_keys if k in config}
        self.save_json(self.cache_folder / "config_structural.json", data)


    def save_processor_hash(self, processor_name: str, hash_value: str):
        """
        Save the hash of a specific processor.

        Args:
            processor_name (str): Name of the processor.
            hash_value (str): Hash of the processor file.
        """
        hashes = self.load_json(self.processor_hash_file)
        hashes[processor_name] = hash_value
        self.save_json(self.processor_hash_file, hashes)

    def load_processor_hashes(self):
        """
        Load previously saved processor hashes.

        Returns:
            dict: Processor names mapped to their hashes.
        """
        return self.load_json(self.processor_hash_file)

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