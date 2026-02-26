import sys
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product

import src.config_reader as config_reader
from src.pipeline.cache_manager import CacheManager
from src.pipeline.logger import IterationLogger
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from src.pipeline.timeseries_pipeline import TimeseriesPipeline, TimeseriesRunResult
from src.pipeline.bb_excel_context import BBExcelBuildContext
from src.build_input_excel import BuildInputExcel


def main(input_folder: Path, config_file: Path):
    # --- 1. Prep ---
    # Timer to follow the progress
    start_time = time.time()

    # Check versions and other dependencies
    _check_dependencies()

    # Guarantee that input_folder is Path, check it exists
    input_folder = Path(input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Could not find directory {input_folder.resolve()}, please check spelling")
        return 1  # or: sys.exit(1)

    # Guarantee that config_file is Path, check it exists
    config_file = Path(config_file)
    if not config_file.exists() or not config_file.is_file():
        raise ValueError(f"Could not find file {config_file.resolve()}, please check spelling")

    # Load config file
    config = config_reader.load_config(config_file)



    # --- 2. The (scenario, year, alternative) loop ---

    # Lists of scenarios, scenario_years, and alternatives
    scenarios = config['scenarios']
    scenario_years = config['scenario_years']
    scenario_alternatives = config['scenario_alternatives']
    scenario_alternatives2 = config['scenario_alternatives2']
    scenario_alternatives3 = config['scenario_alternatives3']
    scenario_alternatives4 = config['scenario_alternatives4']

    # Reference folder for copying input-data-independent timeseries between iterations
    reference_ts_folder = None

    for scenario, year, alt1, alt2, alt3, alt4 in product(
        scenarios, scenario_years,
        scenario_alternatives, scenario_alternatives2,
        scenario_alternatives3, scenario_alternatives4
    ):


        # --- 2.1. Preparations ---
        # Create per-iteration logger: resets warning log and elapsed-time clock
        logger = IterationLogger(print_all_elapsed_times=config['print_all_elapsed_times'])
        iteration_start_time = time.time()

        # Collect non-empty alternatives for this combination
        active_alts = [a for a in [alt1, alt2, alt3, alt4] if a]

        # Printing the (scenario, year, alternatives) combination and storing them to scenario_tags
        if active_alts:
            logger.log_status(f"{scenario}, {year}, {', '.join(active_alts)}", section_start_length=70, add_empty_line_before=True)
        else:
            logger.log_status(f"{scenario}, {year}", section_start_length=70, add_empty_line_before=True)
        # Each active alternative is stored as a separate element; build_input_excel.py
        # uses the list length to determine which alternative columns to write.
        scen_tags = [scenario, str(year)] + active_alts

        # Print date and time
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.log_status(f"Run timestamp: {now_str}", level="none")

        # Build output folder_name, check existence
        output_folder_prefix = config['output_folder_prefix']
        folder_name = "_".join(part.replace(" ", "") for part in [output_folder_prefix, scenario, str(year)] + active_alts)
        output_folder = Path("") / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.log_status(f"Using output folder: {output_folder}", level="info")


        # --- 2.2. Cache manager ---
        # Initialize cache manager
        cache_manager = CacheManager(input_folder, output_folder, config, logger=logger)

        # Run cache manager to check which parts of code need rerunning
        cache_manager.run()

        # On full rerun, delete root-level output files (inputData.xlsx, GAMS files, logs, etc.)
        # Subdirectories such as cache/ are preserved — cache_manager.run() already cleaned those.
        if cache_manager.full_rerun:
            output_path = Path(output_folder)

            # glob("*") + is_file() matches only direct children, not files inside subdirectories
            files = list(output_path.glob("*"))
            files = [f for f in files if f.is_file()]

            if files:
                for f in files:
                    f.unlink(missing_ok=True)
                logger.log_status(f"Cleared {len(files)} output files from {output_folder}.",
                           level="info", add_empty_line_before=True)

        # --- 2.3. Input data phase ---
        # Initialize source excel pipeline
        source_excel_data_pipeline = SourceExcelDataPipeline(
            config=config,
            input_folder=input_folder,
            scenario=scenario,
            scenario_year=year,
            scenario_alternative=alt1,
            scenario_alternative2=alt2,
            scenario_alternative3=alt3,
            scenario_alternative4=alt4,
            country_codes=config['country_codes'],
            logger=logger
        )

        # Run if needed
        if cache_manager.reimport_source_excels:
            logger.log_status("Processing source Excel files.",
                       level="run", add_empty_line_before=True, section_start_length=55)
            source_excel_data_pipeline.run()
        else:
            logger.log_status("Skipping source excel processing.", level="skip")


        # --- 2.4. Timeseries processing phase ---
        # Run timeseries or load cached results
        if cache_manager.needs_timeseries_run:
            logger.log_status(
                "Starting timeseries processing phase",
                level="run",
                add_empty_line_before=True, section_start_length=55
            )

            ts_pipeline = TimeseriesPipeline(
                config,
                input_folder,
                output_folder,
                cache_manager,
                source_excel_data_pipeline,
                reference_ts_folder=reference_ts_folder,
                scenario_year=year,
                logger=logger
            )
            ts_results = ts_pipeline.run()

            # Set reference folder for subsequent iterations to enable copy optimization
            if reference_ts_folder is None:
                reference_ts_folder = output_folder
        else:
            logger.log_status(
                "Timeseries results are up-to-date. Loading from cache.",
                level="skip"
            )
            # Load cached results
            ts_results = TimeseriesRunResult(
                secondary_results=cache_manager.load_all_secondary_results(),
                ts_domains=cache_manager.load_dict_from_cache("all_ts_domains.json"),
                ts_domain_pairs=cache_manager.load_dict_from_cache("all_ts_domain_pairs.json"),
            )

        # --- 2.5. Backbone Input Excel building phase ---
        # Checking if this step is needed or not
        if cache_manager.rebuild_bb_excel:
            logger.log_status("Building Backbone input Excel", level="run", section_start_length=45, add_empty_line_before=True)

            excel_context = BBExcelBuildContext(
                input_folder=input_folder,
                output_folder=output_folder,
                scen_tags=scen_tags,
                config=config,
                cache_manager=cache_manager,
                source_data=source_excel_data_pipeline,
                ts_results=ts_results
            )

            builder = BuildInputExcel(excel_context, logger=logger)
            builder.run()
            bb_excel_succesfully_built = builder.bb_excel_succesfully_built

        else:
            logger.log_status("Backbone input excel is up-to-date. Skipping build phase.", level="skip")
            # Flagging bb excel succesfully built to pass checks at the end
            bb_excel_succesfully_built = True

        # Update the general flag for succesfull BB excel building
        if not bb_excel_succesfully_built:
            logger.log_status("Backbone input excel building failed. Rerun the python script.", level="warn")
        status_dict = {"bb_excel_succesfully_built": bb_excel_succesfully_built}
        cache_manager.merge_dict_to_cache(status_dict, "general_flags.json")

        # --- 2.6. Finalizing ---

        logger.log_status("Finalizing", level="run", section_start_length=55, add_empty_line_before=True)

        # Copying GAMS files for a new run or changed topology
        if cache_manager.full_rerun:
            logger.log_status(f"Copying GAMS files to {output_folder}  ...", level="none")
            gams_src_folder = input_folder / "GAMS_files"
            if not gams_src_folder.exists():
                logger.log_status(f"GAMS source folder not found: {gams_src_folder}", level="warn")
            else:
                copied_any = False
                for file in gams_src_folder.glob("*.*"):
                    shutil.copy(file, output_folder / file.name)
                    logger.log_status(f"Copied {file.name}", level="info")
                    copied_any = True
                if not copied_any:
                    logger.log_status(f"No GAMS files found to copy in {gams_src_folder}", level="warn")

        # Flagging the run successful and writing the flag status
        status_dict = {"workflow_run_successfully": True}
        cache_manager.merge_dict_to_cache(status_dict, "general_flags.json")

        # Printing elapsed time (per iteration)
        minutes, seconds = logger.elapsed_time(iteration_start_time)
        logger.log_status(f"Completed in {minutes} min {seconds} sec.", level="done", add_empty_line_before=True)

        # Cumulative time (console only, not in log)
        cum_minutes, cum_seconds = logger.elapsed_time(start_time)
        print(f"  (Cumulative time: {cum_minutes} min {cum_seconds} sec)")

        # Repeat collected warnings and errors at the end for visibility
        warnings = logger.warnings
        if warnings:
            logger.log_status("Warnings and errors summary",
                       level="none",
                       add_empty_line_before=True,
                       section_start_length=55)
            for w in warnings:
                logger.messages.append(w)
                print(w)

        # Define log path
        log_path = output_folder / "summary.log"
        logger.log_status(f"Writing the log to {log_path}", level="run", add_empty_line_before=True)

        # If previous log exist, add its contents to a "Previous logs" section
        if log_path.exists():
            logger.log_status("Previous logs found and added to current log", level="info")
            logger.log_status("Previous logs",
                       level="none",
                       add_empty_line_before=True,
                       section_start_length=90,
                       print_to_screen=False)
            with open(log_path, "r", encoding="utf-8") as f:
                previous_logs = f.read().splitlines()
            logger.messages.extend(previous_logs)

        # Write final merged log
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(logger.messages))





def _parse_sys_args():
    # Instructions in case of mispelled input cmd
    USAGE_MSG = (
        "Usage: python build_input_data.py <input_folder> <config_file>,\n"
        "       e.g. python build_input_data.py src_files config_test.ini"
    )
        
    # detect legacy key=val syntax
    if any("=" in arg for arg in sys.argv[1:]):
        print(USAGE_MSG)
        sys.exit(1)
    else:
        # strict positional: both required
        parser = argparse.ArgumentParser(
            usage=USAGE_MSG,
            description="NorthEuropeanBackbone Input Builder"
        )
        parser.add_argument(
            "input_folder",
            type=str,
            help="Input folder (e.g. src_files)"
        )
        parser.add_argument(
            "config_file",
            type=str,
            help="Config file name (relative to input_folder)"
        )
        # argparse will print our USAGE_MSG if args are missing
        args = parser.parse_args()
        input_folder = Path(args.input_folder)
        config_file  = Path(input_folder, args.config_file)

        return (input_folder, config_file)
    

def _check_dependencies():
    """
    Verifies required dependencies.
        - Python >= 3.12
        - pandas >= 2.2
        - pyarrow
        - tqdm
        - gams.transfer importable
        - gams executable accessible in PATH

    Raises RuntimeError if any requirement is not met.
    """
    import importlib

    errors = []

    # Check Python version
    py_major, py_minor = sys.version_info[:2]
    if (py_major, py_minor) < (3, 12):
        errors.append(f"Python {py_major}.{py_minor} detected (requires ≥3.12), see readme.md how to install/update the environment.")

    # Check pandas version
    try:
        import pandas as pd
        pd_major, pd_minor = map(int, pd.__version__.split('.')[:2])
        if (pd_major, pd_minor) < (2, 2):
            errors.append(f"pandas {pd_major}.{pd_minor} detected (requires ≥2.2)")
    except ImportError:
        errors.append("pandas not installed, see readme.md how to install/update the environment.")  

    # Check pyarrow availability
    try:
        importlib.import_module("pyarrow")
    except ImportError:
        errors.append("pyarrow not installed, see readme.md how to install/update the environment.")

    # Check tqdm availability
    try:
        importlib.import_module("tqdm")
    except ImportError:
        errors.append("tqdm not installed, see readme.md how to install/update the environment.")

    # Check gams.transfer importability
    try:
        importlib.import_module("gams.transfer")
    except ImportError:
        errors.append("gams.transfer not importable (GAMS Python API missing), see readme.md how to install/update the environment.")

    # Check gams executable availability in PATH
    gams_exec = shutil.which("gams") or shutil.which("gams.exe")
    if gams_exec is None:
        errors.append("GAMS not found in PATH")

    # Final decision
    if errors:
        msg = "Dependency check failed:\n  - " + "\n  - ".join(errors)
        raise RuntimeError(msg)




if __name__ == "__main__":
    # Parse CLI arguments
    input_folder, config_file = _parse_sys_args()
    print(f"\nLaunching pipelines defined in: {config_file}")
    main(input_folder, config_file)
