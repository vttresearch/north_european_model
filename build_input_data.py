import time
from pathlib import Path
import src.config_reader as config_reader
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from src.pipeline.timeseries_pipeline import TimeseriesPipeline, TimeseriesRunResult
from src.pipeline.bb_excel_context import BBExcelBuildContext
import src.utils as utils
from itertools import product
from src.build_input_excel import BuildInputExcel
from datetime import datetime

 
def main(input_folder: Path, config_file: Path):
    # --- 1. Prep ---
    # Timer to follow the progress
    start_time = time.time()

    # Check versions and other dependencies
    utils.check_dependencies()

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

    # Initialize logging function
    utils.init_logging(
        print_all_elapsed_times = config['print_all_elapsed_times'],
        start_time = start_time
    )


    # --- 2. The (scenario, year, alternative) loop ---

    # Lists of scenarios, scenario_years, and alternatives
    scenarios = config['scenarios']
    scenario_years = config['scenario_years']
    scenario_alternatives = config['scenario_alternatives']

    # Reference folder for copying input-data-independent timeseries between iterations
    reference_ts_folder = None

    for scenario, year, alternative in product(scenarios, scenario_years, scenario_alternatives):

        # --- 2.1. Preparations ---
        # Reset warning log and elapsed-time clock from previous iteration
        utils.reset_warning_log()
        utils.reset_start_time()
        iteration_start_time = time.time()

        # accumulated log messages to be written to summary.log
        log_messages = []

        # Printing the (scenario, year, alternative) combination and storing them to scenario_tags
        
        if alternative != "":
            utils.log_status(f"{scenario}, {year}, {alternative}", log_messages, section_start_length=70, add_empty_line_before=True)      
            scen_tags = [scenario, str(year), alternative]  
        else:
            utils.log_status(f"{scenario}, {year}", log_messages, section_start_length=70, add_empty_line_before=True)   
            scen_tags = [scenario, str(year), ""]

        # Print date and time
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        utils.log_status(f"Run timestamp: {now_str}", log_messages, level="info")

        # Build output folder_name, check existence
        output_folder_prefix = config['output_folder_prefix']
        if alternative:
            folder_name = f"{output_folder_prefix}_{scenario}_{year}_{alternative}"
        else:
            folder_name = f"{output_folder_prefix}_{scenario}_{year}"
        output_folder = Path("") / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        utils.log_status(f"Using output folder: {output_folder}", log_messages, level="info")


        # --- 2.2. Cache manager ---
        # Initialize cache manager
        cache_manager = CacheManager(input_folder, output_folder, config)

        # Run cache manager to check which parts of code need rerunning, pick logs to log messages
        cache_manager.run()
        log_messages.extend(cache_manager.logs)

        # if full rerun, clear all files from output folder, keep subfolders and their files
        if cache_manager.full_rerun:
            output_path = Path(output_folder)

            # Find all files under output_folder, exclude subdirectories
            files = list(output_path.glob("*"))
            files = [f for f in files if f.is_file()]

            # If there were any, log once and delete them
            if files:
                for f in files:
                    f.unlink(missing_ok=True)
                utils.log_status(f"Cleared {len(files)} files from the output folder: {output_folder}, kept subfolders and their files.",
                           log_messages, level="done", add_empty_line_before=True)

        # --- 2.3. Input data phase ---
        # Initialize source excel pipeline
        source_excel_data_pipeline = SourceExcelDataPipeline(
            config=config,
            input_folder=input_folder,
            scenario=scenario,
            scenario_year=year,
            scenario_alternative=alternative,
            country_codes=config['country_codes']
        )

        # Run if needed
        if cache_manager.reimport_source_excels:
            utils.log_status("Processing source Excel files.", 
                             log_messages, level="run", 
                             add_empty_line_before=True, section_start_length=55)
            source_excel_data_pipeline.run()
            log_messages.extend(source_excel_data_pipeline.logs)
            utils.log_status("Source excel files processed successfully.", log_messages, level="done")
        else:
            utils.log_status("Skipping source excel processing.", log_messages, level="skip")


        # --- 2.4. Timeseries processing phase ---
        # Run timeseries or load cached results
        if cache_manager.needs_timeseries_run:
            utils.log_status(
                "Starting timeseries processing phase",
                log_messages,
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
                scenario_year=year
            )
            ts_results = ts_pipeline.run()
            log_messages.extend(ts_results.logs)

            # Set reference folder for subsequent iterations to enable copy optimization
            if reference_ts_folder is None:
                reference_ts_folder = output_folder
        else:
            utils.log_status(
                "Timeseries results are up-to-date. Loading from cache.",
                log_messages,
                level="skip"
            )
            # Load cached results
            ts_results = TimeseriesRunResult(
                secondary_results=cache_manager.load_all_secondary_results(),
                ts_domains=cache_manager.load_dict_from_cache("all_ts_domains.json"),
                ts_domain_pairs=cache_manager.load_dict_from_cache("all_ts_domain_pairs.json"),
                logs=[]
            )

        # --- 2.5. Backbone Input Excel building phase ---
        # Checking if this step is needed or not
        if cache_manager.rebuild_bb_excel:
            utils.log_status("Building Backbone input Excel", log_messages, level="run", section_start_length=45, add_empty_line_before=True)

            excel_context = BBExcelBuildContext(
                input_folder=input_folder,
                output_folder=output_folder,
                scen_tags=scen_tags,
                config=config,
                cache_manager=cache_manager,
                source_data=source_excel_data_pipeline,
                ts_results=ts_results
            )  

            builder = BuildInputExcel(excel_context)
            builder.run()
            log_messages.extend(builder.builder_logs)
            bb_excel_succesfully_built = builder.bb_excel_succesfully_built

        else:
            utils.log_status("Backbone input excel is up-to-date. Skipping build phase.", log_messages, level="skip")
            # Flagging bb excel succesfully built to pass checks at the end
            bb_excel_succesfully_built = True

        # Update the general flag for succesfull BB excel building
        if not bb_excel_succesfully_built:
            utils.log_status("Backbone input excel building failed. Rerun the python script.", log_messages, level="warn")
        status_dict = {"bb_excel_succesfully_built": bb_excel_succesfully_built}
        cache_manager.merge_dict_to_cache(status_dict, "general_flags.json")

        # --- 2.6. Finalizing ---

        utils.log_status("Finalizing", log_messages, level="run", section_start_length=55, add_empty_line_before=True)

        # Copying GAMS files for a new run or changed topology
        if cache_manager.full_rerun:
            utils.log_status("Copying GAMS files to input folder.", log_messages, level="run")
            utils.copy_gams_files(input_folder, output_folder, log_messages)

        # Flagging the run successful and writing the flag status
        status_dict = {"workflow_run_successfully": True}
        cache_manager.merge_dict_to_cache(status_dict, "general_flags.json")

        # Printing elapsed time (per iteration)
        minutes, seconds = utils.elapsed_time(iteration_start_time)
        utils.log_status(f"Completed in {minutes} min {seconds} sec.", log_messages, level="done")

        # Cumulative time (console only, not in log)
        cum_minutes, cum_seconds = utils.elapsed_time(start_time)
        print(f"  (Cumulative time: {cum_minutes} min {cum_seconds} sec)")

        # Repeat collected warnings and errors at the end for visibility
        warnings = utils.get_warning_log()
        if warnings:
            utils.log_status("Warnings and errors summary",
                             log_messages, level="none",
                             add_empty_line_before=True,
                             section_start_length=55)
            for w in warnings:
                log_messages.append(w)
                print(w)

        # Define log path
        log_path = output_folder / "summary.log"
        utils.log_status(f"Writing the log to {log_path}", log_messages, level="run", add_empty_line_before=True)
        
        # If previous log exist, add its contents to a "Previous logs" section
        if log_path.exists():
            utils.log_status("Previous logs found and added to current log", log_messages, level="info")
            utils.log_status(f"Previous logs", 
                       log_messages, level="none", 
                       add_empty_line_before=True, 
                       section_start_length=90, 
                       print_to_screen=False)
            with open(log_path, "r", encoding="utf-8") as f:
                previous_logs = f.read().splitlines()
            log_messages.extend(previous_logs)

        # Write final merged log
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_messages))


if __name__ == "__main__":
    # Parse CLI arguments 
    input_folder, config_file = utils.parse_sys_args()
    print(f"\nLaunching pipelines defined in: {config_file}")
    main(input_folder, config_file)
