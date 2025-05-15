import time
from pathlib import Path
from gdxpds import to_gdx   # needed here to ensure gdxpds is imported before pandas
from src.config_reader import load_config
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from src.pipeline.timeseries_pipeline import TimeseriesPipeline
from src.pipeline.bb_excel_context import BBExcelBuildContext
from src.utils import parse_sys_args, elapsed_time, check_dependencies, log_status, copy_gams_files
from itertools import product
from src.build_input_excel import BuildInputExcel
from datetime import datetime

 
def main():
    # --- 1. setup ---
    # Timer to follow the progress
    start_time = time.time()

    # Check package versions
    check_dependencies()

    # Parse CLI arguments 
    input_folder, config_file = parse_sys_args()
    print(f"\nLaunching pipelines defined in: {config_file}")

    # --- 2. Loading config file, fetching parameters needed to launch pipelines ---
    config = load_config(config_file)


    # --- 3. The (scenario, year, alternative) loop ---
    # load list of processed countries
    country_codes = config.get('country_codes') # config parser checks that this mandatory field is given

    # Lists of scenarios, scenario_years, and alternatives
    scenarios = config.get('scenarios')
    scenario_years = config.get('scenario_years')
    scenario_alternatives = config.get('scenario_alternatives', [])

    for scenario, year, alternative in product(scenarios, scenario_years, scenario_alternatives):

        # --- 3.1. Preparations ---
        # accumulated log messages to be written to summary.log
        log_messages = []

        # Printing the (scenario, year, alternative) combination and storing them to scenario_tags
        if alternative != "":
            log_status(f"{scenario}, {year}, {alternative}", log_messages, section_start_length=70, add_empty_line_before=True)      
            scen_tags = [scenario, str(year), alternative]  
        else:
            log_status(f"{scenario}, {year}", log_messages, section_start_length=70, add_empty_line_before=True)   
            scen_tags = [scenario, str(year), ""]

        # Print date and time
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_status(f"Run timestamp: {now_str}", log_messages, level="info")

        # Build output folder name, check existence
        output_folder_prefix = config.get('output_folder_prefix', 'output')
        if alternative:
            folder_name = f"{output_folder_prefix}_{scenario}_{year}_{alternative}"
        else:
            folder_name = f"{output_folder_prefix}_{scenario}_{year}"
        output_folder = Path("") / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        log_status(f"Using output folder: {output_folder}", log_messages, level="info")


        # --- 3.2. Cache manager ---
        # Initialize cache manager
        cache_manager = CacheManager(input_folder, output_folder, config)

        # Run cache manager to check which parts of code need rerunning, pick logs to log messages
        log_messages.extend(cache_manager.run())
        

        # --- 3.3. Input data phase ---
        # Initialize source excel pipeline
        source_excel_data_pipeline = SourceExcelDataPipeline(
            config=config,
            input_folder=input_folder,
            scenario=scenario,
            scenario_year=year,
            scenario_alternative=alternative,
            country_codes=country_codes
        )

        # Run if needed, otherwise print skip message
        if cache_manager.reimport_source_excels:
            log_status("Processing source Excel files.", log_messages, level="run")
            source_excel_data_pipeline.run()
            log_status("Source excel files processed successfully.", log_messages, level="info")
        else:
            log_status("Skipping source excel processing.", log_messages, level="skip")


        # --- 3.4. Timeseries processing phase ---
        # Running timeseries if any processor needs rerunning or bb input excel needs to be created
        # Note: BB input excel needs correct ts_results from previous runs
        if cache_manager.rerun_all_ts or cache_manager.timeseries_changed or cache_manager.rebuild_bb_excel:
            # Instantiate pipeline and run
            ts_pipeline = TimeseriesPipeline(config, input_folder, output_folder, cache_manager, source_excel_data_pipeline)
            ts_results = ts_pipeline.run()
        
            # Merge logs
            log_messages.extend(ts_results.logs)
        else:
            log_status("Timeseries results are up-to-date. Skipping timeseries processing.", log_messages, level="skip")


        # --- 3.5. Backbone Input Excel building phase ---
        # Checking if this step is needed or not
        if cache_manager.rebuild_bb_excel:
            log_status("Building Backbone input Excel", log_messages, level="run", section_start_length=45, add_empty_line_before=True)
            # Instantiate excel_context and builder
            excel_context = BBExcelBuildContext(
                # From sys args
                input_folder=input_folder,
                # From currently looped run
                output_folder=output_folder,
                scen_tags=scen_tags,
                # From config file       
                config=config,
                # Cacge manager
                cache_manager=cache_manager,
                # From InputDataPipeline
                df_demanddata=source_excel_data_pipeline.df_demanddata,
                df_transferdata=source_excel_data_pipeline.df_transferdata,
                df_unittypedata=source_excel_data_pipeline.df_unittypedata,
                df_unitdata=source_excel_data_pipeline.df_unitdata,
                df_remove_units=source_excel_data_pipeline.df_remove_units,
                df_storagedata=source_excel_data_pipeline.df_storagedata,
                df_fueldata=source_excel_data_pipeline.df_fueldata,
                df_emissiondata=source_excel_data_pipeline.df_emissiondata,
                # From TimeseriesPipeline
                secondary_results=ts_results.secondary_results,
                ts_domains=ts_results.ts_domains,
                ts_domain_pairs=ts_results.ts_domain_pairs
            )  
            builder = BuildInputExcel(excel_context)
            # run builder, pick builder logs to log messages
            log_messages.extend(builder.run())
        else:
            log_status("Backbone input excel is up-to-date. Skipping build phase.", log_messages, level="skip")


        # --- 3.6. Finalizing ---
        log_status("Finalizing", log_messages, level="run", section_start_length=45, add_empty_line_before=True)

        # Copying GAMS files for a new run or changed topology
        if cache_manager.topology_changed:
            log_messages.extend(copy_gams_files(input_folder, output_folder))

        # Printing elapsed time
        minutes, seconds = elapsed_time(start_time)
        log_status(f"Completed in {minutes} min {seconds} sec.", log_messages, level="done")

        # Define log path
        log_path = output_folder / "summary.log"
        
        if cache_manager.topology_changed:
            # If full rerun, remove import_timeseries.inc and start a new log
            log_path.unlink(missing_ok=True)
            log_status(f"Writing a new log to {log_path}", log_messages, level="run", add_empty_line_before=True)
        else:
            # otherwise extend the existing log
            log_status(f"Writing the log to {log_path}", log_messages, level="run", add_empty_line_before=True)
        
            # If file exists, add its contents to a "Previous logs" section
            if log_path.exists():
                log_status("Previous logs found and added to current log", log_messages, level="info")
                log_status(f"Previous logs", 
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
    main()
