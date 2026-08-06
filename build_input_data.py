import sys
import re
import math
import shutil
import time
from pathlib import Path
from datetime import datetime
from itertools import product

import src.infrastructure.config_reader as config_reader
from src.infrastructure.cache_manager import CacheManager
from src.infrastructure.logger import IterationLogger
from src.source_data.source_data_inputs import SourceDataPipelineInputs
from src.source_data.source_data_pipeline import SourceDataPipeline
from src.timeseries.timeseries_inputs import TimeseriesPipelineInputs
from src.timeseries.timeseries_pipeline import TimeseriesPipeline
from src.timeseries.timeseries_results import TimeseriesPipelineOutput
from src.bb_excel.bb_excel_inputs import BBExcelInputs
from src.bb_excel.bb_excel_pipeline import BBExcelPipeline
from src.utils import parse_sys_args, force_utf8_output

#: Project root -- this file's own directory. Output folders are created here by
#: default rather than in the working directory, so the same command produces
#: the same output wherever it is run from.
_REPO_ROOT = Path(__file__).resolve().parent


def main(input_folder: Path, config_file: Path, output_root: Path | None = None):
    """Build Backbone input data for every scenario the config declares.

    output_root:
        Where the per-scenario output folders are created. Defaults to the
        project root, i.e. beside build_input_data.py. It used to be the working
        directory, which meant the same command run from two places maintained
        two independent caches and two half-built outputs -- and since the cache
        lives inside the output folder, neither knew about the other.
    """
    # --- 1. Prep ---
    output_root = Path(output_root) if output_root is not None else _REPO_ROOT

    # Make log output survive redirection before anything logs. On Windows a
    # redirected stream falls back to the locale encoding, and the status
    # prefixes are non-ASCII, so `> build.log` used to kill the run on its first
    # line. Called here rather than under __main__ so the Spine Toolbox wrapper
    # gets it too.
    force_utf8_output()

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
        # Each active alternative is stored as a separate element; bb_excel_pipeline.py
        # uses the list length to determine which alternative columns to write.
        scen_tags = [scenario, str(year)] + active_alts

        # Print date and time
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.log_status(f"Run timestamp: {now_str}", level="none")

        # Build output folder_name, check existence
        output_folder_prefix = config['output_folder_prefix']
        folder_name = "_".join(part.replace(" ", "") for part in [output_folder_prefix, scenario, str(year)] + active_alts)
        output_folder = output_root / folder_name
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
                locked_file = None
                for f in files:
                    try:
                        f.unlink(missing_ok=True)
                    except PermissionError:
                        locked_file = f
                        break
                if locked_file:
                    logger.log_status(
                        f"Cannot delete '{locked_file.name}' — it is open in another program. "
                        f"Please close it and rerun the code.",
                        level="error"
                    )
                    continue
                logger.log_status(f"Cleared {len(files)} output files from {output_folder}.",
                           level="info", add_empty_line_before=True)

        # --- 2.3. Input data phase ---
        # Initialize source data pipeline
        source_data_pipeline = SourceDataPipeline(SourceDataPipelineInputs(
            config=config,
            input_folder=input_folder,
            scenario=scenario,
            scenario_year=year,
            scenario_alternative=alt1,
            scenario_alternative2=alt2,
            scenario_alternative3=alt3,
            scenario_alternative4=alt4,
            country_codes=config['country_codes'],
            logger=logger,
        ))

        # Run if needed
        error_count_before_source_excel = logger.error_count
        if cache_manager.reimport_source_excels:
            logger.log_status("Processing source Excel files.",
                       level="run", add_empty_line_before=True, section_start_length=55)
            source_data_pipeline.run()
        else:
            logger.log_status("Skipping source excel processing.", level="skip")
        source_excel_run_successfully = (logger.error_count == error_count_before_source_excel)


        # --- 2.4. Timeseries processing phase ---
        # Run timeseries or load cached results
        error_count_before_ts = logger.error_count
        if cache_manager.needs_timeseries_run:
            logger.log_status(
                "Starting timeseries processing phase",
                level="run",
                add_empty_line_before=True, section_start_length=55
            )

            ts_pipeline = TimeseriesPipeline(TimeseriesPipelineInputs(
                config=config,
                input_folder=input_folder,
                output_folder=output_folder,
                cache_manager=cache_manager,
                source_data_pipeline=source_data_pipeline,
                logger=logger,
                reference_ts_folder=reference_ts_folder,
                scenario_year=year,
            ))
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
            ts_results = TimeseriesPipelineOutput(
                secondary_results=cache_manager.load_all_secondary_results(),
                ts_domains=cache_manager.load_dict_from_cache("all_ts_domains.json"),
                ts_domain_pairs=cache_manager.load_dict_from_cache("all_ts_domain_pairs.json"),
            )
        timeseries_run_successfully = (logger.error_count == error_count_before_ts)

        # --- 2.5. Backbone Input Excel building phase ---

        # Checking if this step is needed or not
        if cache_manager.rebuild_bb_excel:
            logger.log_status("Building Backbone input Excel", level="run", section_start_length=45, add_empty_line_before=True)

            excel_context = BBExcelInputs(
                input_folder=input_folder,
                output_folder=output_folder,
                scen_tags=scen_tags,
                config=config,
                cache_manager=cache_manager,
                logger=logger,
                source_data=source_data_pipeline,
                ts_results=ts_results
            )

            builder = BBExcelPipeline(excel_context)
            builder.run()
            bb_excel_succesfully_built = builder.bb_excel_succesfully_built

        else:
            logger.log_status("Backbone input excel is up-to-date. Skipping build phase.", level="skip")
            # Flagging bb excel succesfully built to pass checks at the end
            bb_excel_succesfully_built = True

        # Update the general flag for succesfull BB excel building
        status_dict = {
            "bb_excel_succesfully_built": bb_excel_succesfully_built,
            "source_excel_run_successfully": source_excel_run_successfully,
            "timeseries_run_successfully": timeseries_run_successfully,
        }
        cache_manager.merge_dict_to_cache(status_dict, "general_flags.json")

        # --- 2.6. Finalizing ---

        logger.log_status("Finalizing", level="run", section_start_length=55, add_empty_line_before=True)

        bb_ts_length = config.get('bb_ts_length')    
        if cache_manager.full_rerun and bb_ts_length != 365:
            logger.log_status(
                f"Modifying gams files to work with bb_ts_length = {bb_ts_length}...",
                level="none"
            )            


        # Copying GAMS files for a new run or changed topology
        if cache_manager.full_rerun:
            logger.log_status(f"Copying GAMS files to {output_folder}  ...", level="none")
            gams_src_folder = input_folder / "GAMS_files"
            if not gams_src_folder.exists():
                logger.log_status(f"GAMS source folder not found: {gams_src_folder}", level="warn")
            else:
                copied_any = False
                for file in gams_src_folder.glob("*.*"):
                    content = file.read_text(encoding="utf-8")
                    content = _patch_gams_file_content(file.name, content, config)
                    (output_folder / file.name).write_text(content, encoding="utf-8")
                    logger.log_status(f"Copied {file.name}", level="info")
                    copied_any = True
                if not copied_any:
                    logger.log_status(f"No GAMS files found to copy in {gams_src_folder}", level="warn")

        # Flagging the run successful and writing the flag status
        # workflow_run_successfully is False if any error-level message was logged.
        # Source/timeseries failures trigger a full rerun next time; BB excel failure
        # triggers only a BB excel rebuild (already reported above at line 201).
        if not source_excel_run_successfully:
            logger.log_status(
                "Source excel phase had errors — a full rerun will be triggered on next run.",
                level="error"
            )
        if not timeseries_run_successfully:
            logger.log_status(
                "Timeseries phase had errors — a full rerun will be triggered on next run.",
                level="error"
            )
        if not bb_excel_succesfully_built:
            logger.log_status("Backbone input excel building failed — rerun the code.",
                level="error"
            )

        status_dict = {"workflow_run_successfully": not logger.has_errors}
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
        logger.log_status(f"Writing the log to {log_path}", level="none", add_empty_line_before=True)

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




def _patch_gams_file_content(filename: str, content: str, config: dict) -> str:
    """Return content with config-derived substitutions for known GAMS template files."""
    bb_ts_length = config.get("bb_timeseries_length", 365)

    # Mirrors mSettings('schedule', 't_horizon') = 24*7*65 in scheduleInit.gms.
    # Used to size the t-index upper bound in timeAndSamples.inc.
    def_t_horizon = 24 * 7 * 65  # 10920

    if filename == "scheduleInit.gms":
        data_length = bb_ts_length * 24
        content = content.replace(
            "mSettings('schedule', 'dataLength') =  8760;",
            f"mSettings('schedule', 'dataLength') =  {data_length};",
        )

        prob_lines = "".join(
            f"    p_mfProbability('schedule', '{label}') = {weight:g};\n"
            for label, weight in config["forecast_weights"].items()
        )
        content = re.sub(
            r'(    // NOTE: do not edit the lines below[^\n]*\n'
            r'    // unless[^\n]*\n'
            r'    // No restrictions for local versions\.[^\n]*\n)'
            r'(?:    p_mfProbability\([^\n]*\n)+',
            rf'\1{prob_lines}',
            content,
        )

    elif filename == "timeAndSamples.inc":
        t_max = math.ceil((bb_ts_length * 24 + def_t_horizon) / 1000) * 1000
        content = content.replace(
            "t000000 * t020000",
            f"t000000 * t{t_max:06d}",
        )

        # Floor at f01 so the declared range is never the invalid GAMS "f00 * f00".
        # Backbone filters active f at runtime, so an unused f01 here is harmless.
        last_f = f"f{max(len(config['forecast_quantiles']), 1):02d}"
        content = re.sub(
            r'(f00 \* )f\d+',
            rf'\g<1>{last_f}',
            content,
        )

    elif filename == "changes.inc":
        # forecastNumber = number of climatological forecast branches + 1 for f00 (realized weather)
        forecast_number = len(config["forecast_quantiles"]) + 1
        content = re.sub(
            r'(\$if not set forecasts \$evalglobal forecastNumber )\d+',
            rf'\g<1>{forecast_number}',
            content,
        )

    return content



if __name__ == "__main__":
    # Parse CLI arguments
    input_folder, config_file = parse_sys_args()
    print(f"\nLaunching pipelines defined in: {config_file}")
    main(input_folder, config_file)
