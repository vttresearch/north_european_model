# src/source_excel_data_pipeline.py
"""
Source Excel Data Pipeline
==========================

Purpose
-------
SourceExcelDataPipeline is the first and only stage of the pipeline that is aware of
the scenario context. It reads source Excel files, filters them to the current
(scenario, year, scenario_alternative) combination, merges them row-by-row, and
exposes one normalized DataFrame per data category. Downstream stages (time series
processing, BuildInputExcel) receive scenario-neutral data and do not need to repeat
scenario filtering.

Scenario and alternative handling
----------------------------------
build_input_data.py drives execution by iterating over the Cartesian product of
(scenarios x scenario_years x scenario_alternatives x scenario_alternatives2 x
scenario_alternatives3 x scenario_alternatives4). For each combination a new
pipeline instance is created and run(). Alternative axes that are unused in a config
file default to [""], which contributes no filtering and no folder-name segment.

Within each data category the scenario filtering works in two steps:

1. apply_whitelist() retains only rows whose 'scenario' column matches either the
   current scenario, any of the active scenario_alternatives (if provided), or the
   catch-all value 'all'. The 'year' column is similarly filtered to the target year
   or the catch-all value 1. This means a single source Excel file can contain rows
   for multiple scenarios and years; only the relevant rows survive.

2. merge_row_by_row() processes the surviving rows in file load order. Each row
   carries a 'method' value (replace / replace-partial / add / multiply / remove /
   ...) that controls how it interacts with previously seen rows for the same key.
   Because rows are handled in the order of appearance, base-scenario data can be
   overwritten or supplemented in further input data files, and alternative-axis rows
   can override or supplement base values without duplicating the full dataset.

Output DataFrames
-----------------
After run() the following attributes are populated (empty DataFrame if no source
files are configured):

  df_unittypedata        generator type parameters       key: generator_id
  df_fueldata            fuel/carrier cost parameters    key: grid
  df_emissiondata        emission factors                key: emission
  df_demanddata          demand parameters               key: country, grid, node
  df_storagedata         storage parameters              key: country, grid, node
  df_unitdata            unit capacity parameters        key: country, generator_id, unit_name_prefix
  df_transferdata        interconnector parameters       key: from, from_suffix, to, to_suffix, grid
  df_userconstraintdata  custom constraint parameters    key: group, dimensions, parameter

Data conventions (all output DataFrames)
-----------------------------------------
- Column names: lowercase, whitespace-stripped.
- String values in key and categorical columns: lowercase, whitespace-stripped.
- Numeric columns: pandas Float64 (nullable).
- Missing values: pd.NA (empty strings are converted during normalize_dataframe).
- Metadata columns (_source_file, _source_sheet, method): dropped by merge_row_by_row.
- Scenario and year columns: retained in most output DataFrames. Downstream stages can
  drop them; they carry no further information because filtering has already been
  applied. Exception: df_userconstraintdata drops 'scenario', 'year', and 'country'
  before merging because its key structure makes them redundant.
- Topology columns (node, from_node, to_node): derived from component columns (country,
  grid, suffix) by data_loader helpers during run(); the component columns are also
  retained.
"""

from pathlib import Path
import src.data_loader as data_loader
import src.utils as utils
import pandas as pd



class SourceExcelDataPipeline:
    """
    InputDataPipeline handles reading, merging, filtering, and validating all input Excel files.
    """

    def __init__(self, config: dict, input_folder: Path,
                 scenario: str, scenario_year: int,
                 scenario_alternative: str = None,
                 scenario_alternative2: str = None,
                 scenario_alternative3: str = None,
                 scenario_alternative4: str = None,
                 country_codes: list = None):
        """
        Initialize InputDataPipeline.

        Args:
            config (dict): Parsed configuration dictionary.
            input_folder (Path): Base input folder containing src_files/data_files.
            scenario (str): Selected scenario.
            scenario_year (int): Selected scenario year.
            scenario_alternative (str, optional): Selected scenario alternative (1st axis).
            scenario_alternative2 (str, optional): Selected scenario alternative (2nd axis).
            scenario_alternative3 (str, optional): Selected scenario alternative (3rd axis).
            scenario_alternative4 (str, optional): Selected scenario alternative (4th axis).
            country_codes (list, optional): List of country codes.
        """
        self.config = config
        self.input_folder = input_folder
        self.data_folder = input_folder / "data_files"

        self.scenario = scenario
        self.scenario_year = scenario_year
        self.scenario_alternative = scenario_alternative
        self.scenario_alternative2 = scenario_alternative2
        self.scenario_alternative3 = scenario_alternative3
        self.scenario_alternative4 = scenario_alternative4
        self.country_codes = country_codes

        self.df_demanddata = pd.DataFrame()
        self.df_transferdata = pd.DataFrame()
        self.df_unittypedata = pd.DataFrame()
        self.df_unitdata = pd.DataFrame()
        self.df_remove_units = pd.DataFrame()
        self.df_storagedata = pd.DataFrame()
        self.df_fueldata = pd.DataFrame()
        self.df_emissiondata = pd.DataFrame()
        self.df_userconstraintdata = pd.DataFrame()        

        self.logs = []

    def run(self):
        """
        Run the full pipeline: load, process, and filter all input DataFrames.
        """
        input_folder = self.data_folder

        # Build scenario whitelist: base scenario plus any non-empty alternatives
        scen_and_alt = [self.scenario]
        for alt in (self.scenario_alternative, self.scenario_alternative2,
                    self.scenario_alternative3, self.scenario_alternative4):
            if alt:
                scen_and_alt.append(alt)

        # --- global datasets ---
        # unittypedata
        files = self.config['unittypedata_files']
        if len(files) > 0:
            dfs = data_loader.read_input_excels(input_folder, files, 'unittypedata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'unittypedata', self.logs) for df in dfs]
            dfs = [data_loader.drop_underscore_values(df, 'unittypedata', self.logs) for df in dfs]
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year}, self.logs, 'unittypedata')
                   for df in dfs
                   ]
            self.df_unittypedata = data_loader.merge_row_by_row(dfs, self.logs, key_columns=['generator_id'])
        else:
            utils.log_status(
                f"No Excel files for 'unittypedata_files' defined in the config file",
                self.logs, level="info"
            )

        # fueldata
        files = self.config['fueldata_files']
        if len(files) > 0:
            dfs = data_loader.read_input_excels(input_folder, files, 'fueldata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'fueldata', self.logs) for df in dfs]
            dfs = [data_loader.drop_underscore_values(df, 'fueldata', self.logs) for df in dfs]
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year}, self.logs, 'fueldata')
                   for df in dfs
                   ]
            self.df_fueldata = data_loader.merge_row_by_row(dfs, self.logs, key_columns=['grid'])
        else:
            utils.log_status(
                f"No Excel files for 'fueldata_files' defined in the config file",
                self.logs, level="info"
            )                      

        # emissiondata
        files = self.config['emissiondata_files']
        if len(files) > 0:        
            dfs = data_loader.read_input_excels(input_folder, files, 'emissiondata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'emissiondata', self.logs) for df in dfs]
            dfs = [data_loader.drop_underscore_values(df, 'emissiondata', self.logs) for df in dfs]
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year}, self.logs, 'emissiondata')
                   for df in dfs
                   ]
            self.df_emissiondata = data_loader.merge_row_by_row(dfs, self.logs, key_columns=['emission'])
        else:
            utils.log_status(
                f"No Excel files for 'emissiondata_files' defined in the config file",
                self.logs, level="info"
            )                   
 

        # --- country-level datasets ---
        exclude_grids = self.config['exclude_grids']
        exclude_nodes = self.config['exclude_nodes']

        # demanddata
        files = self.config['demanddata_files']
        if len(files) > 0:           
            dfs = data_loader.read_input_excels(input_folder, files, 'demanddata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'demanddata', self.logs) for df in dfs]
            dfs = [data_loader.drop_underscore_values(df, 'demanddata', self.logs) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, 'demanddata', {'grid': exclude_grids}) for df in dfs]
            dfs = [data_loader.build_node_column(df, self.logs) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, 'demanddata', {'node': exclude_nodes}) for df in dfs]            
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes}, 
                                   self.logs, 'demanddata')
                   for df in dfs
                   ]
            self.df_demanddata = data_loader.merge_row_by_row(dfs, self.logs, key_columns=['country', 'grid', 'node'])
            self.df_demanddata = data_loader.filter_nonzero_numeric_rows(self.df_demanddata, exclude=['year'])
        else:
            utils.log_status(
                f"No Excel files for 'demanddata_files' defined in the config file",
                self.logs, level="info"
            )                

        # storagedata
        files = self.config['storagedata_files']
        if len(files) > 0:            
            dfs = data_loader.read_input_excels(input_folder, files, 'storagedata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'storagedata', self.logs) for df in dfs]
            dfs = [data_loader.drop_underscore_values(df, 'storagedata', self.logs) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, 'storagedata', {'grid': exclude_grids}) for df in dfs]
            dfs = [data_loader.build_node_column(df, self.logs) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, 'storagedata', {'node': exclude_nodes}) for df in dfs]            
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes}, 
                                   self.logs, 'storagedata')
                   for df in dfs
                   ]
            self.df_storagedata = data_loader.merge_row_by_row(dfs, self.logs, key_columns=['country', 'grid', 'node'])
        else:
            utils.log_status(
                f"No Excel files for 'storagedata_files' defined in the config file",
                self.logs, level="info"
            )        

        # unitdata
        files = self.config['unitdata_files']
        if len(files) > 0:           
            dfs = data_loader.read_input_excels(input_folder, files, 'unitdata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'unitdata', self.logs) for df in dfs]
            dfs = [data_loader.drop_underscore_values(df, 'unitdata', self.logs) for df in dfs]     
            dfs = [data_loader.build_unittype_unit_column(df, self.df_unittypedata, self.logs) for df in dfs]           
            dfs = [data_loader.build_unit_grid_and_node_columns(df, self.df_unittypedata, self.logs) for df in dfs]            
            dfs = [data_loader.apply_unit_grids_blacklist(d, exclude_grids, df_name="unitdata", logs=self.logs) for d in dfs]     
            dfs = [data_loader.apply_unit_nodes_blacklist(d, exclude_nodes, df_name="unitdata", logs=self.logs) for d in dfs]
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes}, 
                                   self.logs, 'unitdata')
                   for df in dfs
                   ]
            self.df_unitdata = data_loader.merge_row_by_row(dfs, self.logs, key_columns=['country', 'generator_id', 'unit_name_prefix'])
        else:
            utils.log_status(
                f"No Excel files for 'unitdata_files' defined in the config file",
                self.logs, level="info"
            )                    
         
        # transferdata
        files = self.config['transferdata_files']
        if len(files) > 0:         
            dfs = data_loader.read_input_excels(input_folder, files, 'transferdata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'transferdata', self.logs) for df in dfs]
            dfs = [data_loader.drop_underscore_values(df, 'transferdata', self.logs) for df in dfs]       
            dfs = [data_loader.apply_blacklist(df, 'transferdata', {'grid': exclude_grids}) for df in dfs]        
            dfs = [data_loader.build_from_to_columns(df, self.logs) for df in dfs]               
            dfs = [data_loader.apply_blacklist(df, 'transferdata', {'from_node': exclude_nodes}) for df in dfs]  
            dfs = [data_loader.apply_blacklist(df, 'transferdata', {'to_node': exclude_nodes}) for df in dfs]  
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'from': self.country_codes}, 
                                   self.logs, 'transferdata')
                   for df in dfs
                   ]
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'to': self.country_codes}, 
                                   self.logs, 'transferdata')
                   for df in dfs
                   ]     
            self.df_transferdata = data_loader.merge_row_by_row(dfs, self.logs, key_columns=['from', 'from_suffix', 'to', 'to_suffix', 'grid'])
        else:
            utils.log_status(
                f"No Excel files for 'transferdata_files' defined in the config file",
                self.logs, level="info"
            )

        # --- custom datasets ---
        # userconstraintdata
        files = self.config['userconstraintdata_files']
        if len(files) > 0:         
            dfs = data_loader.read_input_excels(input_folder, files, 'userconstraintdata', self.logs,)
            dfs = [data_loader.normalize_dataframe(df, 'userconstraintdata', self.logs) for df in dfs]    
            dfs = [data_loader.apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes}, 
                                   self.logs, 'userconstraintdata')
                   for df in dfs
                   ]   
            dfs = [df.drop(columns=['scenario', 'year', 'country']) for df in dfs]
            self.df_userconstraintdata = data_loader.merge_row_by_row(
                                            dfs, self.logs, 
                                            key_columns=['group', '1st dimension', '2nd dimension', '3rd dimension', '4th dimension', 'parameter']
                                         )
        else:
            utils.log_status(
                f"No Excel files for 'userconstraintdata_files' defined in the config file",
                self.logs, level="info"
            )      


