from pathlib import Path
import src.data_loader as data_loader
import src.excel_exchange as excel_exchange
import src.utils as utils
import pandas as pd



class SourceExcelDataPipeline:
    """
    InputDataPipeline handles reading, merging, filtering, and validating all input Excel files.
    """

    def __init__(self, config: dict, input_folder: Path, 
                 scenario: str, scenario_year: int, 
                 scenario_alternative: str = None, country_codes: list = None):
        """
        Initialize InputDataPipeline.

        Args:
            config (dict): Parsed configuration dictionary.
            input_folder (Path): Base input folder containing src_files/data_files.
            scenario (str): Selected scenario.
            scenario_year (int): Selected scenario year.
            scenario_alternative (str, optional): Selected scenario alternative.
            country_codes (list, optional): List of country codes.
        """
        self.config = config
        self.input_folder = input_folder
        self.data_folder = input_folder / "data_files"

        self.scenario = scenario
        self.scenario_year = scenario_year
        self.scenario_alternative = scenario_alternative
        self.country_codes = country_codes or []

        self.df_demanddata = pd.DataFrame()
        self.df_transferdata = pd.DataFrame()
        self.df_unittypedata = pd.DataFrame()
        self.df_unitdata = pd.DataFrame()
        self.df_remove_units = pd.DataFrame()
        self.df_storagedata = pd.DataFrame()
        self.df_fueldata = pd.DataFrame()
        self.df_emissiondata = pd.DataFrame()

        self.logs = []

    def run(self):
        """
        Run the full pipeline: load, process, and filter all input DataFrames.
        """
        input_folder = self.data_folder

        # Make a combination of scenario and alternative
        scen_and_alt = [self.scenario]
        if self.scenario_alternative:
            scen_and_alt.append(self.scenario_alternative)

        # --- global datasets ---
        # unittypedata
        files = self.config.get('unittypedata_files', [])
        if len(files) > 0:
            dfs = excel_exchange.read_input_excels(input_folder, files, 'unittypedata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'unittypedata', self.logs) for df in dfs]
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
        files = self.config.get('fueldata_files', [])
        if len(files) > 0:
            dfs = excel_exchange.read_input_excels(input_folder, files, 'fueldata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'fueldata', self.logs) for df in dfs]
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
        files = self.config.get('emissiondata_files', [])
        if len(files) > 0:        
            dfs = excel_exchange.read_input_excels(input_folder, files, 'emissiondata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'emissiondata', self.logs) for df in dfs]
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
        exclude_grids = self.config.get('exclude_grids', [])
        exclude_nodes = self.config.get('exclude_nodes', [])

        # demanddata
        files = self.config.get('demanddata_files', [])
        if len(files) > 0:           
            dfs = excel_exchange.read_input_excels(input_folder, files, 'demanddata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'demanddata', self.logs) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, 'demanddata', {'grid': exclude_grids}) for df in dfs]
            dfs = [data_loader.build_node_column(df) for df in dfs]       
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
        files = self.config.get('storagedata_files', [])
        if len(files) > 0:            
            dfs = excel_exchange.read_input_excels(input_folder, files, 'storagedata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'storagedata', self.logs) for df in dfs]
            dfs = [data_loader.apply_blacklist(df, 'storagedata', {'grid': exclude_grids}) for df in dfs]
            dfs = [data_loader.build_node_column(df) for df in dfs]       
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
        files = self.config.get('unitdata_files', [])
        if len(files) > 0:           
            dfs = excel_exchange.read_input_excels(input_folder, files, 'unitdata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'unitdata', self.logs) for df in dfs]     
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
        files = self.config.get('transferdata_files', [])
        if len(files) > 0:         
            dfs = excel_exchange.read_input_excels(input_folder, files, 'transferdata', self.logs)
            dfs = [data_loader.normalize_dataframe(df, 'transferdata', self.logs) for df in dfs]       
            dfs = [data_loader.apply_blacklist(df, 'transferdata', {'grid': exclude_grids}) for df in dfs]        
            dfs = [data_loader.build_from_to_columns(df) for df in dfs]               
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