from pathlib import Path
from src.data_loader import apply_whitelist, apply_blacklist
from src.data_loader import normalize_dataframe, merge_row_by_row
from src.data_loader import build_node_column, build_from_to_columns, build_unittype_unit_column
from src.data_loader import filter_nonzero_numeric_rows
from src.excel_exchange import read_input_excels
from src.utils import log_status
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

        self.df_demanddata = None
        self.df_transferdata = None
        self.df_unittypedata = None
        self.df_unitdata = None
        self.df_remove_units = None
        self.df_storagedata = None
        self.df_fueldata = None
        self.df_emissiondata = None

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
            dfs = read_input_excels(input_folder, files, 'unittypedata', self.logs)
            dfs = [normalize_dataframe(df, 'unittypedata', self.logs)
                   for df in dfs
                   ]
            dfs = [apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year}, self.logs, 'unittypedata')
                   for df in dfs
                   ]
            self.df_unittypedata = merge_row_by_row(dfs, self.logs, key_columns=['generator_id'])
        else:
            log_status(
                f"skipping Excel files for 'unittypedata_files': {len(files)} file(s)",
                self.logs, level="info"
            )
            self.df_unittypedata = pd.DataFrame()

        # fueldata
        files = self.config.get('fueldata_files', [])
        if len(files) > 0:
            dfs = read_input_excels(input_folder, files, 'fueldata', self.logs)
            dfs = [normalize_dataframe(df, 'fueldata', self.logs)
                   for df in dfs
                   ]
            dfs = [apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year}, self.logs, 'fueldata')
                   for df in dfs
                   ]
            self.df_fueldata = merge_row_by_row(dfs, self.logs, key_columns=['grid'])
        else:
            log_status(
                f"skipping Excel files for 'fueldata_files': {len(files)} file(s)",
                self.logs, level="info"
            )            
            self.df_fueldata = pd.DataFrame()            

        # emissiondata
        files = self.config.get('emissiondata_files', [])
        if len(files) > 0:        
            dfs = read_input_excels(input_folder, files, 'emissiondata', self.logs)
            dfs = [normalize_dataframe(df, 'emissiondata', self.logs)
                   for df in dfs
                   ]
            dfs = [apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year}, self.logs, 'emissiondata')
                   for df in dfs
                   ]
            self.df_emissiondata = merge_row_by_row(dfs, self.logs, key_columns=['emission'])
        else:
            log_status(
                f"skipping Excel files for 'emissiondata_files': {len(files)} file(s)",
                self.logs, level="info"
            )                   
            self.df_emissiondata = pd.DataFrame()    

        # --- country-level datasets ---
        exclude_grids = self.config.get('exclude_grids', [])
        exclude_nodes = self.config.get('exclude_nodes', [])

        # demanddata
        files = self.config.get('demanddata_files', [])
        if len(files) > 0:           
            dfs = read_input_excels(input_folder, files, 'demanddata', self.logs)
            dfs = [normalize_dataframe(df, 'demanddata', self.logs)
                   for df in dfs
                   ]
            dfs = [apply_blacklist(df, 'demanddata', {'grid': exclude_grids})
                   for df in dfs
                   ]
            dfs = [build_node_column(df)
                   for df in dfs
                   ]       
            dfs = [apply_blacklist(df, 'demanddata', {'node': exclude_nodes})
                   for df in dfs
                   ]            
            dfs = [apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes}, 
                                   self.logs, 'demanddata')
                   for df in dfs
                   ]
            self.df_demanddata = merge_row_by_row(dfs, self.logs, key_columns=['country', 'grid', 'node'])
            self.df_demanddata = filter_nonzero_numeric_rows(self.df_demanddata, exclude=['year'])
        else:
            log_status(
                f"skipping Excel files for 'demanddata_files': {len(files)} file(s)",
                self.logs, level="info"
            )                
            self.df_demanddata = pd.DataFrame()             

        # storagedata
        files = self.config.get('storagedata_files', [])
        if len(files) > 0:            
            dfs = read_input_excels(input_folder, files, 'storagedata', self.logs)
            dfs = [normalize_dataframe(df, 'storagedata', self.logs)
                   for df in dfs
                   ]
            dfs = [apply_blacklist(df, 'storagedata', {'grid': exclude_grids})
                   for df in dfs
                   ]
            dfs = [build_node_column(df)
                   for df in dfs
                   ]       
            dfs = [apply_blacklist(df, 'storagedata', {'node': exclude_nodes})
                   for df in dfs
                   ]            
            dfs = [apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes}, 
                                   self.logs, 'storagedata')
                   for df in dfs
                   ]
            self.df_storagedata = merge_row_by_row(dfs, self.logs, key_columns=['country', 'grid', 'node'])
        else:
            log_status(
                f"skipping Excel files for 'storagedata_files': {len(files)} file(s)",
                self.logs, level="info"
            )                       
            self.df_storagedata = pd.DataFrame()             

        # unitdata
        files = self.config.get('unitdata_files', [])
        if len(files) > 0:           
            dfs = read_input_excels(input_folder, files, 'unitdata', self.logs)
            dfs = [normalize_dataframe(df, 'unitdata', self.logs)
                   for df in dfs
                   ]     
            dfs = [build_unittype_unit_column(df, self.df_unittypedata, self.logs)
                   for df in dfs
                   ]           
            dfs = [apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes}, 
                                   self.logs, 'unitdata')
                   for df in dfs
                   ]
            self.df_unitdata = merge_row_by_row(dfs, self.logs, key_columns=['country', 'generator_id', 'unit_name_prefix'])
        else:
            log_status(
                f"skipping Excel files for 'unitdata_files': {len(files)} file(s)",
                self.logs, level="info"
            )                    
            self.df_unitdata = pd.DataFrame()             

        # transferdata
        files = self.config.get('transferdata_files', [])
        if len(files) > 0:         
            dfs = read_input_excels(input_folder, files, 'transferdata', self.logs)
            dfs = [normalize_dataframe(df, 'transferdata', self.logs)
                   for df in dfs
                   ]       
            dfs = [apply_blacklist(df, 'transferdata', {'grid': exclude_grids})
                   for df in dfs
                   ]        
            dfs = [build_from_to_columns(df)
                   for df in dfs
                   ]               
            dfs = [apply_blacklist(df, 'transferdata', {'from_node': exclude_nodes})
                   for df in dfs
                   ]  
            dfs = [apply_blacklist(df, 'transferdata', {'to_node': exclude_nodes})
                   for df in dfs
                   ]  
            dfs = [apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'from': self.country_codes}, 
                                   self.logs, 'transferdata')
                   for df in dfs
                   ]
            dfs = [apply_whitelist(df, {'scenario':scen_and_alt, 'year':self.scenario_year, 'to': self.country_codes}, 
                                   self.logs, 'transferdata')
                   for df in dfs
                   ]     
            self.df_transferdata = merge_row_by_row(dfs, self.logs, key_columns=['from', 'from_suffix', 'to', 'to_suffix', 'grid'])
        else:
            log_status(
                f"skipping Excel files for 'transferdata_files': {len(files)} file(s)",
                self.logs, level="info"
            )                
            self.df_transferdata = pd.DataFrame()  