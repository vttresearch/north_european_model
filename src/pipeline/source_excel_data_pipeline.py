from pathlib import Path
from src.data_loader import process_dataset
from src.data_loader import filter_df_blacklist, filter_df_whitelist, keep_last_occurance
from src.data_loader import build_node_column, build_from_to_columns, build_unittype_unit_column
from src.data_loader import filter_nonzero_numeric_rows



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
        # Make a combination of scenario and alternative
        scen_and_alt = [self.scenario]
        if self.scenario_alternative:
            scen_and_alt.append(self.scenario_alternative)


        # Load global datasets
        self.df_unittypedata = process_dataset(self.data_folder, self.config.get('unittypedata_files', []), 'unittypedata')
        self.df_fueldata     = process_dataset(self.data_folder, self.config.get('fueldata_files', []), 'fueldata')
        self.df_emissiondata = process_dataset(self.data_folder, self.config.get('emissiondata_files', []), 'emissiondata')

        # Process to get the data for the current scenario and year.
        self.df_unittypedata = filter_df_whitelist(self.df_unittypedata, 'unittypedata_files', {'scenario':scen_and_alt, 'year':self.scenario_year})
        self.df_fueldata     = filter_df_whitelist(self.df_fueldata, 'fueldata_files', {'scenario':scen_and_alt, 'year':self.scenario_year})
        self.df_emissiondata = filter_df_whitelist(self.df_emissiondata, 'emissiondata_files', {'scenario':scen_and_alt, 'year':self.scenario_year})

        # remove duplicates. Keep the last value. This implicitly handles overwriting earlier values with the latest.
        self.df_unittypedata = keep_last_occurance(self.df_unittypedata, ['generator_id'])
        self.df_fueldata     = keep_last_occurance(self.df_fueldata, ['fuel'])
        self.df_emissiondata = keep_last_occurance(self.df_emissiondata, ['emission'])


        # Load country specific input files
        self.df_demanddata   = process_dataset(self.data_folder, self.config.get('demanddata_files', []), 'demanddata')
        self.df_transferdata = process_dataset(self.data_folder, self.config.get('transferdata_files', []), 'transferdata')
        self.df_unitdata     = process_dataset(self.data_folder, self.config.get('unitdata_files', []), 'unitdata')
        self.df_remove_units = process_dataset(self.data_folder, self.config.get('unitdata_files', []), 'remove', isMandatory=False)
        self.df_storagedata  = process_dataset(self.data_folder, self.config.get('storagedata_files', []), 'storagedata')

        # Exclude grids
        exclude_grids = self.config.get('exclude_grids', [])
        self.df_demanddata =   filter_df_blacklist(self.df_demanddata, 'demand_files', {'grid': exclude_grids})
        self.df_transferdata = filter_df_blacklist(self.df_transferdata, 'transferdata_files', {'grid': exclude_grids})
        self.df_storagedata =  filter_df_blacklist(self.df_storagedata, 'storagedata_files', {'grid': exclude_grids})

        # Build node columns
        self.df_demanddata =   build_node_column(self.df_demanddata)
        self.df_storagedata =  build_node_column(self.df_storagedata)
        self.df_transferdata = build_from_to_columns(self.df_transferdata)

        # Exclude nodes
        exclude_nodes = self.config.get('exclude_nodes', [])
        self.df_demanddata =   filter_df_blacklist(self.df_demanddata, 'demand_files', {'node': exclude_nodes})
        self.df_storagedata =  filter_df_blacklist(self.df_storagedata, 'storage_files', {'node': exclude_nodes})
        self.df_transferdata = filter_df_blacklist(self.df_transferdata, 'transferdata_files', {'from_node': exclude_nodes})
        self.df_transferdata = filter_df_blacklist(self.df_transferdata, 'transferdata_files', {'to_node': exclude_nodes})

        # Build unittype and unit columns
        self.df_unitdata =     build_unittype_unit_column(self.df_unitdata, self.df_unittypedata, self.logs)
        self.df_remove_units = build_unittype_unit_column(self.df_remove_units, self.df_unittypedata, self.logs)

        # Process to get the data for the current scenario, year, and country.
        self.df_demanddata   = filter_df_whitelist(self.df_demanddata, 'demanddata_files', {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes})
        self.df_transferdata = filter_df_whitelist(self.df_transferdata, 'transferdata_files', {'scenario':scen_and_alt, 'year':self.scenario_year, 'from': self.country_codes})
        self.df_transferdata = filter_df_whitelist(self.df_transferdata, 'transferdata_files', {'scenario':scen_and_alt, 'year':self.scenario_year, 'to': self.country_codes})
        self.df_unitdata     = filter_df_whitelist(self.df_unitdata, 'unitdata_files', {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes})
        self.df_remove_units = filter_df_whitelist(self.df_remove_units, 'remove_files', {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes})
        self.df_storagedata  = filter_df_whitelist(self.df_storagedata, 'storagedata_files', {'scenario':scen_and_alt, 'year':self.scenario_year, 'country': self.country_codes})

        # remove duplicates. Keep the last value. This implicitly handles overwriting earlier values with the latest.
        self.df_demanddata   = keep_last_occurance(self.df_demanddata, ['country', 'grid', 'node'])
        self.df_transferdata = keep_last_occurance(self.df_transferdata, ['from', 'from_suffix', 'to', 'to_suffix', 'grid'])
        self.df_unitdata     = keep_last_occurance(self.df_unitdata, ['country', 'generator_id', 'unit_name_prefix'])
        self.df_storagedata  = keep_last_occurance(self.df_storagedata, ['country', 'grid', 'node'])

        # Remove zero rows from demanddata
        self.df_demanddata = filter_nonzero_numeric_rows(self.df_demanddata, exclude=['year'])
