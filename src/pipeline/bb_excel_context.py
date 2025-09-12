# src/pipeline/excel_context.py

from dataclasses import dataclass
from src.pipeline.cache_manager import CacheManager
from pathlib import Path
import pandas as pd

@dataclass
class BBExcelBuildContext:
    # From sys args
    input_folder: Path
    
    # From currently looped run
    output_folder: Path
    scen_tags: list[str]

    # From config file
    config: dict

    # Cache manager
    cache_manager: CacheManager

    # DataFrames from InputDataPipeline
    df_demanddata: pd.DataFrame
    df_transferdata: pd.DataFrame
    df_unittypedata: pd.DataFrame
    df_unitdata: pd.DataFrame
    df_storagedata: pd.DataFrame
    df_fueldata: pd.DataFrame
    df_emissiondata: pd.DataFrame

    # From TimeseriesPipeline
    secondary_results: dict
    ts_domains: dict
    ts_domain_pairs: dict