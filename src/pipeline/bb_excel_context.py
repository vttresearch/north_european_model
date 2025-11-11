# src/pipeline/bb_excel_context.py

from dataclasses import dataclass
from src.pipeline.cache_manager import CacheManager
from src.pipeline.source_excel_data_pipeline import SourceExcelDataPipeline
from src.pipeline.timeseries_pipeline import TimeseriesRunResult
from pathlib import Path

@dataclass
class BBExcelBuildContext:
    # From sys args
    input_folder: Path
    
    # From currently looped run
    output_folder: Path
    scen_tags: list[str]

    # From config file
    config: dict

    # Pipeline components
    cache_manager: CacheManager
    source_data: SourceExcelDataPipeline
    ts_results: TimeseriesRunResult  