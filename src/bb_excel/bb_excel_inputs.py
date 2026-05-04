from dataclasses import dataclass
from src.infrastructure.cache_manager import CacheManager
from src.infrastructure.logger import IterationLogger
from src.source_data.source_data_pipeline import SourceDataPipeline
from src.timeseries.timeseries_results import TimeseriesPipelineOutput
from pathlib import Path


@dataclass
class BBExcelInputs:
    # From sys args
    input_folder: Path

    # From currently looped run
    output_folder: Path
    scen_tags: list[str]

    # From config file
    config: dict

    # Pipeline components
    cache_manager: CacheManager
    logger: IterationLogger
    source_data: SourceDataPipeline
    ts_results: TimeseriesPipelineOutput