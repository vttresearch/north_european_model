from dataclasses import dataclass
from pathlib import Path
from src.infrastructure.cache_manager import CacheManager
from src.infrastructure.logger import IterationLogger
from src.source_data.source_data_pipeline import SourceDataPipeline


@dataclass
class TimeseriesPipelineInputs:
    config: dict
    input_folder: Path
    output_folder: Path
    cache_manager: CacheManager
    source_data_pipeline: SourceDataPipeline
    logger: IterationLogger
    scenario_year: int | None = None
