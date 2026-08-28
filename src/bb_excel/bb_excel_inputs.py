from dataclasses import dataclass
from src.infrastructure.cache_manager import CacheManager
from src.infrastructure.logger import IterationLogger
from src.source_data.source_data_pipeline import SourceDataPipeline
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

    # Pipeline components.
    #
    # source_data is the only data channel: the timeseries phase has already
    # merged whatever it had to contribute into those frames, so the builder
    # reads one set of tables and never asks which stage produced a row.
    cache_manager: CacheManager
    logger: IterationLogger
    source_data: SourceDataPipeline