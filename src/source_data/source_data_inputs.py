from dataclasses import dataclass
from pathlib import Path
from src.infrastructure.logger import IterationLogger


@dataclass
class SourceDataPipelineInputs:
    config: dict
    input_folder: Path
    scenario: str
    scenario_year: int
    country_codes: list[str]
    logger: IterationLogger
    #: The build of config['variants'] to choose variant options by; '' is the
    #: build with no folder suffix.
    variant: str = ""
