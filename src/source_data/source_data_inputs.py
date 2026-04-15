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
    scenario_alternative: str = ""
    scenario_alternative2: str = ""
    scenario_alternative3: str = ""
    scenario_alternative4: str = ""
