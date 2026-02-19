import json
import logging.config
from pathlib import Path
import numpy as np

## Numpy print options
np.set_printoptions(precision=4, suppress=True, sign=" ")
## Global logger
logger = logging.getLogger("application")


def setup_logging_from_json(config_file: Path, log_output_file: Path):
    assert config_file.exists(), f"Config file {config_file} does not exist"

    log_output_file.parent.mkdir(parents=True, exist_ok=True)
    logger_config_dict = json.load(open(config_file))
    logger_config_dict["handlers"]["file_handler"]["filename"] = str(log_output_file)
    logging.config.dictConfig(logger_config_dict)


