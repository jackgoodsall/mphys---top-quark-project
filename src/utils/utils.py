import yaml
from pydantic import BaseModel
from typing import Dict, Any

## Define some config models so that we have a first layer file 
## on passing bricked configs
class DataConfig(BaseModel):
    train: Dict[str, Any]
    validation: Dict[str, Any]
    test: Dict[str, Any]
### Can add even further structued configs if want
class BaseConfig(BaseModel):
    data: DataConfig
    model_parameters: Dict[str, Any]
    train: Dict[str, Any]
    data_pipeline: Dict[str, Any]

def load_and_split_config(config_input_file: str) -> BaseConfig:

    with open(config_input_file, "r") as f:
        raw_config = yaml.safe_load(f)
    cfg = BaseConfig(**raw_config)
    return cfg

def load_any_config(config_input_file: str) -> Dict[str: Any]:
    with open(config_input_file, "r") as f:
        raw_config = yaml.safe_load(f)
    return f