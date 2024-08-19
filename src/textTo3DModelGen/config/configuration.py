from textTo3DModelGen.constants import *
from textTo3DModelGen.utils.common import read_yaml, create_directories
from textTo3DModelGen.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
            self, 
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = HYPER_PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([
            self.config.artifacts_root
        ])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([
            config.root_dir
        ])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_url= config.source_url,
            number_of_samples_to_download= config.number_of_samples_to_download, 
            local_data_file= Path(config.local_data_file)
        )

        return data_ingestion_config
