from textTo3DModelGen import logger
from textTo3DModelGen.utils.common import get_size, load_from_url, save_csv, create_directories, load_from_objaverse
import multiprocessing
from textTo3DModelGen.entity.config_entity import DataIngestionConfig
import objaverse



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        try:
            root_dir = self.config.root_dir
            create_directories([root_dir])
            source_url = self.config.source_url
            num_of_samples = self.config.number_of_samples_to_download
            local_data_file = self.config.local_data_file
            
            descriptions = load_from_url(url= source_url, num_of_samples= num_of_samples)
            logger.info(f"Downloaded description from {source_url} with lenght {len(descriptions)}.")

            processes = multiprocessing.cpu_count()
            objects = load_from_objaverse(uids = descriptions['uids'], processes = processes)
            logger.info(f"Downloaded {len(objects)} objects from Objaverse Dataset.")

            paths = objaverse._load_object_paths()
            saved_path = ['/root/.objaverse/hf-objaverse-v1/' + str(paths[uid]) for uid in descriptions['uids']]
            descriptions['saved_path'] = saved_path
            save_csv(local_data_file, descriptions)
            logger.info(f"saved objects with description data into {local_data_file} with size {get_size(local_data_file)}.")
        except Exception as e:
            raise e