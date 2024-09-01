from textTo3DModelGen import logger
import pandas as pd
from textTo3DModelGen.entity.config_entity import DataRenderingConfig
import os

class DataRendering:
    def __init__(self, config: DataRenderingConfig):
        self.config = config
        self.render_cmd = f"blender -b -P {self.config.render_script} -- \
                            --object_path %s \
                            --output_dir {self.config.output_dir} \
                            --engine {self.config.engine} \
                            --num_images {self.config.num_images} \
                            --camera_dist {self.config.camera_dist} \
                            --resolution {self.config.resolution}"

    def render_all(self):
        try:
            data = pd.read_csv(self.config.local_data_file)
            logger.info(f"Read of data from {self.config.local_data_file} is done.")

            for obj in data["saved_path"]:
                file_path = obj.replace("/root", "C:/Users/lenovo")
                if os.path.exists(file_path):
                    logger.info(f"Start rendering of the model {os.path.basename(file_path)}.")
                    os.system(self.render_cmd%(file_path))
                    logger.info(f"End rendering of the model {os.path.basename(file_path)}.")
                else:
                    logger.info(f"The file {file_path} does not exist.")
        except Exception as e:
            raise e