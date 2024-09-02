from textTo3DModelGen import logger
from textTo3DModelGen.utils.common import save_list_to_textfile
from textTo3DModelGen.entity.config_entity import DataSplitConfig
import random
import pandas as pd
import os

class DataSplit:
    def __init__(self, config: DataSplitConfig):
        self.config = config

    def split_data(self):
        try:
            data = pd.read_csv(self.config.local_data_file)
            logger.info(f"Read of data from {self.config.local_data_file} is done.")

            uids = data["uids"].to_list()

            random.shuffle(uids)
            logger.info(f"Shuffle the list of uids is done.")

            # Compute the indices for splitting
            train_size = int(self.config.train_ratio * len(uids))
            val_size = int(self.config.val_ratio * len(uids))
            test_size = len(uids) - train_size - val_size 
            logger.info(f"train_size: {train_size} || val_size: {val_size} || test_size: {test_size}.")

            # Split the list
            train_uids = uids[:train_size]
            val_uids = uids[train_size:train_size + val_size]
            test_uids = uids[train_size + val_size:]

            train_filename = os.path.join(self.config.output_dir, "train.txt")
            test_filename = os.path.join(self.config.output_dir, "test.txt")
            val_filename = os.path.join(self.config.output_dir, "val.txt")

            # Save the train, validation, and test IDs to text files
            save_list_to_textfile(train_filename, train_uids, "train_uids")
            save_list_to_textfile(test_filename, test_uids, "test_uids")
            save_list_to_textfile(val_filename, val_uids, "val_uids")
        except Exception as e:
            raise e