from textTo3DModelGen.config.configuration import ConfigurationManager
from textTo3DModelGen.components.data_split import DataSplit
from textTo3DModelGen import logger


STAGE_NAME = "Data Split stage"

class DataSplitPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_split_config = config.get_data_split_config()
        data_split = DataSplit(config= data_split_config)
        data_split.split_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = DataSplitPipeline()
        obj.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
        logger.exception(e)
        raise e