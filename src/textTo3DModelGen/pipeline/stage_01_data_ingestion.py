from textTo3DModelGen.config.configuration import ConfigurationManager
from textTo3DModelGen.components.data_ingestion import DataIngestion
from textTo3DModelGen import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config= data_ingestion_config)
        data_ingestion.download_data()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
        logger.exception(e)
        raise e