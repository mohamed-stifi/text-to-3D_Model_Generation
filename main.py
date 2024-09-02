from textTo3DModelGen.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from textTo3DModelGen.pipeline.stage_02_data_rendering import DataRenderingPipeline
from textTo3DModelGen.pipeline.stage_03_text_embedding import TextEmbeddingPipeline
from textTo3DModelGen.pipeline.stage_04_data_split import DataSplitPipeline
from textTo3DModelGen import logger
import argparse


parser = argparse.ArgumentParser(description="Run different stages of the process.")
parser.add_argument("--stage", required=True, help="Specify the stage to run: \n1: run data ingestion pipeline.\n2: run data rendering pipeline.\n3: run text embedding pipeline.\
                    \n3: run data split pipeline.\n5: run all pipeline.")
args = parser.parse_args()

if __name__ == "__main__":
    if args.stage == "1":
        STAGE_NAME = "Data Ingestion stage"
        try:
            logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
            obj = DataIngestionPipeline()
            obj.main()
            logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
        except Exception as e:
            logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
            logger.exception(e)
            raise e

    elif args.stage == "2":
        STAGE_NAME = "Data Rendering stage"
        try:
            logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
            obj = DataRenderingPipeline()
            obj.main()
            logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
        except Exception as e:
            logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
            logger.exception(e)
            raise e
        
    elif args.stage == "3":
        STAGE_NAME = "Text Embedding stage"
        try:
            logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
            obj = TextEmbeddingPipeline()
            obj.main()
            logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
        except Exception as e:
            logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
            logger.exception(e)
            raise e
        
    elif args.stage == "4":
        STAGE_NAME = "Data Split stage"
        try:
            logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
            obj = DataSplitPipeline()
            obj.main()
            logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
        except Exception as e:
            logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
            logger.exception(e)
            raise e

    else:
        print(f"{args.stage} is an invalid stage, should be in [1,2,3,4,5]")


