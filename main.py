from textTo3DModelGen.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from textTo3DModelGen import logger
import argparse


parser = argparse.ArgumentParser(description="Run different stages of the process.")
parser.add_argument("--stage", required=True, help="Specify the stage to run: \n1: run data ingestion pipeline.\n2: run data validation pipeline.\n3: run data loader pipeline.\n4: run all pipeline.")
args = parser.parse_args()


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
    
else:
    print(f"{args.stage} is an invalid stage, should be in [1,2,3,4]")


