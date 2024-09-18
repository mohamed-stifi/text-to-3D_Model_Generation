from textTo3DModelGen.config.configuration import ConfigurationManager
from textTo3DModelGen.components.model_training import TrainingModel
from textTo3DModelGen import logger


STAGE_NAME = "Training Model stage"

class TrainingModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_model_config = config.get_training_model_config()
        training_model = TrainingModel(config= training_model_config)
        training_model.train_step()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = TrainingModelPipeline()
        obj.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
        logger.exception(e)
        raise e