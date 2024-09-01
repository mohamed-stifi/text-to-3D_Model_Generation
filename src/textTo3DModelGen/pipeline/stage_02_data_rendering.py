from textTo3DModelGen.config.configuration import ConfigurationManager
from textTo3DModelGen.components.data_rendering import DataRendering
from textTo3DModelGen import logger


STAGE_NAME = "Data Rendering stage"

class DataRenderingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_rendering_config = config.get_data_rendering_config()
        data_rendering = DataRendering(config= data_rendering_config)
        data_rendering.render_all()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = DataRenderingPipeline()
        obj.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
        logger.exception(e)
        raise e
