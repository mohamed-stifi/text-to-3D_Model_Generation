from textTo3DModelGen.config.configuration import ConfigurationManager
from textTo3DModelGen.components.text_embedding import TextEmbedding
from textTo3DModelGen import logger


STAGE_NAME = "Text Embedding stage"

class TextEmbeddingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        text_embedding_config = config.get_text_embedding_config()
        text_embedding = TextEmbedding(config= text_embedding_config)
        text_embedding.embedding_all()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
        obj = TextEmbeddingPipeline()
        obj.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
    except Exception as e:
        logger.info(f">>>>>>>>> stage {STAGE_NAME} stoped -ERROR <<<<<<<<")
        logger.exception(e)
        raise e