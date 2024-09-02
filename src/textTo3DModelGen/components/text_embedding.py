from textTo3DModelGen import logger
from textTo3DModelGen.entity.config_entity import TextEmbeddingConfig
from textTo3DModelGen.utils.common import create_directories
import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import torch

class TextEmbedding:
    def __init__(self, config: TextEmbeddingConfig):
        self.config = config
        
    # Function to convert text to latent vectors
    def text_to_latent_vector(self, texts, model, processor):
        # Preprocess the texts
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        # Forward pass through the model
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
        return outputs

    def embedding_all(self):
        try:
            data = pd.read_csv(self.config.local_data_file)
            logger.info(f"Read of data from {self.config.local_data_file} is done.")

            model = CLIPModel.from_pretrained(self.config.model_name, cache_dir=self.config.cache_dir)
            logger.info(f"Load CLIP Model {self.config.model_name} to {self.config.cache_dir} is done.")

            processor = CLIPProcessor.from_pretrained(self.config.model_name, cache_dir=self.config.cache_dir)
            logger.info(f"Load CLIP Processor {self.config.model_name} to {self.config.cache_dir} is done.")

            for index, row in data.iterrows():
                model_id = row['uids']
                description = row[' description']

                # Ensure the model_id folder exists in the embedding directory
                model_embedding_dir = os.path.join(self.config.embedding_dir, model_id)
                create_directories([model_embedding_dir])

                text_embedding = self.text_to_latent_vector(description, model, processor)

                # Save the embedding
                path_to_save_emb = os.path.join(model_embedding_dir, 'condition.pt')
                torch.save(text_embedding.squeeze(0), path_to_save_emb)
                logger.info(f"Text embedding of model {model_id} is saved to {path_to_save_emb}.")
        except Exception as e:
            raise e