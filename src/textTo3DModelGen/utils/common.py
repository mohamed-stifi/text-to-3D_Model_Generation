import os
from box.exceptions import BoxValueError
import yaml
from textTo3DModelGen import logger
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import torch.nn as nn
import torch.optim as optim
import trimesh
import pandas as pd
import requests
from io import StringIO
import objaverse


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as file:
            content = yaml.safe_load(file)
            logger.info(
                f"yaml file: {path_to_yaml} loaded successfully"
            )
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_dirs: list, verbose=True):
    for path in path_to_dirs:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_csv(path, data: pd.DataFrame):
    # Save the DataFrame to the specified path as a CSV
    data.to_csv(path, index=False)

@ensure_annotations
def load_csv(path: Path) -> ConfigBox:
    # Read the CSV file using pandas
    df = pd.read_csv(path)
    
    # Convert the DataFrame to a dictionary
    data_dict = df.to_dict(orient='list')
    
    # Convert the dictionary to a ConfigBox for structured access
    return ConfigBox(data_dict)

@ensure_annotations
def load_from_url(url: str, num_of_samples: int) -> pd.DataFrame:
    # Download the content from the URL
    response = requests.get(url)
    #print('get response from requestes')
    # Convert the content to a pandas DataFrame
    csv_data = StringIO("uids, description\n"+response.text)
    df = pd.read_csv(csv_data)
    
    return df[:num_of_samples]


def load_from_objaverse(uids, processes: int) -> pd.DataFrame:
    objects = objaverse.load_objects(
        uids=uids,
        download_processes=processes
    )
    objects = pd.DataFrame(objects, columns=['uids', 'path'])
    return objects

@ensure_annotations
def save_model(path: Path, model: nn.Module, optimizer: optim.Adam, info: dict):
    pass

@ensure_annotations
def load_model(path: Path, model: nn.Module, optimizer: optim.Adam) -> ConfigBox:
    pass

@ensure_annotations
def save_glb_mesh(path: Path, mesh):
    pass

@ensure_annotations
def load_glb_mesh(glb_file_path: str):
    mesh = trimesh.load(glb_file_path)
    logger.info(f"glb mesh file loaded from: {glb_file_path.split()[-1]}")
    return mesh


@ensure_annotations
def get_size(path: Path):
    size_in_mb = round(os.path.getsize(path)/1024**2)
    return f"~ {size_in_mb} MB"