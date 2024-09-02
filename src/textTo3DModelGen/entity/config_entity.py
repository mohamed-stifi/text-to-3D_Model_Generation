from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    number_of_samples_to_download: int
    local_data_file: Path


@dataclass(frozen=True)
class DataRenderingConfig:
    local_data_file : Path
    render_script : Path
    output_dir : Path
    num_images : int
    resolution : int
    engine : str
    camera_dist: float


@dataclass(frozen=True)
class TextEmbeddingConfig:
    local_data_file: Path
    embedding_dir: Path
    model_name: str
    cache_dir: Path


@dataclass(frozen=True)
class DataSplitConfig:
    local_data_file: Path
    output_dir : Path
    train_ratio: float
    test_ratio: float
    val_ratio: float


    