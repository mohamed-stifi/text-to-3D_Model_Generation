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


@dataclass(frozen=True)
class StyleGAN2Config:
    outdir: Path
    cfg: str
    gpus: int
    batch: int
    gamma: float

@dataclass(frozen=True)
class InferenceConfig:
    ### Configs for inference
    resume_pretrain: Path
    inference_vis: bool
    inference_to_generate_textured_mesh: bool
    inference_save_interpolation: bool
    inference_compute_fid: bool
    inference_generate_geo: bool

@dataclass(frozen=True)
class DatasetConfig:
    ### Configs for dataset
    data: Path
    img_res: int
    data_split_file: Path
    use_labels: bool

@dataclass(frozen=True)
class GeneratorConfig:
    ### Configs for 3D generator
    iso_surface: str
    use_style_mixing: bool
    one_3d_generator: bool
    dmtet_scale: float
    n_implicit_layer: int
    feat_channel: int
    mlp_latent_channel: int
    deformation_multiplier: float
    tri_plane_resolution: int
    n_views: int
    use_tri_plane: bool
    tet_res: int
    latent_dim: int
    geometry_type: str
    render_type: str
@dataclass(frozen=True)
class LossAndDiscriminatorConfig:
    ### Configs for training loss and discriminator#
    d_architecture: str
    use_pl_length: bool
    gamma_mask: float
    d_reg_interval: int
    add_camera_cond: bool
    lambda_flexicubes_surface_reg: float
    lambda_flexicubes_weights_reg: float
@dataclass(frozen=True)
class FeaturesConfig:
    # Optional features.
    cond: bool
    freezed: int

@dataclass(frozen=True)
class HyperparametersConfig:
    # hyperparameters:
    batch_gpu: int
    cbase: int
    cmax: int
    glr: float
    dlr: float
    map_depth: int
    mbstd_group: int
@dataclass(frozen=True)
class SettingsConfig:
    # settings:
    desc: str
    metrics: list
    kimg: int
    tick: int
    snap: int
    seed: int
    fp32: bool
    nobench: bool
    workers: int
    dry_run: bool

@dataclass(frozen=True)
class TrainingModelConfig:
    styleGAN2: StyleGAN2Config
    inference: InferenceConfig
    dataset: DatasetConfig
    generator: GeneratorConfig
    loss_and_discriminator: LossAndDiscriminatorConfig
    features: FeaturesConfig
    hyperparameters: HyperparametersConfig
    settings: SettingsConfig
      