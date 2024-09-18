from textTo3DModelGen.constants import *
from textTo3DModelGen.utils.common import read_yaml, create_directories
from textTo3DModelGen.entity.config_entity import (DataIngestionConfig,
                                                   DataRenderingConfig,
                                                   TextEmbeddingConfig,
                                                   DataSplitConfig)

from textTo3DModelGen.entity.config_entity import (TrainingModelConfig,
                                                   StyleGAN2Config,
                                                   InferenceConfig,
                                                   DatasetConfig,
                                                   GeneratorConfig,
                                                   LossAndDiscriminatorConfig,
                                                   FeaturesConfig,
                                                   HyperparametersConfig,
                                                   SettingsConfig)


class ConfigurationManager:
    def __init__(
            self, 
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = HYPER_PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.params = self.params.model_hyperprams

        create_directories([
            self.config.artifacts_root
        ])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([
            config.root_dir
        ])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_url= config.source_url,
            number_of_samples_to_download= config.number_of_samples_to_download, 
            local_data_file= Path(config.local_data_file)
        )

        return data_ingestion_config
    

    def get_data_rendering_config(self) -> DataRenderingConfig:
        config = self.config.data_rendering

        create_directories([
            config.output_dir
        ])

        data_rendering_config = DataRenderingConfig(
            local_data_file = config.local_data_file,
            render_script = config.render_script,
            output_dir = config.output_dir,
            num_images = config.num_images,
            resolution = config.resolution,
            engine = config.engine,
            camera_dist = config.camera_dist
        )

        return data_rendering_config


    def get_text_embedding_config(self) -> TextEmbeddingConfig:
        config = self.config.text_embedding

        text_embedding_config = TextEmbeddingConfig(
            local_data_file = config.local_data_file,
            embedding_dir = config.embedding_dir,
            model_name = config.model_name,
            cache_dir = config.cache_dir
        )

        return text_embedding_config
    

    def get_data_split_config(self) -> DataSplitConfig:
        config = self.config.data_split

        create_directories([
            config.output_dir
        ])

        data_split_config = DataSplitConfig(
            local_data_file= config.local_data_file,
            output_dir= config.output_dir,
            train_ratio= config.train_ratio,
            test_ratio= config.test_ratio,
            val_ratio= config.val_ratio
        )

        return data_split_config
    

    def get_training_model_config(self) -> TrainingModelConfig:
        config = self.config.training_prams
        self.styleGAN2 = self.params.StyleGAN2
        self.inference = self.params.inference
        self.dataset = self.params.dataset
        self.generator = self.params.generator
        self.loss_and_discriminator = self.params.loss_and_discriminator
        self.features = self.params.features
        self.hyperparameters = self.params.hyperparameters
        self.settings = self.params.settings

        create_directories([
            config.outdir  # , config.desc
        ])

        styleGAN2= StyleGAN2Config(
            outdir = config.outdir,
            cfg = self.styleGAN2.cfg,
            gpus = config.gpus,
            batch = config.batch,
            gamma = config.gamma
        )
        inference= InferenceConfig(
            resume_pretrain = self.inference.resume_pretrain,
            inference_vis = self.inference.inference_vis,
            inference_to_generate_textured_mesh = self.inference.inference_to_generate_textured_mesh,
            inference_save_interpolation = self.inference.inference_save_interpolation,
            inference_compute_fid = self.inference.inference_compute_fid,
            inference_generate_geo = self.inference.inference_generate_geo
        )
        dataset= DatasetConfig(
            data = config.data,
            img_res = self.dataset.img_res,
            data_split_file = config.data_split_file,
            use_labels = self.dataset.use_labels
        )
        generator= GeneratorConfig(
            iso_surface = self.generator.iso_surface,
            use_style_mixing = self.generator.use_style_mixing,
            one_3d_generator = config.one_3d_generator,
            dmtet_scale = config.dmtet_scale,
            n_implicit_layer = self.generator.n_implicit_layer,
            feat_channel = self.generator.feat_channel,
            mlp_latent_channel = self.generator.mlp_latent_channel,
            deformation_multiplier = self.generator.deformation_multiplier,
            tri_plane_resolution = self.generator.tri_plane_resolution,
            n_views = self.generator.n_views,
            use_tri_plane = self.generator.use_tri_plane,
            tet_res = self.generator.tet_res,
            latent_dim = self.generator.latent_dim,
            geometry_type = self.generator.geometry_type,
            render_type = self.generator.render_type
        )
        loss_and_discriminator= LossAndDiscriminatorConfig(
            d_architecture = self.loss_and_discriminator.d_architecture,
            use_pl_length = self.loss_and_discriminator.use_pl_length,
            gamma_mask =self.loss_and_discriminator.gamma_mask,
            d_reg_interval = self.loss_and_discriminator.d_reg_interval,
            add_camera_cond = self.loss_and_discriminator.add_camera_cond,
            lambda_flexicubes_surface_reg = self.loss_and_discriminator.lambda_flexicubes_surface_reg,
            lambda_flexicubes_weights_reg =self.loss_and_discriminator.lambda_flexicubes_weights_reg
        )
        features= FeaturesConfig(
            cond = self.features.cond,
            freezed =self.features.freezed
        )
        hyperparameters= HyperparametersConfig(
            batch_gpu = self.hyperparameters.batch_gpu,
            cbase =self.hyperparameters.cbase,
            cmax =self.hyperparameters.cmax,
            glr = self.hyperparameters.glr,
            dlr = self.hyperparameters.dlr,
            map_depth = self.hyperparameters.map_depth,
            mbstd_group =self.hyperparameters.mbstd_group
        )
        settings= SettingsConfig(
            desc= config.desc ,
            metrics=self.settings.metrics,
            kimg= self.settings.kimg ,
            tick=self.settings.tick ,
            snap=self.settings.snap ,
            seed=self.settings.seed,
            fp32=self.settings.fp32 ,
            nobench=self.settings.nobench,
            workers=self.settings.workers ,
            dry_run = self.settings.dry_run,

        )

        training_model_config = TrainingModelConfig(
            styleGAN2 ,
            inference,
            dataset,
            generator,
            loss_and_discriminator,
            features,
            hyperparameters,
            settings,
        )

        return training_model_config
   