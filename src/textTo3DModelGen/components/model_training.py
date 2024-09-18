from textTo3DModelGen import logger
from textTo3DModelGen.dnnlib import  EasyDict
from textTo3DModelGen.utils.training_utils import *
from dataclasses import asdict
from textTo3DModelGen.metrics import metric_main
from textTo3DModelGen.entity.config_entity import TrainingModelConfig


class TrainingModel:
    def __init__(self, config: TrainingModelConfig):
        self.config = asdict(config)

    def train_step(self):
        try:
            logger.info(f"Start Initialize of Config..")
            c = EasyDict()  # Main config dict.
            kwargs = {}
            for _, value in self.config.items():
                kwargs.update(value)
            opts = EasyDict(kwargs)  # Command line arguments.
            c.G_kwargs = EasyDict(
                class_name=None, z_dim=opts.latent_dim,
                w_dim=opts.latent_dim,
                mapping_kwargs=EasyDict())
            c.D_kwargs = EasyDict(
                class_name='textTo3DModelGen.training.networks_get3d.Discriminator',
                block_kwargs=EasyDict(),
                mapping_kwargs=EasyDict(),
                epilogue_kwargs=EasyDict())
            
            c.G_opt_kwargs = EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
            c.D_opt_kwargs = EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
            c.loss_kwargs = EasyDict(class_name='textTo3DModelGen.training.loss.StyleGAN2Loss')

            c.data_loader_kwargs = EasyDict(pin_memory=True, prefetch_factor=2)
            c.inference_vis = opts.inference_vis

            # Training set.
            if opts.inference_vis:
                c.inference_to_generate_textured_mesh = opts.inference_to_generate_textured_mesh
                c.inference_save_interpolation = opts.inference_save_interpolation
                c.inference_compute_fid = opts.inference_compute_fid
                c.inference_generate_geo = opts.inference_generate_geo

            c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, opt=opts)
            if opts.cond and not c.training_set_kwargs.use_labels:
                logger.info(f"--cond is True but there is not label in dataset.")
                raise ValueError('--cond is True but there is not label in dataset')
            
            # c.training_set_kwargs.split = 'train' if opts.use_shapenet_split else 'all'
            if opts.inference_vis:
                c.training_set_kwargs.data_split_file = './artifacts/data_split/test.txt'
            c.training_set_kwargs.use_labels = opts.cond
            c.training_set_kwargs.xflip = False

            # Hyperparameters & settings.p
            c.G_kwargs.iso_surface = opts.iso_surface
            c.G_kwargs.one_3d_generator = opts.one_3d_generator
            c.G_kwargs.n_implicit_layer = opts.n_implicit_layer
            c.G_kwargs.deformation_multiplier = opts.deformation_multiplier
            c.resume_pretrain = opts.resume_pretrain
            c.D_reg_interval = opts.d_reg_interval
            c.G_kwargs.use_style_mixing = opts.use_style_mixing
            c.G_kwargs.dmtet_scale = opts.dmtet_scale
            c.G_kwargs.feat_channel = opts.feat_channel
            c.G_kwargs.mlp_latent_channel = opts.mlp_latent_channel
            c.G_kwargs.tri_plane_resolution = opts.tri_plane_resolution
            c.G_kwargs.n_views = opts.n_views
            c.G_kwargs.render_type = opts.render_type
            c.G_kwargs.use_tri_plane = opts.use_tri_plane
            c.D_kwargs.data_camera_mode = "mode_obejavers"
            c.D_kwargs.add_camera_cond = opts.add_camera_cond
            c.G_kwargs.tet_res = opts.tet_res

            c.G_kwargs.geometry_type = opts.geometry_type
            c.num_gpus = opts.gpus
            c.batch_size = opts.batch
            c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus

            # c.G_kwargs.geo_pos_enc = opts.geo_pos_enc
            c.G_kwargs.data_camera_mode = 'objaverse'
            c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
            c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
            c.G_kwargs.mapping_kwargs.num_layers = 8

            c.D_kwargs.architecture = opts.d_architecture
            c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
            c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
            c.loss_kwargs.gamma_mask = opts.gamma if opts.gamma_mask == 0.0 else opts.gamma_mask
            c.loss_kwargs.r1_gamma = opts.gamma
            c.loss_kwargs.lambda_flexicubes_surface_reg = opts.lambda_flexicubes_surface_reg
            c.loss_kwargs.lambda_flexicubes_weights_reg = opts.lambda_flexicubes_weights_reg
            c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
            c.D_opt_kwargs.lr = opts.dlr

            c.metrics = opts.metrics
            c.total_kimg = opts.kimg
            c.kimg_per_tick = opts.tick
            c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
            c.random_seed = c.training_set_kwargs.random_seed = opts.seed
            c.data_loader_kwargs.num_workers = opts.workers
            c.network_snapshot_ticks = 200

            # Sanity checks.
            if c.batch_size % c.num_gpus != 0:
                raise ValueError('--batch must be a multiple of number of gpus')
            if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
                raise ValueError('--batch must be a multiple of number of gpus times --batch-gpu')
            if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
                raise ValueError('--batch-gpu cannot be smaller than --mbstd')
            if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
                raise ValueError(
                    '\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
            
            # Base configuration.
            c.ema_kimg = c.batch_size * 10 / 32
            c.G_kwargs.class_name = 'textTo3DModelGen.training.networks_get3d.GeneratorDMTETMesh'
            c.loss_kwargs.style_mixing_prob = 0.9  # Enable style mixing regularization.
            c.loss_kwargs.pl_weight = 0.0  # Enable path length regularization.
            c.G_reg_interval = 4  # Enable lazy regularization for G.
            c.G_kwargs.fused_modconv_default = 'inference_only'  # Speed up training by using regular convolutions instead of grouped convolutions.
            # Performance-related toggles.
            if opts.fp32:
                c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
                c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
            if opts.nobench:
                c.cudnn_benchmark = False

            # Description string.
            desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
            if opts.desc is not None:
                desc += f'-{opts.desc}'

            logger.info(f"Description string {desc}")
            
            # Launch.
            logger.info('==> launch training')
            launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

        except Exception as e:
            raise e