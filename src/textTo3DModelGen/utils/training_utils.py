from textTo3DModelGen import dnnlib
import os
import re
import json
from textTo3DModelGen import logger
import torch
import tempfile
from textTo3DModelGen.utils.torch_utils import training_stats
from textTo3DModelGen.utils.torch_utils import custom_ops
from textTo3DModelGen.training import inference_3d
from textTo3DModelGen.training import training_loop_3d


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(
                backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(
                backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    if c.inference_vis:
        inference_3d.inference(rank=rank, **c)
    # Execute training loop.
    else:
        training_loop_3d.training_loop(rank=rank, **c)

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    if c.inference_vis:
        c.run_dir = os.path.join(outdir, 'inference')
    else:
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    
    logger.info('Training options:')
    logger.info(json.dumps(c, indent=2))
    
    logger.info(f'Output directory:    {c.run_dir}')
    logger.info(f'Number of GPUs:      {c.num_gpus}')
    logger.info(f'Batch size:          {c.batch_size} images')
    logger.info(f'Training duration:   {c.total_kimg} kimg')
    logger.info(f'Dataset path:        {c.training_set_kwargs.path}')
    logger.info(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    logger.info(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    logger.info(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    logger.info(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    

    # Dry run?
    if dry_run:
        logger.info('Dry run; exiting.')
        return

    # Create output directory.
    if not os.path.exists(c.run_dir):
        logger.info(f'Creating output directory...{c.run_dir}')
        os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        logger.info(f"Save the training option to {os.path.join(c.run_dir, 'training_options.json')}")
        json.dump(c, f, indent=2)

    # Launch processes.
    logger.info('Launching processes...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

def init_dataset_kwargs(data, opt=None):
    try:
        dataset_kwargs = dnnlib.EasyDict(
            class_name='textTo3DModelGen.training.dataset.ImageFolderDataset',
            path=data,
            data_split_file= opt.data_split_file,
            use_labels=opt.use_labels, 
            max_size=None,
            xflip=False,
            resolution=opt.img_res,
            # data_camera_mode=opt.data_camera_mode,
            add_camera_cond=opt.add_camera_cond,
            # camera_path=opt.camera_path,
            # split='test' if opt.inference_vis else 'train',
        )
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # Subclass of training.dataset.Dataset.
        dataset_kwargs.camera_path = ''
        dataset_kwargs.resolution = dataset_obj.resolution  # Be explicit about resolution.
        logger.info(f"label shape is {dataset_obj.label_shape}")
        dataset_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj)  # Be explicit about dataset size.
        dataset_obj._name = "objaverse"
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise ValueError(f'--data: {err}')


if __name__ == '__main__':
    pass
