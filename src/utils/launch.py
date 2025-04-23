# adapted from dino 
# https://github.com/facebookresearch/dino

import os
import sys
import torch
import torch.distributed as dist

from ._utils import synchronize
from .helpfuns import load_params
from .system_def import define_system_params


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def launch(main_func, args=()):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine`) on each machine.
    Assume everything is happening on a single node!
    """
    params, arguments = args

    # Set OMP_NUM_THREADS for number of CPUs threads per process
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Define GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if params.system_params['use_GPU']:
        if not params.system_params['use_all_GPUs']:
            os.environ["CUDA_VISIBLE_DEVICES"] = params.system_params['which_GPUs']
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        arguments.rank = int(os.environ["RANK"])
        arguments.world_size = int(os.environ['WORLD_SIZE'])
        arguments.gpu = int(os.environ['LOCAL_RANK'])

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        arguments.rank = int(os.environ['SLURM_PROCID'])
        arguments.gpu = arguments.rank % torch.cuda.device_count()

    # launched naively with `python main.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        arguments.rank, arguments.gpu, arguments.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=arguments.dist_url,
        world_size=arguments.world_size,
        rank=arguments.rank,
    )

    torch.cuda.set_device(arguments.gpu)
    print('| distributed init (rank {}): {}'.format(
        arguments.rank, arguments.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(arguments.rank == 0)
    
    main_func(params, arguments)

