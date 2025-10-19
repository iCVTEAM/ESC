import torch
import numpy as np
import random
import time
import os
import sys
import functools
from pathlib import Path
from torch.backends import cudnn
from torch import nn, Tensor
from torch.autograd import profiler
from typing import Union
from torch import distributed as dist
import logging
import datetime

def setup_ddp():
    # print(os.environ.keys())
    if 'SLURM_PROCID' in os.environ and not 'RANK' in os.environ:
        # --- multi nodes
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        gpu = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=7200))
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # gpu = int(os.environ(['LOCAL_RANK']))
        # ---
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group('nccl', init_method="env://",world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=7200))
        dist.barrier()
    else:
        gpu = 0
    return gpu


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()