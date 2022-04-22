import os

import torch

from .adjust_lr import scaled_learning_rate
from .load_checkpoint import load_checkpoint
from .logger import init_log
from .mydataparallel import MyDataParallel
from .save_checkpoint import save_checkpoint
from .diff_hist import differentiable_histogram
from .color_convert import cam_to_ciexyz, rgb_to_gray

def init_env(opts, model, optimizer):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if not os.path.exists(opts.exp_path):
        os.mkdir(opts.exp_path)
    init_log(opts.exp_path)
    epoch = None
    if opts.checkpoint is not None:
        epoch = load_checkpoint(model, optimizer, opts.checkpoint)
        scaled_learning_rate(optimizer, opts.learning_rate, epoch)
    return epoch

