import os
import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

from .config import get_config
cfg = get_config()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Data augmentation
# Copied and edited from https://www.kaggle.com/code/riadalmadani/fastai-effb0-base-model-birdclef2023
# https://towardsdatascience.com/cutout-mixup-and-cutmix-implementing-modern-image-augmentations-in-pytorch-a9d7db3074ad#00b8
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data, bbx1, bby1, bbx2, bby2):
    new_data = torch.clone(data)
    new_data[0, :, bbx1:bbx2, bby1:bby2] = data[1, :, bbx1:bbx2, bby1:bby2]
    new_data[1, :, bbx1:bbx2, bby1:bby2] = data[0, :, bbx1:bbx2, bby1:bby2]
    return new_data

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def IoU_metric(inputs, target):
    TP = (target * inputs).sum()
    TN = (inputs == target).sum() - TP
    FP = (inputs == 1).sum() - TP
    FN = (inputs == 0).sum() - TN
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return TP / (TP+FN+FP)

def dice_metric(inputs, target):
    TP = (target * inputs).sum()
    TN = (inputs == target).sum() - TP
    FP = (inputs == 1).sum() - TP
    FN = (inputs == 0).sum() - TN
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return 2.0*TP / (2.0*TP+FN+FP)

def evaluation(pred, truth, mode = 'dice', threshold = 0.9, eps=1e-12):
    # pred, raw logits: N,C,H,W
    # truth, binary: N,C,H,W
    # mode: 'dice' or 'IoU'
    num_classes = pred.shape[1]
    coefficients = torch.zeros(num_classes)
    for i in range(num_classes):
        pred_ = pred[:,i,:,:]
        truth_ = truth[:,i,:,:]
        pred_ = (pred_ - pred_.min()) / (pred_.max() - pred_.min() + eps)
        pred_[pred_ > threshold] = 1
        pred_[pred_ != 1] = 0
        if mode == 'IoU':
            a = IoU_metric(pred_,truth_)
        if mode == 'dice':
            a = dice_metric(pred_,truth_)
        coefficients[i] = a
    return coefficients # return (C,)


def load_checkpoint(ckpt_dir_or_file, map_location=torch.device('cpu')):
    ckpt = torch.load(ckpt_dir_or_file, map_location=map_location)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)
    return ckpt

def load_model(model_path, model):
    ckpt_path = model_path + f'/FP4S_{cfg.resume_ep}.pth'
    try:
        ckpt = load_checkpoint(ckpt_path)
        start_ep = ckpt['epoch'] + 1
        print(f'Try to load checkpoint of epoch {start_ep-1}!')
        ckpt['model'] = {key.replace("module.", ""): value for key, value in ckpt['model'].items()}
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.1)
        if cfg.lr_schedule:
            scheduler = CyclicLR(optimizer,
                                    base_lr=cfg.base_lr,
                                    max_lr=cfg.max_lr,
                                    step_size_up=cfg.step_size_up,
                                    mode="triangular",
                                    cycle_momentum=False)
            print("CyclicLR is adopted!!!")
        else:
            scheduler = None

        optimizer.load_state_dict(ckpt['optimizer'])
        print(f' Checkpoint of epoch {start_ep-1} loaded!')

    except:
        print(' [*] No checkpoint!')
        model = model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.1)
        if cfg.lr_schedule:
            scheduler = CyclicLR(optimizer,
                                    base_lr=cfg.base_lr,
                                    max_lr=cfg.max_lr,
                                    step_size_up=cfg.step_size_up,
                                    mode="triangular",
                                    cycle_momentum=False)
            print("CyclicLR is adopted!!!")
        else:
            scheduler = None

        start_ep = 0

    return start_ep, model, optimizer, scheduler
