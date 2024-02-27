import sys, os
import numpy as np
import pandas as pd
import random
import requests
from tqdm import tqdm
from itertools import cycle
from tools.config import get_config

cfg = get_config()

from torch.autograd import Variable
import torch
import torchvision.transforms as transforms

from tools.dataloader import *
from tools.utils import *
from tools.loss import *
from nets.models import *
from nets.resnet_modules import *

from datetime import datetime

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
torch.autograd.set_detect_anomaly(True)

seed = cfg.seed

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")
set_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Download unannotated floorplan images
if os.path.isfile(cfg.image_unsupervised_list_path):
    pass
if not os.path.isfile(cfg.image_unsupervised_list_path):
    print(f"{cfg.image_unsupervised_list_path} does not exist! Start downloading {str(cfg.num_unsupervised_imgs)} unannotated floor plan images: ")
    if not os.path.exists(cfg.image_unsupervised_root):
        os.makedirs(cfg.image_unsupervised_root)
    ulist = pd.read_csv('./dataset/unsupervised_list.csv')
    ulist = list(ulist['img_url'])
    random.shuffle(ulist)
    ulist_txt = []
    index = 0
    for url in tqdm(ulist[:cfg.num_unsupervised_imgs]):
        response_image = requests.get(str(url))
        file = open(os.path.join(cfg.image_unsupervised_root,f'unsupervised_{index}.jpg'), 'wb')
        ulist_txt.append(f'unsupervised_{index}.jpg')
        file.write(response_image.content)
        file.close()
        index += 1
    with open(cfg.image_unsupervised_list_path, 'w') as outfile:
        outfile.write('\n'.join(ulist_txt))

## DATA AUGMENTATION
transf_aug = transforms.Compose([ #expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions
                                transforms.RandomHorizontalFlip(0.5), 
                                transforms.RandomVerticalFlip(0.5),
                                transforms.RandomRotation(10), #If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
                                transforms.RandomResizedCrop((cfg.trainsize,cfg.trainsize),scale=(0.7, 1.0)),
                                ])


def train(supervised_loader, unsupervised_loader, model, _epoch, optimizer, scheduler, cfg):

    train_loader = zip(cycle(supervised_loader), unsupervised_loader)

    model.train()
    for i, pack in tqdm(enumerate(train_loader)):
        (images, gts, masks, grays, edges), (images_unlabeled, images_unlabeled_name) = pack
        images = Variable(images).to(device, non_blocking=True)
        gts = Variable(gts).to(device, non_blocking=True)
        masks = Variable(masks).to(device, non_blocking=True)
        grays = Variable(grays).to(device, non_blocking=True)
        edges = Variable(edges).to(device, non_blocking=True)
        images_unlabeled = Variable(images_unlabeled).to(device, non_blocking=True)
        print(i, "train_ul: ", images_unlabeled_name)

        ## DATA AUGMENTATION
        state = torch.get_rng_state()
        images = transf_aug(images)
        torch.set_rng_state(state)
        gts = transf_aug(gts)
        torch.set_rng_state(state)
        masks = transf_aug(masks)
        torch.set_rng_state(state)
        grays = transf_aug(grays)
        torch.set_rng_state(state)
        edges = transf_aug(edges)

        ## CUTMIX AUGMENTATION
        if cfg.cutmix:
            lam = np.random.beta(1, 1)
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images = cutmix(images, bbx1, bby1, bbx2, bby2)
            gts = cutmix(gts, bbx1, bby1, bbx2, bby2)
            masks = cutmix(masks, bbx1, bby1, bbx2, bby2)
            grays = cutmix(grays, bbx1, bby1, bbx2, bby2)
            edges = cutmix(edges, bbx1, bby1, bbx2, bby2)
        
        optimizer.zero_grad()
        total_loss, curr_losses, outputs = model(x_l=images, x_ul=images_unlabeled, gts=gts, masks=masks, grays=grays, edges=edges, curr_iter = i, ep = _epoch)

        #LOSS BACKPROP
        total_loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            print(f"Factor = {i}, Learning Rate = {optimizer.param_groups[0]['lr']}")
            scheduler.step()

        #PRINT and save LOSSES
        print(f'Epoch [{_epoch:03d}/{cfg.num_epochs:03d}], Step [{i:04d}/{len(unsupervised_loader):04d}], lr {optimizer.param_groups[0]["lr"]}, sm_loss_weight {cfg.sm_loss_weight}, edge_loss_weight {cfg.edge_loss_weight}, {str(curr_losses)}'
                + "\n")

        with open(os.path.join(cfg.txt_path, f"loss.txt"),"a",encoding="utf-8") as file:
            file.write(
                f'{datetime.now()} Epoch [{_epoch:03d}/{cfg.num_epochs:03d}], Step [{i:04d}/{len(unsupervised_loader):04d}], lr {optimizer.param_groups[0]["lr"]}, sm_loss_weight {cfg.sm_loss_weight}, edge_loss_weight {cfg.edge_loss_weight}, {str(curr_losses)}'
                + "\n")

    #SAVE CHECKPOINT
    if _epoch % 1 == 0:
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': _epoch,
            }, cfg.model_path + 'FP4S' + '_%d' % _epoch + '.pth')


def validation(val_loader, model, _epoch, cfg):
    model.eval()

    coefficients_IoU_1 = torch.zeros(cfg.num_classes)
    coefficients_dice_1 = torch.zeros(cfg.num_classes)

    n = 0
    for i, pack in enumerate(val_loader, start=1):

        img_, gt_, _ = pack
        img_ = Variable(img_).to(device)
        sal1, _ = model(x_l=img_, testing=True)
        sal1 = sal1.detach().cpu()

        dice_eval_1 = evaluation(pred = sal1, truth = gt_.int(), mode = 'dice')
        IoU_eval_1 = evaluation(pred = sal1, truth = gt_.int(), mode = 'IoU')
        
        coefficients_dice_1 = coefficients_dice_1 + dice_eval_1
        coefficients_IoU_1 = coefficients_IoU_1 + IoU_eval_1
        n += 1

    coefficients_dice_1 = coefficients_dice_1/n
    coefficients_IoU_1 = coefficients_IoU_1/n

    with open(os.path.join(cfg.txt_path, f"coefficients_dice.txt"), "a", encoding="utf-8") as file:
        file.write(f"Epoch [{_epoch:03d}/{cfg.num_epochs:03d}],{','.join([f'{x:.4f}' for x in list(coefficients_dice_1.numpy())])}"
            + "\n")
    with open(os.path.join(cfg.txt_path, f"coefficients_IoU.txt"), "a", encoding="utf-8") as file:
        file.write(f"Epoch [{_epoch:03d}/{cfg.num_epochs:03d}],{','.join([f'{x:.4f}' for x in list(coefficients_IoU_1.numpy())])}"
            + "\n")

def main():

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.txt_path, exist_ok=True)

    if cfg.cutmix:
        cfg.batchsize = 2

    supervised_loader = get_loader(cfg.image_root,
                              cfg.gt_root,
                              cfg.mask_root,
                              cfg.gray_root,
                              cfg.edge_root,
                              batchsize=cfg.batchsize,
                              trainsize=cfg.trainsize,
                              ifNorm=cfg.ifNorm)
    
    unsupervised_loader = get_unsupervised_loader(cfg.image_unsupervised_root, 
                                                batchsize=cfg.batchsize,
                                                trainsize=cfg.trainsize,
                                                ifNorm=cfg.ifNorm,
                                                image_list_path=cfg.image_unsupervised_list_path)

    val_loader = get_val_loader(cfg.image_val_root,
                                cfg.gt_val_root,
                                batchsize=cfg.batchsize,
                                trainsize=cfg.trainsize,
                                ifNorm=cfg.ifNorm)

    if cfg.backbone.lower() == 'resnet18':
        backbone = resnet18
    if cfg.backbone.lower() == 'deepbase_resnet18':
        backbone = deepbase_resnet18
    if cfg.backbone.lower() == 'resnet34':
        backbone = resnet34
    if cfg.backbone.lower() == 'deepbase_resnet34':
        backbone = deepbase_resnet34
    if cfg.backbone.lower() == 'resnet50':
        backbone = resnet50
    if cfg.backbone.lower() == 'deepbase_resnet50':
        backbone = deepbase_resnet50
    if cfg.backbone.lower() == 'resnet101':
        backbone = resnet101
    if cfg.backbone.lower() == 'deepbase_resnet101':
        backbone = deepbase_resnet101
    if cfg.backbone.lower() == 'deepbase_resnet101':
        backbone = deepbase_resnet101
    if cfg.backbone.lower() == 'resnet152':
        backbone = resnet152
    if cfg.backbone.lower() == 'deepbase_resnet152':
        backbone = deepbase_resnet152

    print(backbone)

    model = FP4S(backbone=backbone, channel=cfg.channel, num_classes=cfg.num_classes, ifpretrain = cfg.ifpretrain, iter_per_epoch = len(unsupervised_loader))

    _epoch, model, optimizer, scheduler = load_model(cfg.model_path, model)

    while _epoch < cfg.num_epochs:
        train(supervised_loader, unsupervised_loader, model, _epoch, optimizer, scheduler, cfg)
        validation(val_loader, model, _epoch, cfg)
        _epoch += 1


if __name__ == '__main__':
    main()
