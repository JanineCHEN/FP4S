import os
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from kornia.losses import binary_focal_loss_with_logits as BFL

from tools.config import get_config
cfg = get_config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#for CCT
from tools import ramps

def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.to(device)
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge

def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel,[1,1,3,3])
    filter = filter.to(device)
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx

def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1,3,3])
    filter = filter.to(device)
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy

def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001**2, 0.5)
    return cp_s

def get_saliency_smoothness(pred, gt, size_average=True):
    alpha = 10
    s1 = 10
    s2 = 1
    ## first oder derivative: sobel
    sal_x = torch.abs(gradient_x(pred))
    sal_y = torch.abs(gradient_y(pred))
    gt_x = gradient_x(gt)
    gt_y = gradient_y(gt)
    w_x = torch.exp(torch.abs(gt_x) * (-alpha))
    w_y = torch.exp(torch.abs(gt_y) * (-alpha))
    cps_x = charbonnier_penalty(sal_x * w_x)
    cps_y = charbonnier_penalty(sal_y * w_y)
    cps_xy = cps_x + cps_y

    ## second order derivative: laplacian
    lap_sal = torch.abs(laplacian_edge(pred))
    lap_gt = torch.abs(laplacian_edge(gt))
    weight_lap = torch.exp(lap_gt * (-alpha))
    weighted_lap = charbonnier_penalty(lap_sal*weight_lap)

    smooth_loss = s1*torch.mean(cps_xy) + s2*torch.mean(weighted_lap)

    return smooth_loss

class smoothness_loss(torch.nn.Module):
    def __init__(self, size_average = True):
        super(smoothness_loss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return get_saliency_smoothness(pred, target, self.size_average)


def scribble_loss(sal1, edge_map, sal2, images, gts, masks, grays, edges, iter_per_epoch, curr_iter, ep):

    smooth_loss = smoothness_loss(size_average=True)
    CE = torch.nn.BCELoss()
    if cfg.useabCELoss:
        abCE = abCE_loss(iter_per_epoch=iter_per_epoch, 
                        epochs=cfg.num_epochs,
                        num_classes=cfg.num_classes)

    img_size = images.size(2) * images.size(3) * images.size(0)  #B*C*H*W?
    ratio = img_size / torch.sum(masks)

    sal1_prob = torch.sigmoid(sal1)
    sal2_prob = torch.sigmoid(sal2)
    sal1_prob = sal1_prob * masks.float()
    sal2_prob = sal2_prob * masks.float()

    label_weight = torch.sum(masks) / (torch.sum(gts,(0, 2, 3)) + 1e-8)
    label_weight = (label_weight - label_weight.min()) / (
        label_weight.max() - label_weight.min())
    label_weight = torch.softmax(label_weight, dim=0)

    smoothLoss_cur1 = 0
    for idx, sub_tensor in enumerate(torch.transpose(sal1, 0, 1).unsqueeze(2)):
        smoothLoss_cur1 += smooth_loss(torch.sigmoid(sub_tensor),
                                        grays) * label_weight[idx]
    smoothLoss_cur1 = cfg.sm_loss_weight * (smoothLoss_cur1)

    gts_ = gts.float() * masks.float()
    if cfg.useFocalLoss:
        FL_loss_1 = 0
        sal1_ = sal1 * masks.float()
        for idx, sub_tensor in enumerate(torch.transpose(sal1_, 0, 1).unsqueeze(2)):
            kwargs = {"alpha": label_weight[idx], "gamma": cfg.gamma, "reduction": cfg.reduction}
            logits = sub_tensor
            labels = gts_[:,idx,:,:].float()
            FL_loss_1 += BFL(logits, labels, **kwargs)
        sal_loss1 = ratio * FL_loss_1 + smoothLoss_cur1

    if cfg.useabCELoss:
        abCE_loss_1 = abCE(sal1_prob.float(), gts.float() * masks.float(), ignore_index=None,
                        curr_iter=curr_iter, epoch=ep)
        sal_loss1 = ratio * abCE_loss_1 + smoothLoss_cur1

    else:
        BCE_loss_1 = ratio * CE(sal1_prob.float(),
                                gts.float() * masks.float())
        sal_loss1 = BCE_loss_1 + smoothLoss_cur1

    smoothLoss_cur2 = 0
    for idx, sub_tensor in enumerate(
            torch.transpose(sal2, 0, 1).unsqueeze(2)):
        smoothLoss_cur2 += smooth_loss(torch.sigmoid(sub_tensor),
                                        grays) * label_weight[idx]
    smoothLoss_cur2 = cfg.sm_loss_weight * (smoothLoss_cur2)

    if cfg.useFocalLoss:
        FL_loss_2 = 0
        sal2_ = sal2 * masks.float()
        for idx, sub_tensor in enumerate(torch.transpose(sal2_, 0, 1).unsqueeze(2)):
            kwargs = {"alpha": label_weight[idx], "gamma": cfg.gamma, "reduction": cfg.reduction}
            logits =sub_tensor
            labels = gts_[:,idx,:,:].float()
            FL_loss_2 += BFL(logits, labels, **kwargs)
        sal_loss2 = ratio * FL_loss_2 + smoothLoss_cur2

    if cfg.useabCELoss:
        abCE_loss_2 = abCE(sal2_prob.float(), gts.float() * masks.float(), ignore_index=None,
                        curr_iter=curr_iter, epoch=ep)
        sal_loss2 = ratio * abCE_loss_2 + smoothLoss_cur2

    else:
        BCE_loss_2 = ratio * CE(sal2_prob.float(),
                                gts.float() * masks.float())
        sal_loss2 = BCE_loss_2 + smoothLoss_cur2
        
    edge_loss = cfg.edge_loss_weight * CE(torch.sigmoid(edge_map), edges)

    if cfg.useFocalLoss:
        return sal_loss1, edge_loss, sal_loss2, ratio * FL_loss_1, smoothLoss_cur1, ratio * FL_loss_2, smoothLoss_cur2
    if cfg.useabCELoss:
        return sal_loss1, edge_loss, sal_loss2, ratio * abCE_loss_1, smoothLoss_cur1, ratio * abCE_loss_2, smoothLoss_cur2
    else:
        return sal_loss1, edge_loss, sal_loss2, BCE_loss_1, smoothLoss_cur1, BCE_loss_2, smoothLoss_cur2


#####CCT##########
class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """
    def __init__(self, final_w, iter_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iter_per_epoch = iter_per_epoch
        self.rampup_starts = rampup_starts * iter_per_epoch
        self.rampup_ends = rampup_ends * iter_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iter_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    return F.cross_entropy(input_logits/temperature, target_targets, ignore_index=ignore_index)


class abCE_loss(nn.Module):
    """
    Annealed-Bootstrapped cross-entropy loss
    """
    def __init__(self, iter_per_epoch, epochs, num_classes, weight=None,
                        reduction='mean', thresh=0.7, min_kept=1, ramp_type='log_rampup'):
        super(abCE_loss, self).__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type
        
        if ramp_type is not None:
            self.rampup_func = getattr(ramps, ramp_type)
            self.iter_per_epoch = iter_per_epoch
            self.num_classes = num_classes
            self.start = 1/num_classes
            self.end = 0.9
            self.total_num_iters = (epochs - (0.6 * epochs)) * iter_per_epoch

    def threshold(self, curr_iter, epoch):
        cur_total_iter = self.iter_per_epoch * epoch + curr_iter
        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index, curr_iter, epoch):
        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target = tmp_target.type(torch.int64)
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target)
        mask = target.contiguous().view(-1, ) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        if self.ramp_type is not None:
            thresh =  self.threshold(curr_iter=curr_iter, epoch=epoch)
        else:
            thresh = self.thresh
        min_threshold = torch.min(sort_prob) if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, thresh)
        loss_matrix = F.binary_cross_entropy(predict, target,
                                      reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size() # (batch_size * num_classes * H * W)
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean') # take the mean over the batch_size


def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')
    kl2 = F.kl_div(torch.log(targets+epsilon), M, reduction='mean')
    return (kl1 + kl2) * 0.5

