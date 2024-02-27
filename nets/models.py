import os
import numpy as np
import cv2
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from itertools import chain
from nets.decoders import *
from tools.loss import *

from tools.config import get_config
cfg = get_config()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Edge_Module(nn.Module):

    def __init__(self, channel = 16, mid_fea=32):
        super(Edge_Module, self).__init__()
        self.channel = channel
        if 'resnet50' in cfg.backbone.lower() or 'resnet101' in cfg.backbone.lower() or 'resnet152' in cfg.backbone.lower():
            self.channel = channel * 4
        in_fea=[self.channel, 4*self.channel, 8*self.channel]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self, x2, x4, x5):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge4, edge5], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''
    def __init__(self, in_dim, reduction_dim, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()
        if 'resnet50' in cfg.backbone.lower() or 'resnet101' in cfg.backbone.lower() or 'resnet152' in cfg.backbone.lower():
            in_dim = in_dim * 4
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()
        img_features = self.img_pooling(x) 
        img_features = self.img_conv(img_features) 
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True) 
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = int(sum([np.prod(p.size()) for p in model_parameters]))
        return f'\nNbr of trainable parameters: {nbr_params}'


class FP4S(BaseModel):
    def __init__(self,
                 backbone, 
                 channel=8, 
                 num_classes=25, 
                 ifpretrain = False,
                 iter_per_epoch = None,
                 ):
        
        super(FP4S, self).__init__()
        self.ifpretrain = ifpretrain
        self.iter_per_epoch = iter_per_epoch
        if 'gn' in cfg.backbone.lower():
            self.backbone = backbone(channel = channel*2, num_groups = cfg.num_groups, ifpretrain = self.ifpretrain)
        if 'resnet' in cfg.backbone.lower():
            self.backbone = backbone(num_classes=cfg.num_classes, pretrained=self.ifpretrain)
        else: 
            self.backbone = backbone(channel = channel*2, ifpretrain = self.ifpretrain)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU(True)
        self.edge_layer = Edge_Module(channel = channel*2)
        self.aspp = _AtrousSpatialPyramidPoolingModule(channel*16, channel, output_stride=16)
        self.sal_conv = nn.Conv2d(num_classes, channel, kernel_size=3, padding=1, bias=False)
        self.edge_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(channel*2)
        self.after_aspp_conv5 = nn.Conv2d(channel*6, channel, kernel_size=1, bias=False)
        if 'resnet50' in cfg.backbone.lower() or 'resnet101' in cfg.backbone.lower() or 'resnet152' in cfg.backbone.lower():
            self.after_aspp_conv2 = nn.Conv2d(channel*16, channel, kernel_size=1, bias=False)
        else:
            self.after_aspp_conv2 = nn.Conv2d(channel*4, channel, kernel_size=1, bias=False)
        self.final_sal_seg = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, bias=False))
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.fused_edge_sal = nn.Conv2d(channel*2, num_classes, kernel_size=3, padding=1, bias=False)


        #####CCT#####
        ## Supervised and unsupervised losses
        self.unsuper_loss = softmax_mse_loss
        # self.unsuper_loss = softmax_js_loss
        self.confidence_th = 0.5
        self.confidence_masking = False
        # The shared encoder
        self.encoder = self.backbone
        # The auxilary decoders
        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4
        if 'resnet50' in cfg.backbone.lower() or 'resnet101' in cfg.backbone.lower() or 'resnet152' in cfg.backbone.lower():
            decoder_in_ch = decoder_in_ch * 4
        vat_decoder = [VATDecoder(upscale, decoder_in_ch, num_classes, xi= 1e-6,
                                    eps=2.0) for _ in range(2)]
        drop_decoder = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
                                    drop_rate=0.5, spatial_dropout=True)
                                    for _ in range(6)]
        cut_decoder = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=0.4)
                                    for _ in range(6)]
        context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes)
                                    for _ in range(2)]
        object_masking = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes)
                                    for _ in range(2)]
        feature_drop = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes)
                                    for _ in range(6)]
        feature_noise = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
                                    uniform_range=0.3)
                                    for _ in range(6)]

        self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
                                *context_m_decoder, *object_masking, *feature_drop, *feature_noise])


    def forward(self, x_l=None, x_ul=None, gts=None, masks=None, grays=None, edges=None, testing=False, curr_iter=None, ep=None):
        ####Scribble guidance###############
        x_size = x_l.size() 
        x_1, x_2, x_3, x_4, x_5 = self.encoder(x_l)
        edge_map = self.edge_layer(x_1, x_3, x_4)
        edge_out = torch.sigmoid(edge_map)

        im_arr = x_l.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8) 
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).to(device).float()
        if edge_out.shape != canny.shape:
            edge_out = F.interpolate(edge_out, canny.size()[2:], mode='bilinear', align_corners=True) 
        cat = torch.cat((edge_out, canny), dim=1) 
        acts = self.fuse_canny_edge(cat) 
        acts = torch.sigmoid(acts) 
        x5 = self.aspp(x_5, acts) 
        x_conv5 = self.after_aspp_conv5(x5) 
        x_conv2 = self.after_aspp_conv2(x_2) 
        x_conv5_up = F.interpolate(x_conv5, x_2.size()[2:], mode='bilinear', align_corners=True)
        feat_fuse = torch.cat([x_conv5_up, x_conv2], 1) 

        sal_init = self.final_sal_seg(feat_fuse) 
        sal_init = F.interpolate(sal_init, x_size[2:], mode='bilinear') 
        sal_feature = self.sal_conv(sal_init) 
        edge_feature = self.edge_conv(edge_map) 
        if edge_feature.shape != sal_feature.shape:
            edge_feature = F.interpolate(edge_feature, sal_feature.size()[2:], mode='bilinear', align_corners=True)
        sal_edge_feature = self.relu(torch.cat((sal_feature, edge_feature), 1)) 
        sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
        sal_ref = self.fused_edge_sal(sal_edge_feature)

        if testing:
            print("Testing mode is on!")
            return sal_init, sal_ref

        if edge_out.shape != edge_map.shape:
            edge_map = F.interpolate(edge_map, edge_out.size()[2:], mode='bilinear', align_corners=True)

        if cfg.useFocalLoss:
            sal_loss1, edge_loss, sal_loss2, FL_loss_1, smoothLoss_cur1, FL_loss_2, smoothLoss_cur2 = scribble_loss(sal_init, edge_map, sal_ref, x_l, gts, masks, grays, edges, self.iter_per_epoch, curr_iter, ep)
        if cfg.useabCELoss:
            sal_loss1, edge_loss, sal_loss2, abCE_loss_1, smoothLoss_cur1, abCE_loss_2, smoothLoss_cur2 = scribble_loss(sal_init, edge_map, sal_ref, x_l, gts, masks, grays, edges, self.iter_per_epoch, curr_iter, ep)
        else:
            sal_loss1, edge_loss, sal_loss2, BCE_loss_1, smoothLoss_cur1, BCE_loss_2, smoothLoss_cur2 = scribble_loss(sal_init, edge_map, sal_ref, x_l, gts, masks, grays, edges, self.iter_per_epoch, curr_iter, ep)

        # Get unsupervised predictions
        x_1_ul, x_2_ul, x_3_ul, x_4_ul, x_5_ul = self.encoder(x_ul)
        # Scribble Decoder
        edge_map_ul = self.edge_layer(x_1_ul, x_3_ul, x_4_ul)
        edge_out_ul = torch.sigmoid(edge_map_ul)
        canny_ul = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny_ul[i] = cv2.Canny(im_arr[i], 10, 100)
        canny_ul = torch.from_numpy(canny_ul).to(device).float() 
        if edge_out_ul.shape != canny_ul.shape:
            edge_out_ul = F.interpolate(edge_out_ul, canny_ul.size()[2:], mode='bilinear', align_corners=True)
        cat_ul = torch.cat((edge_out_ul, canny_ul), dim=1) 
        acts_ul = self.fuse_canny_edge(cat_ul) 
        acts_ul = torch.sigmoid(acts_ul) 
        x5_ul = self.aspp(x_5_ul, acts_ul) 
        x_conv5_ul = self.after_aspp_conv5(x5_ul) 
        x_conv2_ul = self.after_aspp_conv2(x_2_ul) 
        x_conv5_up_ul = F.interpolate(x_conv5_ul, x_2_ul.size()[2:], mode='bilinear', align_corners=True) 
        feat_fuse_ul = torch.cat([x_conv5_up_ul, x_conv2_ul], 1) 

        sal_init_ul = self.final_sal_seg(feat_fuse_ul) 
        sal_init_ul = F.interpolate(sal_init_ul, x_size[2:], mode='bilinear') 
        sal_feature_ul = self.sal_conv(sal_init_ul) 
        edge_feature_ul = self.edge_conv(edge_map_ul) 
        if edge_feature_ul.shape != sal_feature_ul.shape:
            edge_feature_ul = F.interpolate(edge_feature_ul, sal_feature_ul.size()[2:], mode='bilinear', align_corners=True)
        sal_edge_feature_ul = self.relu(torch.cat((sal_feature_ul, edge_feature_ul), 1)) 
        sal_edge_feature_ul = self.rcab_sal_edge(sal_edge_feature_ul) 
        sal_ref_ul = self.fused_edge_sal(sal_edge_feature_ul) 

        ### Get auxiliary predictions
        ###sal_init_ul###
        outputs_sal_init_ul = [aux_decoder(x_5_ul, sal_init_ul.detach()) for aux_decoder in self.aux_decoders]
        targets = F.softmax(sal_init_ul.detach(), dim=1)
        # Compute unsupervised loss for sal_init_ul
        loss_unsup_sal_init = sum([self.unsuper_loss(inputs=F.interpolate(u, size=(x_l.size(2), x_l.size(3)), mode='bilinear', align_corners=True), \
                            targets=targets, conf_mask=self.confidence_masking, threshold=self.confidence_th, use_softmax=False)
                            for u in outputs_sal_init_ul])
        loss_unsup_sal_init = (loss_unsup_sal_init / len(outputs_sal_init_ul))
        ###sal_ref_ul###
        outputs_sal_ref_ul = [aux_decoder(x_5_ul, sal_ref_ul.detach()) for aux_decoder in self.aux_decoders]
        targets = F.softmax(sal_ref_ul.detach(), dim=1)
        # Compute unsupervised loss for sal_ref_ul
        loss_unsup_sal_ref = sum([self.unsuper_loss(inputs=F.interpolate(u, size=(x_l.size(2), x_l.size(3)), mode='bilinear', align_corners=True), \
                            targets=targets, conf_mask=self.confidence_masking, threshold=self.confidence_th, use_softmax=False)
                            for u in outputs_sal_ref_ul])
        loss_unsup_sal_ref = (loss_unsup_sal_ref / len(outputs_sal_ref_ul))

        curr_losses = {'sal_loss1': sal_loss1.item()}
        curr_losses['edge_loss'] = edge_loss.item()
        curr_losses['sal_loss2'] = sal_loss2.item()
        curr_losses['loss_unsup_sal_1'] = loss_unsup_sal_init.item()
        curr_losses['loss_unsup_sal_2'] = loss_unsup_sal_ref.item()

        if cfg.useFocalLoss:
            curr_losses['FL_loss_1'] = FL_loss_1.item()
            curr_losses['FL_loss_2'] = FL_loss_2.item()
            curr_losses['smoothLoss_cur1'] = smoothLoss_cur1.item()
            curr_losses['smoothLoss_cur2'] = smoothLoss_cur2.item()
        if cfg.useabCELoss:
            curr_losses['abCE_loss_1'] = abCE_loss_1.item()
            curr_losses['abCE_loss_2'] = abCE_loss_2.item()
            curr_losses['smoothLoss_cur1'] = smoothLoss_cur1.item()
            curr_losses['smoothLoss_cur2'] = smoothLoss_cur2.item()
        else:
            curr_losses['BCE_loss_1'] = BCE_loss_1.item()
            curr_losses['BCE_loss_2'] = BCE_loss_2.item()
            curr_losses['smoothLoss_cur1'] = smoothLoss_cur1.item()
            curr_losses['smoothLoss_cur2'] = smoothLoss_cur2.item()

        outputs = {'sal1': sal_init}
        outputs['edge'] = edge_map
        outputs['sal2'] = sal_ref
        outputs['sal1_ul'] = sal_init_ul
        outputs['edge_ul'] = edge_map_ul
        outputs['sal2_ul'] = sal_ref_ul

        unsup_loss_w = consistency_weight(final_w=30, iter_per_epoch=self.iter_per_epoch,
                                        rampup_ends=int(0.1 * cfg.num_epochs))
        unsup_w = unsup_loss_w(epoch=ep, curr_iter=curr_iter)
        curr_losses['unsup_w'] = unsup_w
        sup_loss = sal_loss1 + edge_loss + sal_loss2
        curr_losses['sup_loss'] = sup_loss.item()
        unsup_loss = loss_unsup_sal_init + loss_unsup_sal_ref
        curr_losses['unsup_loss'] = unsup_loss.item()
        total_loss = sup_loss + unsup_loss * unsup_w
        curr_losses['total_loss'] = total_loss.item()
        total_loss = total_loss.mean()
        curr_losses['total_loss_mean'] = total_loss.item()

        return total_loss, curr_losses, outputs


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'semi':
            return chain(self.encoder.get_module_params(), self.main_decoder.parameters(), 
                        self.aux_decoders.parameters())

        return chain(self.encoder.get_module_params(), self.main_decoder.parameters())

