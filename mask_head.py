from blocks import get_norm
from model import Conv2d
from shape_spec import ShapeSpec
import torch
from instances import Instances
from typing import List
import torch.nn as nn
from config import configurable
from model import Conv2d
import fvcore.nn.weight_init as weight_init
from proposal_utils import cat
from torch.nn.functional import interpolate
from torch.nn.modules import activation, padding
from torch.nn.modules.conv import ConvTranspose2d

def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int=0):
    pass

def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
    cls_agnostic_mask = pred_mask_logits.size(1) ==1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][: ,None].sigmoid()

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)

class BaseMaskRCNNHead(nn.Module):
    @configurable
    def __init__(self, *, vis_period=0):
        super().__init__()
        self.vis_period = vis_period

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'vis_period':cfg.VIS_PERIOD}

    def forward(self, x, instances: List[Instances]):
        x = self.layers(x)
        if self.training:
            return {'loss_mask':mask_rcnn_loss(x, instances, self.vis_period)}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        raise NotImplementedError

class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead, nn.Sequential):
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm='', **kwargs):
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module('mask_fcn{}'.format(k+1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module('deconv_relu', nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),
            conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret['num_classes'] = 1
        else:
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x 

def build_mask_head(cfg, input_shape):
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return MaskRCNNConvUpsampleHead(cfg, input_shape)
