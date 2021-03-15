import sys
import numpy as np

import torchvision
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from image_lists import ImageList
import inspect
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from build_cfg import cfg
from config import configurable
from instances import Instances
from matcher import Matcher
from shape_spec import ShapeSpec
from poolers import ROIPooler
from box_head import build_box_head
from mask_head import build_mask_head
from fast_rcnn import FastRCNNOutputLayers

input_shape = {
                'p2': ShapeSpec(channels=256, height=None, width=None, stride=4), 
                'p3': ShapeSpec(channels=256, height=None, width=None, stride=8), 
                'p4': ShapeSpec(channels=256, height=None, width=None, stride=16), 
                'p5': ShapeSpec(channels=256, height=None, width=None, stride=32), 
                'p6': ShapeSpec(channels=256, height=None, width=None, stride=64)
                }



class ROIHeads(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True,
    ):
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt
    
    @classmethod
    def from_config(cls,cfg):
        return {
            'batch_size_per_image': cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            'positive_fraction': cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'proposal_append_gt': cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            'proposal_matcher': Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False
            ),
        }


class StandardROIHeads(ROIHeads):
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs):
        # print(args, '|', kwargs)
        super().__init__(**kwargs)

        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            pass

        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape:List[ShapeSpec]):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        print(pooler_resolution,' | ', pooler_scales, ' | ', sampling_ratio)
        box_pooler = ROIPooler(
            output_size = pooler_resolution,
            scales = pooler_scales,
            sampling_ratio = sampling_ratio,
            pooler_type = pooler_type
        )

        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # for name, module in box_head.named_modules():
        #     print(name, ' | ', module)
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        # for name, module in box_predictor.named_modules():
        #     print(name, ' | ', module)
        return {
            'box_in_features': in_features,
            'box_pooler': box_pooler,
            'box_head': box_head,
            'box_predictor': box_predictor
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret
    
    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals : List[Instances],
        targets: Optional[List[Instances]] = None
        ):
        del images
        if self.training:
            assert targets, 'targets argument is required during training'
        del targets

        if self.training:
            print('training')
            pass
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        assert not self.training
        assert instances[0].has('pred_boxes') and instances[0].has('pred_classes')

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)

        return instances

    def _forward_box(self, features:Dict[str, torch.Tensor], proposals: List[Instances]):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        # print(predictions[0].shape)
        # print(predictions[1].shape)
        del box_features

        if self.training:
            pass
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
    
    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        if not self.mask_on:
            if self.training:
                return {}
            else:
                return instances

        if self.training:
            print('train')
            # instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            # print(boxes[0].tensor.shape)
            features = self.mask_pooler(features, boxes)
        else:
            features = dict([(f, features[f]) for f in self.mask_in_features])
        return self.mask_head(features, instances)

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        if not self.keypoint_on:
            if self.training:
                return {}
            else:
                return instances



def build_roi_heads(cfg, input_shape):
    name = cfg.MODEL.ROI_HEADS.NAME
    return StandardROIHeads(cfg, input_shape)

# from model import input1 as images
# from model import outputs as features
# from rpn import proposals
# roi_head = build_roi_heads(cfg,input_shape).eval()
# roi_head(images, features, proposals)
# print(roi_head.__class__.__name__)
# shape = ShapeSpec(
#                 channels=256, width=14, height=14
#             )
# build_mask_head(cfg, shape)

