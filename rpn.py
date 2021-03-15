import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shape_spec import ShapeSpec
from config import configurable
import yaml
from build_cfg import cfg
from anchor_generator import build_anchor_generator
from box_regression import Box2BoxTransform
from matcher import Matcher
from image_lists import ImageList
from instances import Instances
# from model import outputs
from proposal_utils import find_top_rpn_proposals

input_shape = {
                'p2': ShapeSpec(channels=256, height=None, width=None, stride=4), 
                'p3': ShapeSpec(channels=256, height=None, width=None, stride=8), 
                'p4': ShapeSpec(channels=256, height=None, width=None, stride=16), 
                'p5': ShapeSpec(channels=256, height=None, width=None, stride=32), 
                'p6': ShapeSpec(channels=256, height=None, width=None, stride=64)
                }

def build_rpn_head(cfg, input_shape):
    return StandardRPNHead(cfg, input_shape)

class StandardRPNHead(nn.Module):
    @configurable
    def __init__(self,*,in_channels:int, num_anchors:int, box_dim:int=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors*box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)


    @classmethod
    def from_config(cls, cfg, input_shape):
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1,  "Each level must have the same channel!"
        in_channels = in_channels[0]
        anchor_generator = build_anchor_generator(cfg,input_shape)

        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim}

    def forward(self,features:List[torch.Tensor]):
        perd_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu_(self.conv(x))
            perd_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return perd_objectness_logits, pred_anchor_deltas

class RPN(nn.Module):
    @configurable
    def __init__(
                self,
                *,
                in_features:List[str],
                head:nn.Module,
                anchor_generator:nn.Module,
                anchor_matcher:Matcher,
                box2box_transform:Box2BoxTransform,
                batch_size_per_image:int,
                positive_fraction: float,
                pre_nms_topk: Tuple[float, float],
                post_nms_topk: Tuple[float, float],
                nms_thresh: float = 0.7,
                min_box_size: float = 0.0,
                anchor_boundary_thresh: float = -1.0,
                loss_weight: Union[float, Dict[str, float]] = 1.0,
                box_reg_loss_type: str = "smooth_l1",
                smooth_l1_beta: float = 0.0,
                ):
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta

    @classmethod
    def from_config(cls,cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        # print(ret)
        return ret

    def forward(self,images:ImageList, features:Dict[str, torch.Tensor], gt_instances:Optional[List[Instances]] = None):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        #reshape from (N,Anchor,H,W) -> (N,H,W,Anchor) -> (N,H*W*Anchor)
        pred_objectness_logits = [
            score.permute(0,2,3,1).flatten(1)
            for score in pred_objectness_logits
            ]
        #reshape from (N,Anchor*box_dim,H,W) -> (N,Anchor,Box_dim,H,W) -> (N,H,W,Anchor,Box_dim) -> (N,H*W*Anchor,Box_dim)
        pred_anchor_deltas = [
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]).permute(0,3,4,1,2).flatten(1,-2)
            for x in pred_anchor_deltas
        ]
        if self.training:
            losses = {}
            pass
        else:
            losses = {}
        proposals = self.predict_proposals(anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes)
        return proposals, losses

    def predict_proposals(self, anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes):
        with torch.no_grad():
            # print(pred_objectness_logits[0].shape)
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training
            )
    
    def _decode_proposals(self, anchors, pred_anchor_deltas):
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1,B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N,-1,-1).reshape(-1,B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            proposals.append(proposals_i.view(N,-1,B))
            # print(proposals[-1][0,:2])
            # print(proposals[-1][[0],[1,1]])
        return proposals

def build_proposal_generator(cfg, input_shape):
    name = cfg.MODEL.PROPOSAL_GENERATOR.NAME
    if name == 'PrecomputeProposals':
        return None
    return RPN(cfg, input_shape)                

# rpn = RPN(cfg, input_shape).eval()
# # features = [outputs[f] for f in cfg.MODEL.RPN.IN_FEATURES]
# # grid_size = [feature.shape[-2:] for feature in features]
# # print(grid_size)
# images = ImageList.from_tensors([torch.rand((3,480,640)), torch.rand((3,480,640))])
# proposals, losses = rpn(images, outputs)
# print(proposals[0])
# boxes = torch.tensor([[1,1,2,2],[0.5,1,2,2.5], [1,1,2,2.5],[3,3,5,5],[3,3,4.5,5.5],[3.5,3.5,5,6]])
# # print(boxes)
# max_coordinates = boxes.max()
# idxs = torch.tensor([0,0,0,1,1,1])
# offsets = idxs.to(boxes) * (max_coordinates + torch.tensor(1).to(boxes))
# # print(offsets)
# boxes_for_nms = boxes + offsets[:,None]
# # print(boxes_for_nms)
# scores = torch.tensor([0.9,0.98,0.95,0.85,0.89,0.80])
# result_mask = scores.new_zeros(scores.size(),dtype=torch.bool)
# print(result_mask)
# from torchvision.ops import nms
# keep = nms(boxes_for_nms, scores, 0.3)
# print(keep)
# a = torch.rand((3,4)).float()
# idx = torch.arange(10)
# print(idx.dtype)
# idx = idx.to(a)
# print(idx.dtype)
# print(a)
# print(a.max())
# print(x[[True,True,False,False,True,True]])
# y = torch.rand((2,3))
# print(torch.cat([x,y], dim=1))
# a = torch.arange(2)
# print(a[:,None])
# print(x[[[0],[1]],[[0],[2]]])
# x, idx = x.sort(descending=True,dim=1)
# print(x)
# print(x[:,:2])
# print(idx[:,:2])
# y = torch.rand((4))
# print(y)
# z = x + y[:,None]
# print(z)
# x = torch.tensor([[1,2,3],[4,5,6]])
# print(x[:,:,None].shape)
# A = torch.rand((2,2,2,2))
# print(A.shape)
# print(A.flatten(1,-3).shape)
# pre = {True:18}
# print(pre)
# print(cfg.MODEL.ANCHOR_GENERATOR.NAME)
# print([input_shape[f].stride for f in cfg.MODEL.RPN.IN_FEATURES])
# rpn = StandardRPNHead(cfg)
# print(cfg.MODEL.RPN.IN_FEATURES)
# for f in cfg.MODEL.RPN.IN_FEATURES:
#     print(f)
# A = (1,2,3)
# in_features = cfg.MODEL.RPN.IN_FEATURES
# print(in_features)
# print[input_shape[f] for f in in_features]