from boxes import Boxes
from instances import Instances
from pickle import TUPLE
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from config import configurable
from box_regression import Box2BoxTransform
from shape_spec import ShapeSpec
from typing import List, Tuple, Union, Dict
import torch.nn.functional as F
from nms import batched_nms

def fast_rcnn_inference(
    boxes:List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    ):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        ) for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

def fast_rcnn_inference_single_image(
    boxes:torch.Tensor, 
    scores:torch.Tensor,
    image_shape:Tuple[int, int],
    score_thresh:float,
    nms_thresh: float,
    topk_per_image:int,
    ):
    vaild_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not vaild_mask.all():
        scores = scores[vaild_mask]
        boxes = boxes[vaild_mask]

    scores = scores[:,:-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)
    
    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero()

    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:,0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    # print('result boxes shape:', result.pred_boxes.tensor.shape)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

class FastRCNNOutputLayers(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape:ShapeSpec,
        *,
        box2box_transform,
        num_classes:int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        ):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.cls_score = Linear(input_size, self.num_classes+1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)
        
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {'loss_cls': loss_weight, 'loss_box_reg': loss_weight}
        self.loss_weight = loss_weight

    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            'input_shape': input_shape,
            'box2box_transform': Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'cls_agnostic_bbox_reg': cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
        }
        return ret

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals:List[Instances]):
        boxes = self.predict_boxes(predictions, proposals)
        # print(boxes[0].shape) # [1000, 320]
        scores = self.predict_probs(predictions, proposals)
        # print(scores[0].shape) # [1000,81]
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image
        )

    def predict_boxes(
        self, preditions: Tuple[torch.Tensor, torch.Tensor],  proposals: List[Instances]
        ):
        if not len(proposals):
            return []
        _, proposal_deltas = preditions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = Boxes.cat(proposal_boxes).tensor
        # print(proposal_deltas.shape)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes
        )
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions:Tuple[torch.Tensor, torch.Tensor], proposals:List[Instances]
        ):
            scores, _ = predictions
            # print(scores.shape)
            num_inst_pre_image = [len(p) for p in proposals]
            probs = F.softmax(scores, dim=-1)
            # print(probs.shape)
            return probs.split(num_inst_pre_image, dim=0)
