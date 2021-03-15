from typing import List
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms

def batched_nms(
    boxes:torch.Tensor,
    scores:torch.Tensor,
    idxs:torch.Tensor,
    iou_threshold:float
    ):
    assert boxes.shape[-1] == 4
    if (len(boxes) < 40000):
        return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)