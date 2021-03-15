import math
import torch
from typing import List, Tuple
from boxes import Boxes
from nms import batched_nms
from instances import Instances

def cat(tensors:List[torch.Tensor], dim:int=0):
    assert isinstance(tensors, (list, tuple))
    if(len(tensors) == 1):
        return tensors[0]
    return torch.cat(tensors, dim=dim)

def nonzero_tuple(x):
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)

def find_top_rpn_proposals(
    proposals:List[torch.Tensor],
    pred_objectness_logits:List[torch.Tensor],
    images_sizes:List[Tuple[int,int]],
    nms_thresh:float,
    pre_nms_topk:int,
    post_nms_topk:int,
    min_box_size:float,
    training:bool
):
    num_images = len(images_sizes)
    device = proposals[0].device

    topk_scores = []
    topk_proposals = []
    level_ids = []
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        #proposals_i:(N, H*W*A,B)  logits_i:(N, H*W*A)
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_score_i = logits_i[batch_idx, :num_proposals_i] #(N, NUM_PROPOSAL)
        topk_idx = idx[batch_idx, :num_proposals_i]  #(N, NUM_PROPOSAL)

        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx] #(N,topk,B)

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_score_i)
        level_ids.append(torch.full((num_proposals_i,),level_id, dtype=torch.int64, device=device)) #(NUM_PROPOSAL)

    #Concat all levels 
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals,dim=1)
    level_ids = cat(level_ids, dim=0)

    #3 For each image, run a per-level NMS, and choose topk results
    results:List[Instances] = []
    for n, images_size in enumerate(images_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(images_size)

        #filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size) #(NUM_PROPOSALS*LEVEL, 1)
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]
        # print(type(boxes))
        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]

        res = Instances(images_size)
        res._name = 'test'
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


        



    




