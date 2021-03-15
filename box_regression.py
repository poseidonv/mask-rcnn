import math
import torch

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

class Box2BoxTransform(object):
    def __init__(self, weights, scale_clamp = _DEFAULT_SCALE_CLAMP):
        self.weights = weights
        self.scale_clamp = scale_clamp

    def apply_deltas(self, deltas:torch.Tensor, boxes:torch.Tensor):
        deltas = deltas.float()
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:,2] - boxes[:,0]
        heights = boxes[:,3] - boxes[:,1]
        ctr_x = boxes[:,0] + widths*0.5
        ctr_y = boxes[:,1] + heights*0.5

        wx,wy,ww,wh = self.weights
        dx = deltas[:,0::4] / wx
        dy = deltas[:,1::4] / wy
        dw = deltas[:,2::4] / ww
        dh = deltas[:,3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx*widths[:,None] + ctr_x[:,None]
        pred_ctr_y = dy*heights[:,None] + ctr_y[:,None]
        pred_w = torch.exp(dw)*widths[:,None]
        pred_h = torch.exp(dh)*heights[:,None]

        # print(pred_ctr_x.shape,'|',pred_ctr_y.shape,'|',pred_w.shape,'|',pred_h.shape)
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:,0::4] = pred_ctr_x - pred_w*0.5
        pred_boxes[:,1::4] = pred_ctr_y - pred_h*0.5
        pred_boxes[:,2::4] = pred_ctr_x + pred_w*0.5
        pred_boxes[:,3::4] = pred_ctr_y + pred_h*0.5
        return pred_boxes
