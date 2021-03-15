from typing import List
import torch
import torch.nn as nn
from torch.nn.modules import activation, conv, padding
from config import configurable
from shape_spec import ShapeSpec
from typing import List
from model import Conv2d, get_norm
import numpy as np
import fvcore.nn.weight_init as weight_init

class FastRCNNConvFCHead(nn.Sequential):
    @configurable
    def __init__(self, input_shape:ShapeSpec, *, conv_dims:List[int], fc_dims:List[int], conv_norm='' ):
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size = 3,
                padding = 1, 
                bias = not conv_norm,
                norm = get_norm(conv_norm, conv_dim),
                activation = nn.ReLU(),
            )
            self.add_module('conv{}'.format(k+1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module('flatten', nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module('fc{}'.format(k+1), fc)
            self.add_module('fc_relu{}'.format(k+1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(self, cfg, input_shape:ShapeSpec):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv__dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            'input_shape': input_shape,
            'conv_dims': [conv__dim] * num_conv,
            'fc_dims': [fc_dim] * num_fc,
            'conv_norm': cfg.MODEL.ROI_BOX_HEAD.NORM
        }

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    @property
    def output_shape(self):
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

def build_box_head(cfg, input_shape):
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return FastRCNNConvFCHead(cfg, input_shape)