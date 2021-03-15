# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
import torch.nn as nn
from config import configurable
from build_cfg import cfg
from shape_spec import ShapeSpec
import math
from typing import List,Tuple
from boxes import Boxes
import torch.nn.functional as F

input_shape = [ShapeSpec(channels=256, height=None, width=None, stride=4), 
                ShapeSpec(channels=256, height=None, width=None, stride=8), 
                ShapeSpec(channels=256, height=None, width=None, stride=16), 
                ShapeSpec(channels=256, height=None, width=None, stride=32), 
                ShapeSpec(channels=256, height=None, width=None, stride=64)]

def _broadcast_params(params, num_features, name):
    assert isinstance(params, (tuple, list)), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], (tuple,list)):
        return [params]*num_features
    if(len(params) == 1):
        return list(params)*num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
        )
    return params

def _create_grid_offsets(size:List[int], stride:int, offset:float, device:torch.device):
    grid_height, gird_width = size
    shift_x = torch.arange(
        offset*stride, gird_width*stride, step=stride, dtype=torch.float32, device=device
        )
    shift_y = torch.arange(
        offset*stride, grid_height*stride, step=stride, dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x,shift_y

class BufferList(nn.Module):
    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(i), buffer)
    
    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

class DefaultAnchorGenerator(nn.Module):

    box_dim: torch.jit.Final[int] = 4

    @configurable
    def __init__(self,*,sizes, aspect_ratios, strides, offset=0.5):
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features, 'sizes')
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, 'aspect_ratios')
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        self.offset = offset
        assert 0.0 <= self.offset <= 1.0, f'offset value should in [0,1], get {self.offset}'


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'sizes':cfg.MODEL.ANCHOR_GENERATOR.SIZES,
            'aspect_ratios':cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            'strides':[x.stride for x in input_shape],
            'offset':cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
        }

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchor(s,a).float() for s,a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    @torch.jit.unused
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    @torch.jit.unused
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self,grid_sizes: List[List[int]] = [[62,62],[30,30],[14,14]]):
        anchors = []
        buffer: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for size, stride, base_anchor in zip(grid_sizes, self.strides, buffer):
            # print(size,'|',stride,'|',base_anchor)
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchor.device)
            shifts = torch.stack((shift_x,shift_y,shift_x,shift_y),dim=1)

            anchors.append((shifts.view(-1,1,4) + base_anchor.view(1,-1,4)).reshape(-1,4))
            # print(anchors[-1].shape)
        return anchors

    def generate_cell_anchor(self, sizes=(32,64,128,256,512), aspect_ratios=(0.5,1.0,2.0)):
        anchors = []
        for size in sizes:
            area = size ** 2
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = w * aspect_ratio
                x0,y0,x1,y1 = -w/2.0, -h/2.0, w/2.0, h/2.0
                anchors.append([x0,y0,x1,y1])
        return torch.tensor(anchors)

    def forward(self, features: List[torch.Tensor]):
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]

def build_anchor_generator(cfg, input_shape):
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME
    return DefaultAnchorGenerator(cfg, input_shape)

# anchor_generator = DefaultAnchorGenerator(cfg,input_shape)
# num_anchors = anchor_generator.num_anchors
# print(num_anchors)
# anchor_generator._grid_anchors()
# print(anchor_generator)
# for name in anchor_generator.cell_anchors.named_buffers():
#     print(name[1].device)

# A = torch.tensor([1,2,3,4,5,6,7,8,9])[:]
# print(float("-inf"))
# x = torch.arange(2,20,step=2)
# y = torch.arange(2,10,step=2)
# x,y = torch.meshgrid(x,y)
# x=x.reshape(-1)
# y=y.reshape(-1)
# print(torch.stack((x,y,x,y),dim=1))
# A = torch.rand((3,256,256))
# print(A.shape[1:-3])
# A = (A[:,0]>0) & (A[:,1]>0)
# print(A)