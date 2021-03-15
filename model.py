import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from file_io import PathManager
import pickle
from backbone import Backbone
from blocks import CNNBlockBase,FrozenBatchNorm2d
from blocks import get_norm
from build_cfg import cfg
from fvcore.nn import weight_init 
import math

from shape_spec import ShapeSpec

class Conv2d(torch.nn.Conv2d):
    def __init__(self,*args, **kwargs):
        norm = kwargs.pop("norm",None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args,**kwargs)
        self.norm = norm
        self.activation = activation
        # if isinstance(self.norm, nn.BatchNorm2d):
        #     print('True!') 

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        
        return out

class BasicStem(CNNBlockBase):
    def __init__(self, in_channels=3, out_channels=64, norm = 'BN'):
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        # print(self.in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False, norm = get_norm(norm,out_channels))
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        # print('after stem shape: ',x.shape)
        return out

class BottleneckBlock(CNNBlockBase):
    def __init__(self, in_channels, out_channels, bottleneck_channels,
                *, stride=1, num_groups=1, norm = 'BN', 
                stride_in_1x1=False, dilation=1,):
                super().__init__(in_channels,out_channels,stride)
                if in_channels != out_channels:
                    self.shortcut = Conv2d(in_channels, out_channels, 
                                            kernel_size=1, stride=stride, bias=False, 
                                            norm = get_norm(norm,out_channels))
                else:
                    self.shortcut = None

                stride_1x1, stride_3x3 = (stride,1) if stride_in_1x1 else (1, stride)

                self.conv1 = Conv2d(in_channels, bottleneck_channels, 
                                            kernel_size=1, stride=stride_1x1, bias=False, 
                                            norm = get_norm(norm,bottleneck_channels))
                self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, 
                                            kernel_size=3, stride=stride_3x3, bias=False, padding = 1*dilation, dilation=dilation, groups=num_groups,
                                            norm = get_norm(norm,bottleneck_channels))
                self.conv3 = Conv2d(bottleneck_channels, out_channels, 
                                            kernel_size=1, stride=1, bias=False, 
                                            norm = get_norm(norm,out_channels))
                """
                just need initilize weight!
                """
                

    def forward(self, x):
        out = F.relu_(self.conv1(x))
        out = F.relu_(self.conv2(out))
        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu_(out)

        return out

class ResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes
        
        current_stride = self.stem.stride
        self._out_features_strides = {'stem': current_stride}
        self._out_features_channels = {'stem': self.stem.out_channels}

        self.stage_names, self.stages = [],[]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = 'res' + str(i+2)
            stage = nn.Sequential(*blocks)
            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_features_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_features_channels[name] = current_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.linear = nn.Linear(current_channels, num_classes)
            nn.init.normal_(self.linear.weight, std=0.01)
            name = 'linear'
        
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        
    def forward(self, x):
        assert x.dim() == 4, 'input dimension must be 4!'
        outputs = {}
        out = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = out
        for name, stage in zip(self.stage_names, self.stages):
            out = stage(out)
            if name in self._out_features:
                outputs[name] = out
        if self.num_classes is not None:
            out = self.avgpool(out)
            out = torch.flatten(out,1)
            out = self.linear(out)
            if 'linear' in self._out_features:
                outputs['linear'] = out
        
        return outputs

    def output_shape(self):
        return {
                name:
                ShapeSpec(channels=self._out_features_channels[name], stride=self._out_features_strides[name]) 
                for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        if freeze_at>0:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, first_stride=None, *, in_channels, out_channels, **kwargs):
        # print(kwargs)
        # print(num_blocks)
        if first_stride is not None:
            assert 'stride' not in kwargs and 'stride_per_block' not in kwargs
            kwargs['stride_per_block'] = [first_stride] + [1]*(num_blocks-1)
        
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k,v in kwargs.items():
                if k.endswith('_per_block'):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[:-len('_per_block')]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs))
            in_channels = out_channels

        return blocks

def _assert_stride_are_log2_contiguous(strides):
    for i, stride in enumerate(strides[1:], 1):
        assert stride == strides[i-1]*2, 'Strides {} {} are not log2 contiguous'.format(stride, strides[i-1])

class FPN(Backbone):
    def __init__(self, bottom_up, in_features, out_channels, norm='', top_block=None, fuse_type='sum'):
        super().__init__()
        assert isinstance(bottom_up, Backbone)
        
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_features = [input_shapes[f].channels for f in in_features]
        _assert_stride_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ''
        for idx, in_channels in enumerate(in_channels_per_features):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=use_bias, norm=output_norm)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module('fpn_lateral{}'.format(stage), lateral_conv)
            self.add_module('fpn_output{}'.format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        self._out_features_strides = {'p{}'.format(int(math.log2(s))):s for s in strides}
        if self.top_block is not None:
            for s in range(stage, stage+self.top_block.num_levels):
                self._out_features_strides['p{}'.format(s+1)] = 2**(s+1)
        
        self._out_features = list(self._out_features_strides.keys())
        self._out_features_channels = {k:out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {'avg', 'sum'}, 'Fuse type should be avg or sum!'
        self._fuse_type = fuse_type

        self.rev_in_features = tuple(in_features[::-1])

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self,x):
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        for features, lateral_conv, output_conv in zip(
            self.rev_in_features[1:], self.lateral_convs[1:], self.output_convs[1:]
            ):
            features = bottom_up_features[features]
            top_down_features = F.interpolate(prev_features,scale_factor=2.0,mode='nearest')
            lateral_features = lateral_conv(features)
            prev_features = top_down_features + lateral_features
            if self._fuse_type == 'avg':
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_features in bottom_up_features:
                top_block_in_features = bottom_up_features[self.top_block.in_features]
            else:
                top_block_in_features = results[self._out_features.index(self.top_block.in_features)]
            results.extend(self.top_block(top_block_in_features))
        assert len(self._out_features) == len(results)
        return dict(list(zip(self._out_features, results)))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_features_channels[name],stride=self._out_features_strides[name]
                )  
            for name in self._out_features
        }

class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_features = 'p5'

    def forward(self,x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

def build_resnet_backbone(cfg, input_shape):
    norm = cfg.MODEL.RESNETS.NORM
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    stem = BasicStem(in_channels=input_shape.channels, out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,norm=norm)
    num_blocks_per_stage = {
        18: [2,2,2,2],
        34: [3,4,6,3],
        50: [3,4,6,3],
        101: [3,4,23,3],
        152: [3,8,36,3],
    }[depth]
    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(deform_on_per_stage), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features if f != "stem"
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx+1)):
        # print(idx,': ',stage_idx)
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            'num_blocks': num_blocks_per_stage[idx],
            'stride_per_block': [first_stride] + [1]*(num_blocks_per_stage[idx] - 1),
            'in_channels': in_channels,
            'out_channels': out_channels,
            'norm': norm
        }
        #Use BasicBlock for R18 and R34
        if depth in [18,34]:
            stage_kargs['block_class'] = BasicBlock
        else:
            stage_kargs['bottleneck_channels'] = bottleneck_channels
            stage_kargs['stride_in_1x1'] = stride_in_1x1
            stage_kargs['dilation'] = dilation
            stage_kargs['num_groups'] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs['block_class'] = DeformBottleneckBlock
                stage_kargs['deform_modulated'] = deform_modulated
                stage_kargs['deform_num_groups'] = deform_num_groups
            else:
                stage_kargs['block_class'] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)

def build_resnet_fpn_backbone(cfg,input_shape:ShapeSpec):
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

# input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

# FPN_RESNET = build_resnet_fpn_backbone(cfg, input_shape)

# # input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
# input1 = torch.rand((2,3,256,256))
# outputs = FPN_RESNET(input1)
# for k,v in outputs.items():
#     print(k,'->',v.shape)
# print(FPN_RESNET.output_shape())

# input1 = torch.rand((1,1,4,4))
# print(input1)
# input1 = F.interpolate(input1,scale_factor=2.0,mode='nearest')
# print(input1)
# print(resnet.output_shape())

# # print(input1.shape)
# model = build_resnet_backbone(cfg, input_shape)
# print(model.output_shape())
# outputs = model(input1)

# A = [1,2,3]
# A.insert(1,4)
# print(A)
# for s in model.output_shape():
#     print(s.channels)
# bottle = BottleneckBlock(64,256,64)
# conv1 = Conv2d(64, 256, kernel_size=7, stride=2, padding=3, bias=False, norm = FrozenBatchNorm2d(256))
# bn_module = nn.modules.batchnorm
# bn_module = (bn_module.BatchNorm2d,bn_module.SyncBatchNorm)
# if isinstance(conv1,bn_module):
#     print('true')
# else:
#     for name, child in conv1.named_children():
#         print(name)

# net = ResNet50()
# for name,child in net.named_children():
#     print(name)
#     print('----------')
#     if name in ['layer1']:
#         for name_son,child_son in child.named_children():
#             print(name_son,'->', child_son)
# print('net: ',net.state_dict()['conv1.weight'].shape)
# for k,v in net.state_dict().items():
#     # print(item)
#     # k,v = item
#     print(idx,'->',k,': ',v.shape)
#     idx+=1

# with PathManager.open('/home/poseidon/Downloads/model_final_f10217.pkl','rb') as f:
# cnt = 0
# with open('/home/poseidon/Downloads/model_final_f10217.pkl','rb') as f:
#     try:
#         data = pickle.load(f,encoding='latin1')
#         net.state_dict()['conv1.weight'] = torch.from_numpy(data['model']['backbone.bottom_up.stem.conv1.weight'])
#         # for k,v in data['model'].items():
#         #     print(cnt,'->',k,': ',v.shape)
#         #     cnt+=1
#     except EOFError:
#         print("None data!")
# for name, parameters in net.named_parameters():
#     print(name)
#     if name in ['conv1','conv2','bias']:
#         print(parameters)
# layers = [1] * 10
# print(*layers)
# model = torch.load('/home/poseidon/Downloads/model_final_f10217.pkl')
# print(model)
# A = torch.DoubleTensor(([0.028252628])).numpy()
# print(A)

# _C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
# _C.MODEL.RPN.IN_FEATURES = ["res4"]
# _C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
# _C.MODEL.RPN.NMS_THRESH = 0.7
# _C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# _C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# _C.MODEL.RPN.LOSS_WEIGHT = 1.0
# _C.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
# _C.MODEL.RPN.BOUNDARY_THRESH = -1
# _C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# _C.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
# _C.MODEL.RPN.SMOOTH_L1_BETA = 0.0
# _C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
# _C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
# _C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
# _C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
# _C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
# _C.MODEL.RPN.IOU_LABELS = [0, -1, 1]
# _C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
# _C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"