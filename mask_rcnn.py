from boxes import Boxes
import sys

from numpy.lib.type_check import imag
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from typing import Dict, List, Optional, Tuple
from read_image import read_image
from torch.nn.modules import module
import torch
import torch.nn as nn
from model import build_resnet_fpn_backbone
from rpn import build_proposal_generator
from roi_heads import build_roi_heads
from config import configurable
from shape_spec import ShapeSpec
from build_cfg import cfg
from backbone import Backbone
import pickle
from instances import Instances
from image_lists import ImageList
from postprocessing import detector_postprocess
from visualizer import ColorMode, Visualizer

from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
# from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

class GeneralizedRCNN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int=0,
        ):
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1,1,1))
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1,1,1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        backbone = build_resnet_fpn_backbone(cfg, input_shape)
        return {
            'backbone':backbone,
            'proposal_generator':build_proposal_generator(cfg, backbone.output_shape()),
            'roi_heads': build_roi_heads(cfg, backbone.output_shape()),
            'input_format': cfg.INPUT.FORMAT,
            'vis_period': cfg.VIS_PERIOD,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

    def inference(
        self, 
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True
        ):
        assert not self.training

        images = self.prepocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            pass

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def prepocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
    
    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get('height', image_size[0])
            # height = 480
            width = input_per_image.get('width', image_size[1])
            # width = 640
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

mask_rcnn = GeneralizedRCNN(cfg).to(torch.device(cfg.MODEL.DEVICE))
mask_rcnn = mask_rcnn.eval()

# input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
# fpn = build_resnet_fpn_backbone(cfg, input_shape).to(torch.device(cfg.MODEL.DEVICE))
# fpn = fpn.eval()
if len(cfg.DATASETS.TEST):
    print(cfg.MODEL.WEIGHTS)

checkpointer = DetectionCheckpointer(mask_rcnn)
checkpointer.load(cfg.MODEL.WEIGHTS)

# for name,module in mask_rcnn.named_modules():
#     if name in ['backbone.bottom_up.res2.0.conv1']:
#         print(module.weight[:3,:3])

aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
import time
# start_time = time.time()
#1341846313.553992.png  1341846313.654184.png
img_path = '/home/poseidon/Downloads/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.654184.png'
original_image = read_image(img_path,format='BGR')
with torch.no_grad():
    height, width = original_image.shape[:2]
    image = original_image
    image = aug.get_transform(image).apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    # inputs = {"image": image, "height": height, "width": width}
    inputs = [{"image": image}]

    class WrapModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.torch_model = mask_rcnn

        def forward(self, image):
            inputs = [{'image': image}]
            outputs = self.torch_model.inference(inputs, do_postprocess=False)[0]
            outputs = outputs.get_fields()
            from detectron2.utils.analysis import _flatten_to_tuple
            out = []
            for _,v in outputs.items():
                if isinstance(v, Boxes):
                    v = v.tensor
                out.extend(_flatten_to_tuple(v))
            
            return out

    # ts_model = torch.jit.trace(WrapModel(), (image,))
    # ts_model.save('susiusi.ts')
    # print('=======')
    # traced_script_module = torch.jit.trace(WrapModel(), [inputs])
    # exit(-1)

    # output = traced_script_module(torch.ones(1, 3, 224, 224))

    # traced_script_module.save("model.pt")
    start_time = time.time()
    predictions = mask_rcnn(inputs)[0]
    print(time.time()-start_time)
image = original_image[:,:,::-1]

metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

cpu_device = torch.device('cpu')

if "instances" in predictions:
    instances = predictions["instances"].to(cpu_device)
    vis_output = visualizer.draw_instance_predictions(predictions=instances)
output_filename = '/home/poseidon/Documents/my_detectron2/output_image.jpg'
vis_output.save(output_filename)
print(time.time()-start_time)
# image = aug.get_transform(original_image).apply_image(original_image)
# print(image.shape)
# cnt = 0
# for name, module in mask_rcnn.named_modules():
#     print(cnt,'->',name)
#     cnt+=1
# with open('/home/poseidon/Downloads/model_final_f10217.pkl','rb') as f:
#     try:
#         data = pickle.load(f,encoding='latin1')
#         # net.state_dict()['conv1.weight'] = torch.from_numpy(data['model']['backbone.bottom_up.stem.conv1.weight'])
#         for k,v in data['model'].items():
#             print(cnt,'->',k,': ',v.shape)
#             cnt+=1
#     except EOFError:
#         print("None data!")