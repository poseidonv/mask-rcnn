from torch import nn
from torchvision.ops import roi_align as tv_roi_align

try:
    from torchvision import __version__
    USE_TORCHVISION = tuple(int(x) for x in __version__.split('.')[:2])
    USE_TORCHVISION = USE_TORCHVISION >= (0,7)
except ImportError:
    USE_TORCHVISION = True

if USE_TORCHVISION:
    roi_align = tv_roi_align

class ROIAlign(nn.Module):
    def __init__(
        self,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned = True
        ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        assert rois.dim() == 2 and rois.size(1) == 5
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned
        )


    def __repr__(self):
        tmpstr = self.__class__.__name__+'('
        tmpstr += 'output_size' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ', aligned='+ str(self.aligned) + ')'
        return tmpstr