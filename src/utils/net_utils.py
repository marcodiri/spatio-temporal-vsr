import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import gaussian


# -------------------- utility functions -------------------- #
def initialize_weights(net_l, init_type="kaiming", scale=1):
    """Modify from BasicSR/MMSR"""

    if not isinstance(net_l, list):
        net_l = [net_l]

    for net in net_l:
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                else:
                    raise NotImplementedError(init_type)

                m.weight.data *= scale  # to stabilize training

                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)


def space_to_depth(x, scale=4):
    """Equivalent to tf.space_to_depth()"""

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output


def backward_warp(x, flow, mode="bilinear", padding_mode="border"):
    """Backward warp `x` according to `flow`

    Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

    Reference:
        https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat(
        [flow[:, 0:1, ...] / ((w - 1.0) / 2.0), flow[:, 1:2, ...] / ((h - 1.0) / 2.0)],
        dim=1,
    )

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default in PyTorch version
    #        lower than 1.4.0
    if int("".join(torch.__version__.split(".")[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True
        )
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


def get_upsampling_func(scale=4, degradation="BI"):
    if degradation == "BI":
        upsample_func = functools.partial(
            F.interpolate, scale_factor=scale, mode="bilinear", align_corners=False
        )

    elif degradation == "BD":
        upsample_func = BicubicUpsample(scale_factor=scale)

    else:
        raise ValueError("Unrecognized degradation: {}".format(degradation))

    return upsample_func


# --------------------- utility classes --------------------- #
class BicubicUpsample(nn.Module):
    """A bicubic upsampling class with similar behavior to that in TecoGAN-Tensorflow

    Note that it's different from torch.nn.functional.interpolate and
    matlab's imresize in terms of bicubic kernel and sampling scheme

    Theoretically it can support any scale_factor >= 1, but currently only
    scale_factor = 4 is tested

    References:
        The original paper: http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
        https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()

        # calculate weights
        cubic = torch.FloatTensor(
            [
                [0, a, -2 * a, a],
                [1, 0, -(a + 3), a + 2],
                [0, -a, (2 * a + 3), -(a + 2)],
                [0, 0, a, -a],
            ]
        )  # accord to Eq.(6) in the reference paper

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s**2, s**3]))
            for s in [1.0 * d / scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer("kernels", torch.stack(kernels))

    def forward(self, input):
        n, c, h, w = input.size()
        s = self.scale_factor

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode="replicate")

        # calculate output (height)
        kernel_h = self.kernels.repeat(c, 1).view(-1, 1, s, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0, groups=c)
        output = (
            output.reshape(n, c, s, -1, w + 3)
            .permute(0, 1, 3, 2, 4)
            .reshape(n, c, -1, w + 3)
        )

        # calculate output (width)
        kernel_w = self.kernels.repeat(c, 1).view(-1, 1, 1, s)
        output = F.conv2d(output, kernel_w, stride=1, padding=0, groups=c)
        output = (
            output.reshape(n, c, s, h * s, -1)
            .permute(0, 1, 3, 4, 2)
            .reshape(n, c, h * s, -1)
        )

        return output


def get_state_dict(filter_size=5, std=1.0, map_func=lambda x: x):
    generated_filters = (
        gaussian(filter_size, std=std).reshape([1, filter_size]).astype(np.float32)
    )

    gaussian_filter_horizontal = generated_filters[None, None, ...]

    gaussian_filter_vertical = generated_filters.T[None, None, ...]

    sobel_filter_horizontal = np.array(
        [[[[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]]], dtype="float32"
    )

    sobel_filter_vertical = np.array(
        [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]], dtype="float32"
    )

    directional_filter = np.array(
        [
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        dtype=np.float32,
    )

    connect_filter = np.array(
        [[[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]], dtype=np.float32
    )

    hysteresis = np.ones((1, 1, 3, 3), dtype=np.float32) + 0.25

    return {
        "gaussian_filter_horizontal.weight": map_func(gaussian_filter_horizontal),
        "gaussian_filter_vertical.weight": map_func(gaussian_filter_vertical),
        "sobel_filter_horizontal.weight": map_func(sobel_filter_horizontal),
        "sobel_filter_vertical.weight": map_func(sobel_filter_vertical),
        "directional_filter.weight": map_func(directional_filter),
        "connect_filter.weight": map_func(connect_filter),
        "hysteresis.weight": map_func(hysteresis),
    }


# https://github.com/jm12138/CannyDetector/
class CannyDetector(nn.Module):
    def __init__(self, filter_size=5, std=1.0):
        super(CannyDetector, self).__init__()

        self.gaussian_filter_horizontal = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, filter_size),
            padding=(0, filter_size // 2),
            bias=False,
        )
        self.gaussian_filter_vertical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(filter_size, 1),
            padding=(filter_size // 2, 0),
            bias=False,
        )

        self.sobel_filter_horizontal = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )
        self.sobel_filter_vertical = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )

        self.directional_filter = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False
        )

        self.connect_filter = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )

        self.hysteresis = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )

        params = get_state_dict(
            filter_size=filter_size,
            std=std,
            map_func=lambda x: torch.from_numpy(x).to(self.device),
        )
        self.load_state_dict(params)

        for param in self.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, img, threshold1=10.0, threshold2=100.0):
        img_r = img[:, 0:1]  # red channel
        img_g = img[:, 1:2]  # green channel
        img_b = img[:, 2:3]  # blue channel

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = torch.atan2(
            grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b
        ) * (180.0 / math.pi)
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        batch, _, height, width = inidices_positive.shape
        pixel_count = height * width * batch
        pixel_range = torch.Tensor([range(pixel_count)]).to(self.device)

        indices = (
            inidices_positive.reshape((-1,)) * pixel_count + pixel_range
        ).squeeze()
        channel_select_filtered_positive = all_filtered.reshape((-1,))[
            indices.long()
        ].reshape((batch, 1, height, width))

        indices = (
            inidices_negative.reshape((-1,)) * pixel_count + pixel_range
        ).squeeze()
        channel_select_filtered_negative = all_filtered.reshape((-1,))[
            indices.long()
        ].reshape((batch, 1, height, width))

        channel_select_filtered = torch.stack(
            [channel_select_filtered_positive, channel_select_filtered_negative]
        )

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        low_threshold = min(threshold1, threshold2)
        high_threshold = max(threshold1, threshold2)
        thresholded = thin_edges.clone()
        lower = thin_edges > low_threshold
        higher = thin_edges > high_threshold
        thresholded = lower * 0.5 + higher * 0.5

        weak = (thin_edges == 0.5) * 1
        weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
        thin_edges = higher * 1 + weak_is_high * 1

        return thresholded
