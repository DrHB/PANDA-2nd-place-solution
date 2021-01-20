from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2
import skimage.io
import openslide
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import types
import re

from collections import OrderedDict
import math

from torch.utils import model_zoo
from torch.utils.checkpoint import checkpoint,checkpoint_sequential

torch.backends.cudnn.benchmark = False


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def features_ckpt(self, x):
        x.requires_grad = True
        x = checkpoint(self.layer0, x, preserve_rng_state=False)
        x = checkpoint_sequential(self.layer1, 3, x, preserve_rng_state=False)
        x = checkpoint_sequential(self.layer2, 4, x, preserve_rng_state=False)
        x = checkpoint_sequential(self.layer3, 6, x, preserve_rng_state=False)
        x = checkpoint_sequential(self.layer4, 3, x, preserve_rng_state=False)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model

"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    # random_tensor = keep_prob
    random_tensor = torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device) + keep_prob
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

url_map_advprop = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',
    'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth',
}


def load_pretrained_weights(model, model_name, load_fc=True, advprop=False):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    # AutoAugment or Advprop (different preprocessing)
    url_map_ = url_map_advprop if advprop else url_map
    state_dict = model_zoo.load_url(url_map_[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def features_ckpt(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = checkpoint(block, x, torch.tensor(drop_connect_rate).cuda(), preserve_rng_state=True)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        # load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class AttentionPool(nn.Module):
    def __init__(self, in_ch, hidden=512, dropout=True):
        super().__init__()
        self.in_ch = in_ch

        module = [nn.Linear(in_ch, hidden, bias=True),
                  nn.Tanh()
                  ]
        if dropout:
            module.append(nn.Dropout(0.25))
        module.append(nn.Linear(hidden, 1, bias=True))
        self.attention = nn.Sequential(*module)

    def forward(self, x):
        num_patch = x.size(1)
        x = x.view(-1, x.size(2))
        A = self.attention(x)
        A = A.view(-1, num_patch, 1)
        wt = F.softmax(A, dim=1)
        return (x.view(-1, num_patch, self.in_ch) * wt).sum(dim=1), A


class AdaptiveConcatPool2d_Attention(nn.Module):
    def __init__(self, in_ch, hidden=512, dropout=True):
        super().__init__()
        sz = (1, 1)
        self.ap = AttentionPool(in_ch, hidden=hidden, dropout=dropout)
        self.mp = nn.AdaptiveMaxPool2d(sz)
        self.in_ch = in_ch

    def forward(self, x):
        ap, A = self.ap(x)  # [batch,num_patch,C]
        mp = torch.max(x, dim=1)[0]
        return torch.cat([ap, mp], dim=1), A


class PANDA_Model_Attention_Concat_MultiTask_Headv2(nn.Module):
    def __init__(self, arch='se_resnext50_32x4d', dropout=0.25, num_classes=6, checkpoint=False, scale_op=True):
        super().__init__()
        self.scale_op = scale_op
        if "se" in arch:
            self.base_model = se_resnext50_32x4d(pretrained=None)
            back_feature = self.base_model.last_linear.in_features
        else:
            self.base_model = EfficientNet.from_pretrained(arch, num_classes=num_classes)
            back_feature = self.base_model._fc.in_features

        self.checkpoint = checkpoint
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attention = AdaptiveConcatPool2d_Attention(in_ch=back_feature, hidden=512, dropout=dropout > 0)

        self.label_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(back_feature, 1, bias=True)
        )

        self.reg_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * back_feature, 1, bias=True),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * back_feature, num_classes, bias=True),
        )

    def forward(self, x):
        # x [bs,n,3,h,w]
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = self.base_model.features(x)
        x = self.avg_pool(x).view(x.size(0), -1)

        patch_pred = self.label_head(x)
        x = x.view(B, N, -1)
        x, A = self.attention(x)

        reg_pred = self.reg_head(x).view(-1)
        if self.scale_op:
            reg_pred = 7. * torch.sigmoid(reg_pred) - 1.
        cls_pred = self.cls_head(x)
        return reg_pred, cls_pred, patch_pred, A


def crop_white(image, value=255):
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image, np.array([0, 0], dtype=np.int)
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1], np.array([ys.min(), xs.min()], dtype=np.int)


def crop_patches(img, bg_threshold, sz=192):
    W = img.shape[1]

    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz * sz, 3)

    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
    background_ratio = (sat < 20).astype(np.float32).reshape(img.shape[0], -1).sum(1) / (sz * sz)

    fg_idx = np.where(background_ratio < bg_threshold)[0]

    img = img[fg_idx]

    coord = np.stack([fg_idx // (W // sz), fg_idx % (W // sz)], axis=1) * sz
    return img.reshape(-1, sz, sz, 3), coord, background_ratio[fg_idx]


def crop_upper_level(img_id, coords, sz, scale=0.5):
    image = openslide.OpenSlide(os.path.join(tiff_dir, "{}.tiff".format(img_id)))
    patches = []
    for coord in coords:
        x = coord[1] * 4
        y = coord[0] * 4  # coordinate in upper level
        x = max(0, x)
        y = max(0, y)
        region_sz = int(sz * 4)  # new size
        patch = image.read_region((x, y), 0, (region_sz, region_sz))
        patch = np.asarray(patch.convert("RGB"))
        if scale != 1:
            patch = cv2.resize(patch, dsize=(int(scale * patch.shape[1]), int(scale * patch.shape[0])))
        patches.append(patch)
    return patches


def get_next_level_patches(img_id, attention, coords, sz=192, scale=0.5, max_patch=64):
    idx = np.argsort(attention)[::-1]
    N = min(max_patch, len(idx))
    idx = idx[:N]
    coords = coords[idx]
    next_level_patches = crop_upper_level(img_id, coords, sz, scale)
    return next_level_patches, attention[idx], idx


def prepare_next_level_input(next_level_patches, patch_num=32, crop_func=None,
                             mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225), ):
    next_level_patches = np.stack(next_level_patches, axis=0)

    patches = []
    for patch in next_level_patches:
        _, encoded_img = cv2.imencode(".jpg", patch, (int(cv2.IMWRITE_JPEG_QUALITY), 95))
        patch = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
        patches.append(patch)

    if crop_func is not None:
        patches = [crop_func(image=x)['image'] for x in patches]
    patches = np.stack(patches, axis=0)
    if len(patches) < patch_num:
        patches = np.pad(patches, [[0, patch_num - len(patches)], [0, 0], [0, 0], [0, 0]], constant_values=255)

    patches = 1.0 - patches.astype(np.float32) / 255
    patches = (patches - mean) / std
    return torch.tensor(patches, dtype=torch.float32, device='cuda').permute(0, 3, 1, 2).unsqueeze(0)


class PANDAPatchExtraction_Test(object):
    def __init__(self,
                 df,
                 tiff_dir,

                 # Patch parameter
                 patch_size=192,
                 bg_threshold=0.8,
                 trail_offset=[0, 1 / 2],

                 # Augmentation & Normalization
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),

                 ):
        self.image_ids = df['image_id'].tolist()

        self.tiff_dir = tiff_dir
        self.patch_size = patch_size
        self.bg_threshold = bg_threshold
        self.trail_offset = trail_offset

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image = skimage.io.MultiImage(os.path.join(self.tiff_dir, "{}.tiff".format(img_id)))[1]

        image, offset = crop_white(image)
        _, encoded_img = cv2.imencode(".jpg", image, (int(cv2.IMWRITE_JPEG_QUALITY), 100))
        image = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

        shape = image.shape

        pad0 = (self.patch_size - shape[0] % self.patch_size) % self.patch_size
        pad1 = (self.patch_size - shape[1] % self.patch_size) % self.patch_size

        pad_up = pad0 // 2
        pad_left = pad1 // 2

        best_mean_bg_ratio = 1000
        best_patch = None
        best_coord = None
        best_pad_offset = None

        for trail_offset in self.trail_offset:
            trail_pad_up = pad_up + int(self.patch_size * trail_offset)
            trail_pad_left = pad_left + int(self.patch_size * trail_offset)
            image_tmp = np.pad(image, [[trail_pad_up, pad0 + self.patch_size - trail_pad_up],
                                       [trail_pad_left, pad1 + self.patch_size - trail_pad_left], [0, 0]],
                               constant_values=255)

            patches, coords, bg_ratio = crop_patches(image_tmp, self.bg_threshold, sz=self.patch_size)
            if np.mean(bg_ratio) < best_mean_bg_ratio:
                best_mean_bg_ratio = np.mean(bg_ratio)
                best_patch = patches
                best_coord = coords
                best_pad_offset = (trail_pad_up, trail_pad_left)

        # print("best",best_mean_bg_ratio)
        offset[0] -= best_pad_offset[0]
        offset[1] -= best_pad_offset[1]

        if len(best_patch) == 0:
            best_patch = 255 * np.ones((1, self.patch_size, self.patch_size, 3))
            best_coord = np.zeros((1, 2))

        best_coord = best_coord + offset.reshape(1, 2)

        best_patch = 1.0 - best_patch.astype(np.float32) / 255
        best_patch = (best_patch - self.mean) / self.std
        return torch.tensor(best_patch, dtype=torch.float32).permute(0, 3, 1, 2), best_coord, img_id

    def __len__(self):
        return len(self.image_ids)


def safe_run(model, images, max_bs=128):
    num_patch = images.shape[1]
    split_dim = [max_bs] * (num_patch // max_bs)
    if num_patch % max_bs > 0:
        split_dim += [num_patch % max_bs]
    attention = []
    for split_img in torch.split(images, split_dim, dim=1):
        with torch.no_grad():
            split_img = split_img.cuda()
            output, _, _, A = model(split_img)
            attention.append(A.cpu())
    return torch.cat(attention, dim=1)

import json
with open("./SETTINGS.json") as f:
    SETTINGS=json.load(f)

debug=False
if debug:
    tiff_dir = SETTINGS["RAW_DATA_DIR"]+"train_images/"
    df = pd.read_csv(SETTINGS["RAW_DATA_DIR"]+"train.csv")[:100]
else:
    df = pd.read_csv(SETTINGS["RAW_DATA_DIR"]+"test.csv")
    tiff_dir = SETTINGS["RAW_DATA_DIR"]+"test_images/"



if __name__=="__main__":
    model_path_last_level=[]
    model_path_this_level=[]
    for f in range(5):
        model_path_last_level.append(SETTINGS["LEVEL1_WEIGHTS_RGUO"]+"se_resnext50_32x4d_fold{}_bestOptimQWK.pth".format(f,f))
    print("Last level weight")
    print(model_path_last_level)
    #for f in range(5):
    #    model_path_this_level.append("../input/panda-highresolution-ef/efficientnet-b0_fold{}_bestOptimQWK.pth".format(f,f))
    model_path_this_level.append(SETTINGS["PRED_WEIGHTS_RGUO"]+"efficientnet-b0_fold0_bestOptimQWK.pth")
    model_path_this_level.append(SETTINGS["PRED_WEIGHTS_RGUO"]+"efficientnet-b0_fold1_bestOptimQWK.pth")
    model_path_this_level.append(SETTINGS["PRED_WEIGHTS_RGUO"]+"efficientnet-b0_fold2_bestOptimQWK.pth")
    model_path_this_level.append(SETTINGS["PRED_WEIGHTS_RGUO"]+"efficientnet-b0_fold4_bestOptimQWK.pth")
    print("This level weight")
    print(model_path_this_level)

    model_last_level = []
    model_this_level = []
    for p in model_path_last_level:
        print("Loading last level", p)
        model = PANDA_Model_Attention_Concat_MultiTask_Headv2(arch='se_resnext50_32x4d',
                                                              dropout=0.25,
                                                              num_classes=6,
                                                              scale_op=False,
                                                              )
        model.cuda()
        ckpt = torch.load(p)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        model_last_level.append(model)

    for p in model_path_this_level:
        print("Loading this level", p)
        model = PANDA_Model_Attention_Concat_MultiTask_Headv2(arch='efficientnet-b0',
                                                              dropout=0.4,
                                                              num_classes=6,
                                                              scale_op=True,
                                                              )
        model.cuda()
        ckpt = torch.load(p)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        model_this_level.append(model)
    print(len(model_last_level), "Median resolution models")
    print(len(model_this_level), "High resolution models")

    dataset = PANDAPatchExtraction_Test(df,
                                        tiff_dir,
                                        patch_size=192,
                                        bg_threshold=0.95,
                                        trail_offset=[0, 1 / 2],
                                        )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    coef = [0.5, 1.5, 2.5, 3.5, 4.5]


    def predict(X, coef):
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3, 4, 5])


    if os.path.exists(tiff_dir):
        prediction = []
        name = []
        for images, coords, img_ids in tqdm(dataloader):
            with torch.no_grad():
                # run median level to collect image
                A = []
                for model in model_last_level:
                    att = safe_run(model, images).view(-1).numpy()
                    A.append(att)
                A = np.stack(A, axis=0).mean(axis=0)

                coords = coords.numpy().reshape(-1, 2)

                next_level_patches, attention, idx = get_next_level_patches(img_ids[0], A, coords, sz=192, scale=0.5,
                                                                            max_patch=36)

                # run again level1 get prediction
                # idx=np.sort(idx)
                # this_level_input=images[:,idx].cuda()
                # reg_output_last, cls_output_last, patch_label_last,A_last = model_this_level(next_level_patches)

                # collect next level and run
                next_level_patches = prepare_next_level_input(next_level_patches, patch_num=36, crop_func=None)

                reg_pred = 0.
                cls_pred = 0.
                next_level_patches = torch.cat(
                    [next_level_patches, next_level_patches.flip((3, 4))], dim=0)
                for model in model_this_level:
                    reg_output, cls_output, patch_pred, A = model(next_level_patches)
                    cls_output = cls_output.view(-1, 6).mean(dim=0)
                    reg_output = reg_output.view(-1).mean(dim=0)
                    reg_pred += reg_output.cpu().numpy()
                    cls_pred += cls_output.cpu().numpy()
                reg_pred /= len(model_this_level)
                cls_pred /= len(model_this_level)
                prediction.append(reg_pred)
                # prediction.append(reg_output.detach().cpu().numpy()+reg_output_last.detach().cpu().numpy())
                name.extend(img_ids)

        prediction = np.array(prediction).reshape(-1)
        # prediction=predict(prediction,coef)
        result = pd.DataFrame({
            'image_id': name,
            'isup_grade': prediction
        })
        result.to_csv(SETTINGS["PREDICTION_DIR"]+"submission_highefb0.csv",index=False)
    else:
        df = pd.read_csv(SETTINGS["RAW_DATA_DIR"]+"sample_submission.csv")
        df.to_csv(SETTINGS["PREDICTION_DIR"]+"submission_highefb0.csv",index=False)