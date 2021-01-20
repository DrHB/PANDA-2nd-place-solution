# coding=utf-8
import math
import json
import torch
import numpy as np
import torch.nn.functional as F
import timm
import collections
from torch import nn
from torch.nn import Sequential
from torch.nn.parameter import Parameter

from blocks import *


with open('SETTINGS.json', 'r') as fp:
	settings = json.load(fp)
	pt_weights = settings['MODEL_PT_WEIGHTS']


class AdaptiveConcatPool2d(nn.Module):
	def __init__(self, sz=1):
		super().__init__()
		self.output_size = sz or 1
		self.ap = nn.AdaptiveAvgPool2d(self.output_size)
		self.mp = nn.AdaptiveMaxPool2d(self.output_size)
		
	def forward(self, x):
		return torch.cat([self.mp(x), self.ap(x)], 1)
	

def sigmoid_scale(x, low=-1.0, high=6.0):
	return torch.sigmoid(x) * (high-low) + low


class c_Conv2d1(nn.Conv2d):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super(c_Conv2d1, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
												   dilation, groups, bias)
		self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
												   dilation, groups, bias))

	def forward(self, x):
		return self.conv(x)
		

class c_Conv2d(nn.Conv2d):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super(c_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
				 padding, dilation, groups, bias)

	def forward(self, x):
		# return super(Conv2d, self).forward(x)
		weight = self.weight
		weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
		weight = weight - weight_mean
		std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
		weight = weight / (std.expand_as(weight) + 1e-6)
		return F.conv2d(x, weight, self.bias, self.stride,
						self.padding, self.dilation, self.groups)


def c_BatchNorm2d(num_features):
	return nn.GroupNorm(num_channels=num_features, num_groups=32)


class Bottleneck(nn.Module):
	"""
	RexNeXt bottleneck type C
	"""
	expansion = 4

	def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
		""" Constructor
		Args:
			inplanes: input channel dimensionality
			planes: output channel dimensionality
			baseWidth: base width.
			cardinality: num of convolution groups.
			stride: conv stride. Replaces pooling layer.
		"""
		super(Bottleneck, self).__init__()

		D = int(math.floor(planes * (baseWidth / 64)))
		C = cardinality

		self.conv1 = c_Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn1 = c_BatchNorm2d(D*C)
		self.conv2 = c_Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
		self.bn2 = c_BatchNorm2d(D*C)
		self.conv3 = c_Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = c_BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)

		self.downsample = downsample

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

		out += residual
		out = self.relu(out)

		return out


class ResNeXt(nn.Module):
	"""
	ResNext optimized for the ImageNet dataset, as specified in
	https://arxiv.org/pdf/1611.05431.pdf
	"""
	def __init__(self, baseWidth, cardinality, layers, num_classes):
		""" Constructor
		Args:
			baseWidth: baseWidth for ResNeXt.
			cardinality: number of convolution groups.
			layers: config of layers, e.g., [3, 4, 6, 3]
			num_classes: number of classes
		"""
		super(ResNeXt, self).__init__()
		block = Bottleneck

		self.cardinality = cardinality
		self.baseWidth = baseWidth
		self.num_classes = num_classes
		self.inplanes = 64
		self.output_size = 64

		self.conv1 = c_Conv2d(3, 64, 7, 2, 3, bias=False)
		self.bn1 = c_BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], 2)
		self.layer3 = self._make_layer(block, 256, layers[2], 2)
		self.layer4 = self._make_layer(block, 512, layers[3], 2)
		self.avgpool = nn.AvgPool2d(7)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		""" Stack n bottleneck modules where n is inferred from the depth of the network.
		Args:
			block: block type used to construct ResNext
			planes: number of output channels (need to multiply by block.expansion)
			blocks: number of blocks to be built
			stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
		Returns: a Module consisting of n sequential bottlenecks.
		"""
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				c_Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				c_BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


class Resnext50_ws(nn.Module):
	def __init__(self, baseWidth=4, cardinality=32):
		super(Resnext50_ws, self).__init__()
		base_model = ResNeXt(baseWidth, cardinality, [3, 4, 6, 3], 1000)
		base_model.load_state_dict(fix_weights(pt_weights))
		self.base_model = nn.Sequential(*list(base_model.children())[:-2])
		nc = list(base_model.children())[-1].in_features
		self.conv_block = SqueezeExcite(nc)
		self.head = nn.Sequential(AdaptiveConcatPool2d(),
								  nn.Flatten(),
								  nn.Linear(2*nc, 512),
								  nn.ReLU(),
								  nn.Dropout(0.4),
								  nn.Linear(512, 1, bias=False))
	
	def forward(self, x):
		n = x.shape[1]
		x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
		x = self.base_model(x)
		x = x.view(-1, n, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4).contiguous(). \
			view(-1, x.shape[1], x.shape[2] * n, x.shape[3])
		x = x.view(x.shape[0], x.shape[1], x.shape[2] // int(np.sqrt(n)), -1)
		x = self.conv_block(x)
		x = self.head(x)

		return sigmoid_scale(x)


def fix_weights(weights):
	state_dict = torch.load(weights)
	new_state_dict = collections.OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v
	return new_state_dict

