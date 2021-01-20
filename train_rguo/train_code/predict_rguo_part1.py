import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import math
import cv2
import skimage.io


def tile(img, sz=128, N=12):
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

    if len(img) < N:
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]
    return 1 - img.astype(np.float32) / 255


def crop_white(image: np.ndarray, value: int = 255) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class DataSet_Test(object):
    def __init__(self,
                 df,
                 img_dir,
                 size,
                 num_patch,
                 grid_offset=(0, 1 / 3, 2 / 3),
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 ):
        self.image_ids = df.image_id.tolist()
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        self.size = size
        self.num_patch = num_patch
        self.grid_offset = grid_offset

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image = skimage.io.MultiImage(os.path.join(self.img_dir, "{}.tiff".format(img_id)))[1]
        # image = openslide.OpenSlide(os.path.join(self.img_dir, "{}.tiff".format(img_id)))
        # size = image.level_dimensions[1]
        # image = np.array(image.read_region((0, 0), 1, size))[:, :, :3]
        # image=cv2.imread(os.path.join(self.img_dir, "{}.jpg".format(img_id)))[:,:,::-1]
        image = crop_white(image)
        _, encoded_img = cv2.imencode(".jpg", image, (int(cv2.IMWRITE_JPEG_QUALITY), 100))
        image = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

        # shape = image.shape
        # image = cv2.resize(image, dsize=(shape[1] // 2, shape[0] // 2))

        shape = image.shape

        pad0, pad1 = (self.size - shape[0] % self.size) % self.size, (self.size - shape[1] % self.size) % self.size

        pad_up = pad0 // 2
        pad_left = pad1 // 2

        all_patches = []
        for offset in self.grid_offset:
            pad_up_tmp = pad_up + int(offset * self.size)
            pad_left_tmp = pad_left + int(offset * self.size)
            tmp_img = np.pad(image, [[pad_up_tmp, self.size + pad0 - pad_up_tmp],
                                     [pad_left_tmp, self.size + pad1 - pad_left_tmp], [0, 0]], constant_values=255)
            patch = tile(tmp_img, sz=self.size, N=self.num_patch)
            all_patches.append(patch)

        all_patches = np.stack(all_patches, axis=0)  # [ntta,Npatch,sz,sz,3]
        all_patches = (all_patches - self.mean) / self.std

        return torch.tensor(all_patches, dtype=torch.float32).permute(0, 1, 4, 2, 3), img_id

    def __len__(self):
        return len(self.image_ids)


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
    def __init__(self, arch='se_resnext50_32x4d', dropout=0.25, num_classes=6, scale_op=True):
        super().__init__()

        self.base_model = se_resnext50_32x4d(pretrained=None)
        back_feature = self.base_model.last_linear.in_features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.scale_op = scale_op

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

import json
with open("./SETTINGS.json") as f:
    SETTINGS=json.load(f)

if __name__=="__main__":
    debug=False
    model_path = [SETTINGS["PRED_WEIGHTS_RGUO"]+"se_resnext50_32x4d_fold0_bestOptimQWK.pth",
                  SETTINGS["PRED_WEIGHTS_RGUO"]+"se_resnext50_32x4d_fold1_bestOptimQWK.pth",
                  SETTINGS["PRED_WEIGHTS_RGUO"]+"se_resnext50_32x4d_fold2_bestOptimQWK.pth",
                  SETTINGS["PRED_WEIGHTS_RGUO"]+"se_resnext50_32x4d_fold3_bestOptimQWK.pth",
                  SETTINGS["PRED_WEIGHTS_RGUO"]+"se_resnext50_32x4d_fold4_bestOptimQWK.pth"]
    if debug:
        img_dir = SETTINGS["RAW_DATA_DIR"]+"train_images/"
        df = pd.read_csv(SETTINGS["RAW_DATA_DIR"]+"train.csv")[:100]
    else:
        df = pd.read_csv(SETTINGS["RAW_DATA_DIR"]+"test.csv")
        img_dir = SETTINGS["RAW_DATA_DIR"]+"test_images/"
    modellist = []
    for path in model_path:
        model = PANDA_Model_Attention_Concat_MultiTask_Headv2(arch='se_resnext50_32x4d', dropout=0.4, num_classes=6,
                                                              scale_op=True)
        model.cuda()
        print("Loading", path)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        modellist.append(model)
    print(len(modellist), "models Loaded")

    grid_offset = [0, 1 / 2]
    num_offset = len(grid_offset)
    num_patch = 48
    patch_size = 192

    dataset = DataSet_Test(df, img_dir, size=patch_size, num_patch=num_patch, grid_offset=grid_offset)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    def get_expectation(cls_output, label):
        prob = torch.softmax(cls_output.cpu(), dim=2)
        prob = torch.mean(prob, dim=1)
        return (label * prob).sum(dim=1)

    label = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32).view(1, -1)
    reg_weight = 1.
    if os.path.exists(img_dir):
        prediction = []
        name = []
        for images, img_ids in tqdm(dataloader):
            images = images.cuda()
            images = images.view(-1, num_patch, 3, patch_size, patch_size)

            reg_pred = 0.
            cls_pred = 0.
            with torch.no_grad():
                for model in modellist:
                    reg_output, cls_output, patch_pred, A = model(images)
                    cls_output = cls_output.view(-1, num_offset, 6)
                    reg_output = reg_output.view(-1, num_offset)
                    reg_pred += reg_output.detach().cpu().numpy().mean(axis=1)
                    cls_pred += get_expectation(cls_output, label).numpy()
            reg_pred /= len(modellist)
            cls_pred /= len(modellist)

            pred_batch = reg_weight * reg_pred + (1 - reg_weight) * cls_pred

            prediction.append(pred_batch)
            name.extend(img_ids)
        prediction = np.concatenate(prediction).reshape(-1)

        result = pd.DataFrame({
            'image_id': name,
            'isup_grade': prediction
        })
        result.to_csv(SETTINGS["PREDICTION_DIR"]+"submission_medianse50.csv",index=False)
    else:
        df = pd.read_csv(SETTINGS["RAW_DATA_DIR"]+"sample_submission.csv")
        df.to_csv(SETTINGS["PREDICTION_DIR"]+"submission_medianse50.csv",index=False)