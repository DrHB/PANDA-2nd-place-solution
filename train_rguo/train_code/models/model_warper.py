import torch
import torch.nn as nn
from models.base_models.efficientnet import EfficientNet
import models.base_models
from .pooling import AdaptiveConcatPool2d_Attention

class PANDA_Model_Attention_Concat_MultiTask_Headv2(nn.Module):
    '''
        Pytorch Model used in PANDA Challenge
    '''
    def __init__(self,arch='se_resnext50_32x4d',dropout=0.25,num_classes=6,checkpoint=False,scale_op=True,gated=False):
        super().__init__()
        self.scale_op=scale_op
        self.timm=False
        if "efficientnet" in arch:
            self.base_model = EfficientNet.from_pretrained(arch, num_classes=num_classes)
            back_feature = self.base_model._fc.in_features
        else:
            self.base_model= models.base_models.__dict__[arch](pretrained="imagenet")
            try:
                back_feature=self.base_model.last_linear.in_features
            except:
                back_feature=self.base_model.fc.in_features
        self.checkpoint=checkpoint
        self.avg_pool=nn.AdaptiveAvgPool2d(1)

        self.attention=AdaptiveConcatPool2d_Attention(in_ch=back_feature,hidden=512,dropout=dropout,gated=gated)
        self.label_head=nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(back_feature,1,bias=True)
        )

        self.reg_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2*back_feature,1,bias=True),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2*back_feature,num_classes,bias=True),
        )


    def forward(self,x):
        # x [bs,n,3,h,w]
        B,N,C,H,W=x.shape
        x=x.view(-1,C,H,W)
        if self.checkpoint:
            x=self.base_model.features_ckpt(x)
        elif self.timm:
            x=self.base_model.forward_features(x)
        else:
            x=self.base_model.features(x)
        x=self.avg_pool(x).view(x.size(0),-1)

        patch_pred=self.label_head(x)
        x=x.view(B,N,-1)
        x,A=self.attention(x)

        reg_pred=self.reg_head(x).view(-1)
        if self.scale_op:
            reg_pred=7.*torch.sigmoid(reg_pred)-1.
        cls_pred=self.cls_head(x)
        return reg_pred,cls_pred,patch_pred,A

