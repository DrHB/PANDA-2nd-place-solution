'''
    This file defines DataSet Class for high resolution trainning,cropped tiles have 3 tiling modes. During trainning,
    one random mode will be chosen and tiles will be sampled by their attention. During validation, best mode will be
    used and tiles with top attention will be chosen.
'''
import cv2
import os
import torch
import numpy as np
import random

class PANDA_Dataset_Attention(object):
    def __init__(self,
                 df,
                 image_dir,
                 label_dict,
                 valid=None,
                 # Patch parameter
                 num_patches=12,
                 crop_func=None,
                 # Augmentation & Normalization
                 augmentation_patch=None,
                 deterministic=True,
                 sqrt_wt=False,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 ):
        self.image_ids=df['image_id'].tolist()
        self.labels=df['isup_grade_x'].tolist()
        self.provider=df['data_provider'].tolist()
        self.pesudo=df['prediction_reg'].tolist()
        self.pesudo_prob=np.array(df[['prob_0','prob_1','prob_2','prob_3','prob_4','prob_5']])
        self.image_dir=image_dir
        self.label_dict=label_dict
        self.fold=df['fold'].tolist()
        self.num_tiles=num_patches
        self.sqrt_wt=sqrt_wt
        self.crop_func=crop_func
        self.augmentation_patch=augmentation_patch
        self.mean=np.array(mean,dtype=np.float32)
        self.std=np.array(std,dtype=np.float)
        self.deterministic=deterministic
        self.valid=valid
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]
        pesudo_label=self.pesudo[idx]
        pesudo_prob=self.pesudo_prob[idx]
        provider=1 if self.provider[idx]=='radboud' else 0

        if self.valid:
            mode=self.valid[img_id]
            data = self.label_dict[mode][img_id]
            file_dir=self.image_dir+"/mode{}_fold{}/".format(mode,self.fold[idx])
        elif len(self.label_dict)>1:
            mode=random.randint(0,len(self.label_dict)-1)
            data = self.label_dict[mode][img_id]
            file_dir=self.image_dir+"/mode{}_fold{}/".format(mode,self.fold[idx])
        else:
            mode=0
            data = self.label_dict[mode][img_id]
            file_dir=self.image_dir+"/mode{}_fold{}/".format(mode,self.fold[idx])


        attention=data['attention']
        attention=torch.softmax(torch.tensor(attention),dim=0).numpy()
        patch_labels=np.array(data['patch_labels']).astype(np.float32)
        ignore_mask=np.array(data['ignore_mask']).astype(np.float32)
        N=min(len(attention),self.num_tiles)

        images=[]
        if self.deterministic:
            idxs=np.arange(N)
        else:
            if self.sqrt_wt:
                sample_weight=np.sqrt(attention)
                sample_weight/=sample_weight.sum()
            else:
                sample_weight=attention
            idxs=np.random.choice(np.arange(len(attention)),p=sample_weight,replace=False,size=N)
        for idx in idxs:
            tmp=cv2.imread(os.path.join(file_dir,"{}_{}.jpg".format(img_id,idx)))
            images.append(cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB))
        patch_labels=patch_labels[idxs]
        ignore_mask=ignore_mask[idxs]

        if self.augmentation_patch is not None:
            images=[self.augmentation_patch(image=x)['image'] for x in images]
        if self.crop_func is not None:
            images=[self.crop_func(image=x)['image'] for x in images]
        images=np.stack(images,axis=0)
        if len(images)<self.num_tiles:
            images=np.pad(images,[[0,self.num_tiles-len(images)],[0,0],[0,0],[0,0]],constant_values=255)
            patch_labels=np.pad(patch_labels,[[0,self.num_tiles-len(patch_labels)]],constant_values=0)
            ignore_mask=np.pad(ignore_mask,[[0,self.num_tiles-len(ignore_mask)]],constant_values=0)

        images=1.0-images.astype(np.float32)/255
        images=(images-self.mean)/self.std
        return torch.tensor(images,dtype=torch.float32).permute(0,3,1,2),label,patch_labels,ignore_mask,pesudo_label,pesudo_prob,provider

    def __len__(self):
        return len(self.image_ids)