'''
    This file defines DataSet Class for trainning in median resolution.
'''

import cv2
import os
import torch
import numpy as np
from .utilsv2 import *

class PANDA_Dataset_MultiTask_smooth_v2:
    def __init__(self,df,
                 image_dir,
                 mask_dir,

                 #Patch parameter
                 num_patches=12,
                 patch_size=128,
                 deterministic=True,
                 #Augmentation & Normalization
                 augmentation=None,
                 augmentation_patch=None,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 resample=False,
                 resample_threshold=0.05,
                 glue4=False
):
        self.image_ids=df['image_id'].tolist()
        self.labels=df['isup_grade_x'].tolist()
        self.provider=df['data_provider'].tolist()
        #self.pesudo=((df['cls_exp']+df['prediction_reg'])/2).tolist()
        self.pesudo=df['prediction_reg'].tolist()
        self.pesudo_prob=np.array(df[['prob_0','prob_1','prob_2','prob_3','prob_4','prob_5']])
        self.image_dir=image_dir
        self.mask_dir=mask_dir

        self.num_tiles=num_patches
        self.tile_size=patch_size
        self.augmentation=augmentation
        self.augmentation_patch=augmentation_patch
        self.mean=np.array(mean,dtype=np.float32)
        self.std=np.array(std,dtype=np.float)
        self.deterministic=deterministic
        self.resample=resample
        self.resample_threshold=resample_threshold
        self.glue4=glue4
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]
        image = cv2.imread(os.path.join(self.image_dir, "{}.jpg".format(img_id)))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, "{}.png".format(img_id)),cv2.IMREAD_GRAYSCALE)
        pesudo_label=self.pesudo[idx]
        pesudo_prob=self.pesudo_prob[idx]
        if self.augmentation is not None:
            augmented=self.augmentation(image=image,mask=mask)
            image=augmented['image']
            mask=augmented['mask']
        if self.provider[idx]=="radboud":
            mask=(mask==1).astype(np.uint8)+(mask==2).astype(np.uint8)+(mask>2).astype(np.uint8)*2
        provider=1 if self.provider[idx]=='radboud' else 0
        shape = image.shape
        pad0, pad1 = (self.tile_size - shape[0] % self.tile_size) % self.tile_size, (self.tile_size - shape[1] % self.tile_size) % self.tile_size
        if self.deterministic:
            pad_up = pad0 // 2
            pad_left = pad1 // 2
        else:
            pad_up = random.randint(0, pad0)
            pad_left = random.randint(0, pad1)
        image = np.pad(image, [[pad_up, pad0 - pad_up], [pad_left, pad1 - pad_left], [0, 0]], constant_values=255)
        mask = np.pad(mask, [[pad_up, pad0 - pad_up], [pad_left, pad1 - pad_left]], constant_values=0)

        tiles,patch_labels,ignore_mask=tile_with_label(image,mask,sz=self.tile_size,N=self.num_tiles,sat_based=False,hed_based=False,glue4=self.glue4)

        if self.resample:
            empty_idx=detect_empty(tiles,sat_threshold=20,fg_fraction=self.resample_threshold)
            if len(empty_idx)>0:
                #print("Image id{}: replace {} patches".format(img_id,len(empty_idx)))
                #cv2.imshow("raw_patches",cv2.resize(tiles.copy().reshape(6,8,192,192,3).transpose(0,2,1,3,4).reshape(6*192,8*192,3)[:,:,::-1],(800,600)))


                #image = np.pad(image, [[self.tile_size//2,self.tile_size-self.tile_size//2], [self.tile_size//2,self.tile_size-self.tile_size//2], [0, 0]], constant_values=255).transpose(1,0,2)
                #mask = np.pad(mask, [[self.tile_size//2, self.tile_size-self.tile_size//2], [self.tile_size//2,self.tile_size-self.tile_size//2]], constant_values=0).transpose(1,0)
                image=image[self.tile_size//2:-(self.tile_size-self.tile_size//2),self.tile_size-self.tile_size//2:-(self.tile_size-self.tile_size//2)].transpose(1,0,2)
                mask=mask[self.tile_size//2:-(self.tile_size-self.tile_size//2),self.tile_size-self.tile_size//2:-(self.tile_size-self.tile_size//2)].transpose(1,0)

                tiles_second, patch_labels_second, ignore_mask_second = tile_with_label(image, mask, sz=self.tile_size, N=self.num_tiles,sat_based=True)

                tiles[empty_idx]=tiles_second[:len(empty_idx)]
                patch_labels[empty_idx]=patch_labels[:len(empty_idx)]
                ignore_mask[empty_idx]=ignore_mask_second[:len(empty_idx)]
                #cv2.imshow("resampled_patches",cv2.resize(tiles.reshape(6,8,192,192,3).transpose(0,2,1,3,4).reshape(6*192,8*192,3)[:,:,::-1],(800,600)))
                #cv2.waitKey()

        if self.augmentation_patch is not None:
            tiles=np.stack([self.augmentation_patch(image=x)['image'] for x in tiles],axis=0)

        tiles=1.0-tiles.astype(np.float32)/255
        tiles=(tiles-self.mean)/self.std
        return torch.tensor(tiles,dtype=torch.float32).permute(0,3,1,2),label,patch_labels,ignore_mask,pesudo_label,pesudo_prob,provider

    def __len__(self):
        return len(self.image_ids)
