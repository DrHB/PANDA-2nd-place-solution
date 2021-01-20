'''
    DataSet used for extract high resolution tiles
'''


import cv2
import os
import torch
import numpy as np
import skimage.io
import openslide


def crop_white(image,mask, value=255):
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image,mask
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1],mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1],np.array([ys.min(), xs.min()], dtype=np.int)



def crop_patches(img,mask,bg_threshold, sz=192):
    W=img.shape[1]

    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz * sz, 3)
    mask = mask.reshape(mask.shape[0] // sz, sz, mask.shape[1] // sz, sz)
    mask = mask.transpose(0, 2, 1, 3).reshape(-1, sz * sz)

    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
    background_ratio = (sat < 20).astype(np.float32).reshape(img.shape[0], -1).sum(1) / (sz * sz)

    fg_idx = np.where(background_ratio < bg_threshold)[0]

    img = img[fg_idx]
    mask = mask[fg_idx]
    patch_labels = ((mask == 2).astype(np.int).sum(axis=1) > 100).astype(np.float32).reshape(-1)
    ignore_mask = (mask.sum(axis=1) > 0).astype(np.float32).reshape(-1)
    coord = np.stack([fg_idx // (W // sz), fg_idx % (W // sz)], axis=1)*sz
    return img.reshape(-1, sz, sz, 3), coord, background_ratio[fg_idx],patch_labels,ignore_mask


class PANDAPatchExtraction(object):
    def __init__(self,
                 df,
                 tiff_dir,
                 mask_dir,

                 scale=None,
                 # Patch parameter
                 patch_size=192,
                 bg_threshold=0.8,
                 trail_offset=[0,1/2],

                 # Augmentation & Normalization
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),

                 ):
        self.image_ids = df['image_id'].tolist()
        self.provider=df['data_provider'].tolist()
        self.tiff_dir = tiff_dir
        self.mask_dir=mask_dir
        self.patch_size = patch_size
        self.scale = scale

        self.bg_threshold = bg_threshold
        self.trail_offset=trail_offset

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image = skimage.io.MultiImage(os.path.join(self.tiff_dir, "{}.tiff".format(img_id)))[1]
        mask = skimage.io.MultiImage(os.path.join(self.mask_dir, "{}_mask.tiff".format(img_id)))[1][:,:,0]

        image, mask,offset = crop_white(image,mask)
        _, encoded_img = cv2.imencode(".jpg", image, (int(cv2.IMWRITE_JPEG_QUALITY), 100))
        image = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
        if self.provider[idx]=="radboud":
            mask=(mask==1).astype(np.uint8)+(mask==2).astype(np.uint8)+(mask>2).astype(np.uint8)*2



        shape = image.shape

        pad0 = (self.patch_size - shape[0] % self.patch_size) % self.patch_size
        pad1 = (self.patch_size - shape[1] % self.patch_size) % self.patch_size

        pad_up = pad0 // 2
        pad_left = pad1 // 2

        best_mean_bg_ratio=1000
        best_patch=None
        best_coord=None
        best_pad_offset=None
        best_patch_labels=None
        best_ignore_mask=None
        best_offset=0
        for trail_offset in self.trail_offset:
            trail_pad_up = pad_up + int(self.patch_size * trail_offset)
            trail_pad_left = pad_left + int(self.patch_size * trail_offset)
            image_tmp = np.pad(image, [[trail_pad_up, pad0+self.patch_size - trail_pad_up], [trail_pad_left, pad1+self.patch_size - trail_pad_left], [0, 0]], constant_values=255)
            mask_tmp=np.pad(mask, [[trail_pad_up, pad0+self.patch_size - trail_pad_up], [trail_pad_left, pad1+self.patch_size - trail_pad_left]], constant_values=0)
            patches, coords,bg_ratio,patch_labels,ignore_mask = crop_patches(image_tmp,mask_tmp,self.bg_threshold, sz=self.patch_size)
            if np.mean(bg_ratio)<best_mean_bg_ratio:
                best_mean_bg_ratio=np.mean(bg_ratio)
                best_patch=patches
                best_coord=coords
                best_pad_offset=(trail_pad_up,trail_pad_left)
                best_ignore_mask=ignore_mask
                best_patch_labels=patch_labels
                best_offset=trail_offset*2

        #print("best",best_mean_bg_ratio)
        offset[0] -= best_pad_offset[0]
        offset[1] -= best_pad_offset[1]

        best_coord = best_coord + offset.reshape(1, 2)

        best_patch = 1.0 - best_patch.astype(np.float32) / 255
        best_patch = (best_patch - self.mean) / self.std
        return torch.tensor(best_patch, dtype=torch.float32).permute(0, 3, 1, 2), best_coord, img_id,best_patch_labels,best_ignore_mask,int(best_offset)

    def __len__(self):
        return len(self.image_ids)


def crop_upper_level(img_id, coords, sz, scale=0.5):
    image = openslide.OpenSlide("../prostate-cancer-grade-assessment/train_images/{}.tiff".format(img_id))
    patches = []
    for coord in coords:
        x = coord[1] * 4
        y = coord[0] * 4  # coordinate in upper level
        x = max(0,x)
        y = max(0,y)
        region_sz = int(sz * 4)  # new size
        patch = image.read_region((x, y), 0, (region_sz, region_sz))
        patch = np.asarray(patch.convert("RGB"))
        if scale != 1:
            patch = cv2.resize(patch, dsize=(int(scale * patch.shape[1]), int(scale * patch.shape[0])))
        patches.append(patch)
    return patches
