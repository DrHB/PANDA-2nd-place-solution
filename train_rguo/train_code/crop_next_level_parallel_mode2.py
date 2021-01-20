import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
import cv2
from models.model_warper import PANDA_Model_Attention_Concat_MultiTask_Headv2
from dataset.patch_extraction_v2 import PANDAPatchExtraction
from PIL import Image
import matplotlib.pyplot as plt
import json
torch.backends.cudnn.benchmark = False
import openslide
from joblib import Parallel, delayed

def crop_upper_level_single(img_id,i,coord,save_dir,sz=192, scale=0.5):
    image = openslide.OpenSlide(SETTINGS['RAW_DATA_DIR']+"/train_images/{}.tiff".format(img_id))

    x = coord[1] * 4
    y = coord[0] * 4  # coordinate in upper level
    x = max(0,x)
    y = max(0,y)
    region_sz = int(sz * 4)  # new size
    patch = image.read_region((x, y), 0, (region_sz, region_sz))
    patch = np.asarray(patch.convert("RGB"))
    if scale != 1:
        patch = cv2.resize(patch, dsize=(int(scale * patch.shape[1]), int(scale * patch.shape[0])))
    cv2.imwrite(save_dir + "{}_{}.jpg".format(img_id, i), patch[:, :, ::-1], (cv2.IMWRITE_JPEG_QUALITY, 95))
    #print(save_dir + "{}_{}.jpg".format(img_id, i),"saved")

def get_idx(attention,max_patch=64):
    idx=np.argsort(attention)[::-1]
    N=min(max_patch,len(idx))
    idx=idx[:N]
    return idx

def safe_run(model,images,max_bs=128):
    num_patch=images.shape[1]
    split_dim=[max_bs]*(num_patch//max_bs)
    if num_patch%max_bs>0:
        split_dim+=[num_patch%max_bs]
    attention=[]
    for split_img in torch.split(images,split_dim,dim=1):
        with torch.no_grad():
            split_img=split_img.cuda()
            output,_,_,A=model(split_img)
            attention.append(A.cpu())
    return torch.cat(attention,dim=1)

def main(fold):
    print("Processing fold{}".format(fold))
    model_path=SETTINGS["LEVEL1_WEIGHTS_RGUO"]+"se_resnext50_32x4d_fold{}_bestOptimQWK.pth".format(fold,fold)

    tiff_dir=SETTINGS['RAW_DATA_DIR']+"train_images/"
    mask_dir=SETTINGS['RAW_DATA_DIR']+"train_label_masks/"

    df=pd.read_csv(SETTINGS["CSV_PATH"])
    suspicious = pd.read_csv(SETTINGS["SUSPICIOUS_PATH"]).image_id
    df = df[~df.image_id.isin(suspicious)]
    df = df[df.fold==fold]

    model=PANDA_Model_Attention_Concat_MultiTask_Headv2(arch='se_resnext50_32x4d',
                                                        dropout=0.4,
                                                        num_classes=6,
                                                        checkpoint=False,
                                                        scale_op=False,
                                                        )

    model.cuda()
    ckpt=torch.load(model_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    dataset=PANDAPatchExtraction(df,
                                 tiff_dir,
                                 mask_dir,
                                 # Patch parameter
                                 patch_size=192,
                                 # Augmentation & Normalization
                                 mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225),
                                 )

    dataloader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2)
    mode=2

    save_dir=SETTINGS["LEVEL0_JSON_DIR"]+"mode{}/".format(mode)
    save_dir_image=SETTINGS["LEVEL0_TILE_DIR"]+"mode{}_fold{}/".format(mode,fold)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_image,exist_ok=True)

    result_metadata=dict()
    with Parallel(n_jobs=12) as parallel:
        for images,coords,img_id,patch_labels,ignore_mask in tqdm(dataloader):
            img_id=img_id[0]
            patch_labels=patch_labels.numpy().reshape(-1).astype(np.uint8)
            ignore_mask=ignore_mask.numpy().reshape(-1).astype(np.uint8)
            with torch.no_grad():
                A=safe_run(model,images).view(-1).numpy()
                coords=coords.numpy().reshape(-1,2)

                idx = get_idx(A,max_patch=64)
                coords=coords[idx]
                patch_labels=patch_labels[idx].tolist()
                ignore_mask=ignore_mask[idx].tolist()
                attention=A[idx]
                attention=list(attention.astype(np.double))
                coords=[x.astype(np.int32).tolist() for x in coords]

                meta={
                    'coordinate':coords,
                    'size':192,
                    'attention':attention,
                    'patch_labels':patch_labels,
                    'ignore_mask':ignore_mask,
                }
                result_metadata[img_id]=meta
                parallel(delayed(crop_upper_level_single)(img_id,j,coord,save_dir_image) for j,coord in enumerate(coords))

    #print(result_metadata)
    with open(save_dir+"Attention_fold{}.json".format(fold),"w+") as f:
        json.dump(result_metadata,f)

import json
with open("./SETTINGS.json") as f:
    SETTINGS=json.load(f)


if __name__=="__main__":
    import sys
    fold=int(sys.argv[1])
    main(fold)