'''
    Tiling function
'''

import numpy as np
import cv2
import os
import random
import torch
from numba import jit
from skimage.color import rgb2hed
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [rmin, rmax, cmin, cmax]

def tile_with_label(img,mask,sz=128,N=12,sat_based=False,hed_based=False,pad=True,glue4=False):
    #tmp = img.copy().astype(np.uint8)

    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz)
    mask = mask.transpose(0,2,1,3).reshape(-1,sz*sz)
    if pad:
        if len(img) < N:
            img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
            mask = np.pad(mask,[[0,N-len(mask)],[0,0]],constant_values=0)
    #print(img.reshape(img.shape[0],-1).sum(-1))
    if sat_based:
        sat=cv2.cvtColor(img.reshape(-1,sz*sz,3),cv2.COLOR_RGB2HSV)[:,:,1]
        idxs = np.argsort(sat.reshape(sat.shape[0],-1).sum(-1))[-N:]
        idxs=idxs[::-1]
    elif hed_based:
        hed=rgb2hed(img.reshape(-1,sz*sz,3))
        idxs = np.argsort(hed.sum(axis=(1,2)))[-N:]
        idxs=idxs[::-1]
    else:
        idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

    img = img[idxs]
    mask=mask[idxs]
    if glue4:
        #img [48
        img=img.reshape(N//4,2,2,sz,sz,3).transpose(0,1,3,2,4,5).reshape(N//4,2*sz,2*sz,3)
        mask=mask.reshape(N//4,4*sz*sz)

    patch_labels=((mask==2).astype(np.int).sum(axis=1)>100).astype(np.float32).reshape(-1)
    ignore_mask=(mask.sum(axis=1)>0).astype(np.float32).reshape(-1)
    '''
    for i in range(len(img)):
        cv2.imshow("img",img[i][:,:,::-1])
        cv2.imshow("msk",mask[i].reshape(sz,sz)*127)
        print(patch_labels[i],ignore_mask[i])
        cv2.waitKey()
   
    for idx in idxs:
        i=idx//(tmp.shape[1]//sz)
        j=idx%(tmp.shape[1]//sz)
        tmp=cv2.rectangle(tmp,(sz*j,sz*i),(sz*(j+1),sz*(i+1)),(0,255,0),16)
    ratio=float(tmp.shape[0])/float(tmp.shape[1])
    if tmp.shape[0]>900:
        tmp=cv2.resize(tmp,dsize=(int(900/ratio),900))
    if tmp.shape[1]>1600:
        tmp=cv2.resize(tmp,dsize=(1600,int(1600*ratio)))
    cv2.imshow("tmp",tmp[:,:,::-1])
    cv2.waitKey()
    '''
    return img,patch_labels,ignore_mask


def detect_empty(tiles,sat_threshold=30,fg_fraction=0.25):
    sz=tiles.shape[1]
    pix_threshold=int(fg_fraction*sz*sz)
    sat=cv2.cvtColor(tiles.reshape(-1,sz*sz,3),cv2.COLOR_RGB2HSV)[:,:,1]
    empty_idx=np.where((sat>sat_threshold).sum(axis=1)<pix_threshold)[0]
    return empty_idx


def detect_nonempty(tiles,sat_threshold=30,fg_fraction=0.25):
    sz=tiles.shape[1]
    pix_threshold=int(fg_fraction*sz*sz)
    sat=cv2.cvtColor(tiles.reshape(-1,sz*sz,3),cv2.COLOR_RGB2HSV)[:,:,1]
    nonempty_idx=np.where((sat>sat_threshold).sum(axis=1)>pix_threshold)[0]
    fg_ratio=(sat>sat_threshold).sum(axis=1)/sz/sz
    return nonempty_idx,fg_ratio[nonempty_idx]