import os
from pathlib2 import Path
import skimage.io
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
from functools import partial
import json


def tile(img_name, sz=224, N=100):
    img = skimage.io.MultiImage(img_name)[-2]
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
   
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

    
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))
    img = img[idxs][:N]

    if len(img)<N:
        n = N-len(img)
        img = np.concatenate([img, np.full((n,sz,sz,3),255,dtype=np.uint8)],0)

    for i in range(len(img)):
        result.append({'img':img[i],  'idx':i})
    return result



def save_imgs(image_name, folder_path, out_dir):
    image_nm = f'{str(folder_path)}/{image_name}.tiff' 
    try:
        tiles = tile(image_nm)
        for t in tiles:
            img,idx = t['img'], t['idx']
            cv2.imwrite(f'{str(out_dir)}/{image_name}_{idx}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
         
    except: 
         pass
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers", 
        default=2, 
        type=int, 
        help="Number of cpu to use to process data"
    )
    args = parser.parse_args()

    
 
    with open('SETTINGS.json', 'r') as fp: data = json.load(fp)

    DATA_FOLDER = Path(data['RAW_DATA_DIR'])
    CSV_NAME    = Path(data['TRAIN_DATA_CLEAN_PATH'])
    COL_NAME    = 'image_id'
    OUT_DIR     = Path(data['CLEAN_DATA_DIR'])
    NUM_WORKERS = args.num_workers
    
    file_names = pd.read_csv(CSV_NAME)[COL_NAME].to_list()

    #removing bad images
    try:
        file_names.remove('0da0915a236f2fc98b299d6fdefe7b8b')
        file_names.remove('3790f55cad63053e956fb73027179707')
    except:
        pass

    #makign directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    #saving images
    save_img = partial(save_imgs, folder_path=DATA_FOLDER, out_dir=OUT_DIR)
    Parallel(n_jobs=NUM_WORKERS)(delayed(save_img)(i) for i in tqdm(file_names))

    
if __name__=='__main__':
    main()