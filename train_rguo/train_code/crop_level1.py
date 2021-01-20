import numpy as np
import cv2
from multiprocessing import Pool
import openslide
import os
import tqdm
from PIL import Image
import skimage.io
def crop_white(image,mask, value: int = 255):
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image,mask
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1],mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def crop_save(image,mask,img_id):
    image,mask=crop_white(image,mask)
    save_fn_img=os.path.join(SETTINGS["LEVEL1_IMAGE_DIR"],"{}.jpg".format(img_id))
    save_fn_mask=os.path.join(SETTINGS["LEVEL1_MASK_DIR"],"{}.png".format(img_id))
    cv2.imwrite(save_fn_img,image[:,:,::-1],(cv2.IMWRITE_JPEG_QUALITY,100))
    cv2.imwrite(save_fn_mask,mask)


def job(img_id):
    image = skimage.io.MultiImage(os.path.join(SETTINGS['RAW_DATA_DIR'],"train_images", "{}.tiff".format(img_id)))[1]
    #if os.path.exists(os.path.join("../prostate-cancer-grade-assessment/train_label_masks", "{}_mask.tiff".format(img_id))):
        #print("../prostate-cancer-grade-assessment/train_label_masks", "{}_mask.tiff".format(img_id))
    try:
        mask = skimage.io.MultiImage(os.path.join(SETTINGS['RAW_DATA_DIR'],"train_label_masks", "{}_mask.tiff".format(img_id)))[1][:,:,0]
    except:
        print("Failed to load mask {}".format(img_id))
        mask = np.zeros(image.shape[:2],dtype=np.uint8)
    crop_save(image,mask,img_id)

def main():
    paths = sorted(os.listdir(SETTINGS['RAW_DATA_DIR']+"train_images"))
    paths = [x.split(".")[0] for x in paths]
    with Pool(processes=4) as pool:
        for _ in tqdm.tqdm(pool.imap(job, paths), total=len(paths)):
            pass

import json
with open("./SETTINGS.json") as f:
    SETTINGS=json.load(f)

if __name__=="__main__":
    main()