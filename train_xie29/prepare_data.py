import os
import time
import json
import cv2
import openslide
import skimage.io
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import Image, display
from sklearn.model_selection import StratifiedKFold
AUTO = tf.data.experimental.AUTOTUNE
print('tensorflow version : {}'.format(tf.__version__))

tf.random.set_seed(2020)
np.random.seed(2020)


with open('./SETTINGS.json') as f:
    CONFIG = json.load(f)

def get_foreground(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(grayscale, 200, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    if np.sum(thresh) == 0:
        return image
    bbox = cv2.boundingRect(thresh)
    x, y, w, h = bbox
    foreground = image[y:y+h, x:x+w]
    return foreground

def generate_mask(mask, provider):
    masks = np.zeros((*mask.shape, 3))

    if provider == 'karolinska':
        masks[:, :, 0] = (mask == 0).astype(np.uint8) * 255
        masks[:, :, 1] = (mask == 1).astype(np.uint8) * 255
        masks[:, :, 2] = (mask == 2).astype(np.uint8) * 255
    else:
        masks[:, :, 0] = (mask == 0).astype(np.uint8) * 255
        masks[:, :, 1] = ( (mask == 1) | (mask == 2) ).astype(np.uint8) * 255
        masks[:, :, 2] = ( (mask == 3) | (mask == 4) | (mask == 5)).astype(np.uint8) * 255

    return masks

def gleason_score_mapping(gleason_score):

    table = {
        '0+0' : 0,
        'negative' : 0,
        '3+3' : 1,
        '3+4' : 2,
        '4+3' : 3,
        '4+4' : 4,
        '3+5' : 5,
        '4+5' : 6,
        '5+3' : 7,
        '5+4' : 8,
        '5+5' : 9
    }
    return int(table[gleason_score])

def get_example(image, isup_score, image_id, data_provider, gleason_1, gleason_2):

    tfimage = tf.cast(tf.convert_to_tensor(image), tf.uint8)
    tfimage = tf.image.encode_jpeg(tfimage, optimize_size=True, chroma_downsampling=False)

    example = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tfimage.numpy()])),
                    "id" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id.encode('utf-8')])),
                    "isup_grade" : tf.train.Feature(int64_list=tf.train.Int64List(value=[isup_score])),
                    "data_provider" : tf.train.Feature(int64_list=tf.train.Int64List(value=[data_provider])),
                    "gleason_1" : tf.train.Feature(int64_list=tf.train.Int64List(value=[gleason_1])),
                    "gleason_2" : tf.train.Feature(int64_list=tf.train.Int64List(value=[gleason_2])),
    }))

    return example


def get_example_with_g_fold(image, isup_score, mixed_isup_score, image_id, data_provider, gleason_1, gleason_2):
    tfimage = tf.cast(tf.convert_to_tensor(image), tf.uint8)
    tfimage = tf.image.encode_jpeg(tfimage, optimize_size=True, chroma_downsampling=False)

    example = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tfimage.numpy()])),
                    "id" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id.encode('utf-8')])),
                    "isup_grade" : tf.train.Feature(int64_list=tf.train.Int64List(value=[isup_score])),
                    "mixed_isup_grade" : tf.train.Feature(float_list=tf.train.FloatList(value=[mixed_isup_score])),
                    "data_provider" : tf.train.Feature(int64_list=tf.train.Int64List(value=[data_provider])),
                    "gleason_1" : tf.train.Feature(int64_list=tf.train.Int64List(value=[gleason_1])),
                    "gleason_2" : tf.train.Feature(int64_list=tf.train.Int64List(value=[gleason_2])),
    }))

    return example


def glue_tiles(images, y_tiles_num=3, x_tiles_num=4, sz=128):
    glued_image = np.zeros((sz*y_tiles_num, sz*x_tiles_num, 3), np.uint8)
    for i, image in enumerate(images):
        y_start = int(i / x_tiles_num) * sz
        y_end = y_start + sz
        x_start = int(i % x_tiles_num) * sz
        x_end = x_start + sz

        glued_image[y_start:y_end, x_start:x_end, :] = image
    return glued_image


MAIN_DIR = CONFIG['RAW_DATA_DIR']
CUSTOMIZE_DIR = CONFIG['CUSTOMIZE_DATA_DIR']
CLEAN_DATA_DIR = CONFIG['CLEAN_DATA_DIR']

if not os.path.exists(CLEAN_DATA_DIR):
    os.mkdir(CLEAN_DATA_DIR)

TRAIN_IMG_DIR = os.path.join(MAIN_DIR, 'train_images')

use_g_fold = True

if use_g_fold:
    train_csv = pd.read_csv(os.path.join(CUSTOMIZE_DIR, 'cleanlabv2.csv'))
else:
    train_csv = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv'))

train_img_fns = os.listdir(TRAIN_IMG_DIR)

noise_csv = pd.read_csv(os.path.join(CUSTOMIZE_DIR, 'PANDA_Suspicious_Slides.csv'))
for image_id in noise_csv['image_id'].values:
    train_csv = train_csv[train_csv['image_id'] != image_id]

train_csv = train_csv.replace(to_replace='negative', value='0+0')

if not use_g_fold:

    radboud_csv = train_csv[train_csv['data_provider'] == 'radboud']
    karolinska_csv = train_csv[train_csv['data_provider'] != 'radboud']

    print(radboud_csv.shape)
    print(karolinska_csv.shape)

    splits = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    splits = list(splits.split(radboud_csv, radboud_csv.isup_grade))
    fold_splits = np.zeros(len(radboud_csv)).astype(np.int)
    for i in range(5):
        fold_splits[splits[i][1]]=i
    radboud_csv['fold'] = fold_splits
    print(radboud_csv.head(5))

    splits = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    splits = list(splits.split(karolinska_csv, karolinska_csv.isup_grade))
    fold_splits = np.zeros(len(karolinska_csv)).astype(np.int)
    for i in range(5):
        fold_splits[splits[i][1]]=i
    karolinska_csv['fold'] = fold_splits
    print(karolinska_csv.head(5))

    train_csv = pd.concat([radboud_csv, karolinska_csv])
    print(train_csv.shape)

for center in ['radboud', 'karolinska']:
    for fold in range(5):
        target_dir = os.path.join(CLEAN_DATA_DIR, 'fold{}'.format(fold+1))
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        folded_file_nums = np.sum(train_csv['fold'] == fold)
        best_record_file_num = folded_file_nums * 4
        big_slide_count = 0
        N = 144
        sz = 128
        resize_sz= 128
        x_tiles_num= 12
        y_tiles_num= 12
        tfrecord_file_num = 0
        num = 0
        tfrecord_filename = center + '_panda_%d_val_%d.tfrec' %(tfrecord_file_num,fold)
        writer = tf.io.TFRecordWriter(os.path.join(target_dir, tfrecord_filename))
        sample_csv = train_csv[train_csv['fold'] == fold]
        sample_csv = sample_csv[sample_csv['data_provider'] == center]

        for image_id in tqdm(sample_csv['image_id'].values):
            data_provider = train_csv[train_csv['image_id'] == image_id]['data_provider'].values[0]
            gleason_1 = int(train_csv[train_csv['image_id'] == image_id]['gleason_score'].values[0][0])
            gleason_2 = int(train_csv[train_csv['image_id'] == image_id]['gleason_score'].values[0][-1])

            isup_grade=-1
            mixed_isup_grade = None
            isup_grade = None
            if use_g_fold:
                mixed_isup_grade = train_csv[train_csv['image_id'] == image_id]['isup_grade_x'].values[0] * 0.7 + \
                                   train_csv[train_csv['image_id'] == image_id]['prediction_reg'].values[0] * 0.3
                isup_grade = train_csv[train_csv['image_id'] == image_id]['isup_grade_x'].values[0]
            else:
                isup_grade = train_csv[train_csv['image_id'] == image_id]['isup_grade'].values[0]

            slide_index = 1

            image = skimage.io.MultiImage(os.path.join(TRAIN_IMG_DIR,image_id+'.tiff'))[slide_index]

            if image.shape[0] + image.shape[1] > 15000:
                big_slide_count += 1
                slide_index = -1 if fold == 2 else slide_index

            img_dimension = image.shape[:2]
            pad0,pad1 = (sz - img_dimension[0]%sz)%sz, (sz - img_dimension[1]%sz)%sz
            image = np.pad(image,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)

            image = image.reshape(image.shape[0]//sz, sz, image.shape[1]//sz, sz, 3)
            image = image.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

            if len(image) < N:
                image = np.pad(image, [[0, N-len(image)], [0,0], [0,0], [0,0]], constant_values=255)

            indices = np.argsort(image.reshape(image.shape[0], -1).sum(-1))[:N]
            image = image[indices]

            slim_images = []
            for i,m in zip(image, range(len(image))):
                i = get_foreground(i)
                slim_images.append(cv2.resize(i, (resize_sz, resize_sz), cv2.INTER_AREA))

            del image
            glued_image = glue_tiles(slim_images, y_tiles_num, x_tiles_num, resize_sz)
            data_provider_encoded = 0 if data_provider == 'radboud' else 1

            num += 1
            if num >= best_record_file_num:
                num = 1
                tfrecord_file_num += 1
                tfrecord_filename = center + '_panda_%d_val_%d.tfrec' %(tfrecord_file_num,fold)
                writer = tf.io.TFRecordWriter(os.path.join(target_dir, tfrecord_filename))

            if use_g_fold:
                example = get_example_with_g_fold(glued_image, isup_grade, mixed_isup_grade, image_id, data_provider_encoded, gleason_1, gleason_2)
            else:
                example = get_example(glued_image, isup_grade, image_id, data_provider_encoded, gleason_1, gleason_2)
            writer.write(example.SerializeToString())
            serialized_example = example.SerializeToString()
        print('Slides bigger than 15000 threshold : {}'.format(big_slide_count))
        writer.close()
