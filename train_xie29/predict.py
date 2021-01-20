import os
import gc
import cv2
import PIL
import copy
import time
import json
import math
import warnings
import skimage.io
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as albu
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.preprocessing.image import load_img

print('Tensorflow version : {}'.format(tf.__version__))
def get_xie29_raw_prediction():

    class FixedDropout(tf.keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = tf.keras.backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    class Generalized_mean_pooling2D(tf.keras.layers.Layer):
        def __init__(self, p=3, epsilon=1e-6, name='', trainable=True, **kwargs):
            super(Generalized_mean_pooling2D, self).__init__()
            self.init_p = p
            self.epsilon = epsilon

        def build(self, input_shape):
            if isinstance(input_shape, list) or len(input_shape) != 4:
                raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions(b, h, w, c)')
            self.build_shape = input_shape
            self.p = self.add_weight(
                      name='p',
                      shape=[1,],
                      initializer=tf.keras.initializers.Constant(value=self.init_p),
                      regularizer=None,
                      trainable=True,
                      dtype=tf.float32
                      )
            self.built=True

        def call(self, inputs):
            input_shape = inputs.get_shape()
            if isinstance(inputs, list) or len(input_shape) != 4:
                raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions(b, h, w, c)')
            return (tf.reduce_mean(tf.abs(inputs**self.p), axis=[1,2], keepdims=False) + self.epsilon)**(1.0/self.p)

        def get_config(self):

            config = super().get_config().copy()
            config.update({
                'p': self.init_p,
                'epsilon': self.epsilon
            })
            return config

    class TestGenerator(tf.keras.utils.Sequence):

        def __init__(self,
                     image_shape,
                     batch_size,
                     load_dir,
                     test_df
                     ):

            self.image_shape = image_shape
            self.batch_size = batch_size
            self.test_df = test_df
            self.image_ids = test_df['image_id'].values
            self.load_dir = load_dir
            self.indices = range(test_df.shape[0])

        def __len__(self):
            return self.test_df.shape[0] // self.batch_size

        def __getitem__(self, index):
            batch_indices = self.indices[self.batch_size * index : self.batch_size * (index+1)]
            image_ids = self.image_ids[batch_indices]
            batch_images = [self.__getimages__(image_id) for image_id in image_ids]
            return np.stack(batch_images)

        def glue_tiles(self, images, y_tiles_num=3, x_tiles_num=4, sz=128):
            glued_image = np.zeros((sz*y_tiles_num, sz*x_tiles_num, 3), np.uint8)
            for i, image in enumerate(images):
                y_start = int(i / x_tiles_num) * sz
                y_end = y_start + sz
                x_start = int(i % x_tiles_num) * sz
                x_end = x_start + sz

                glued_image[y_start:y_end, x_start:x_end, :] = image
            return glued_image


        def get_foreground(self, image):
            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(grayscale, 200, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            if np.sum(thresh) == 0:
                return image
            bbox = cv2.boundingRect(thresh)
            x, y, w, h = bbox
            foreground = image[y:y+h, x:x+w]
            return foreground

        def __getimages__(self, img_id):
            read_in_path = os.path.join(self.load_dir, img_id + '.tiff')
            img = skimage.io.MultiImage(read_in_path)[1]
            shape = img.shape
            pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
            img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)
            img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
            img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
            if len(img) < N:
                img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
            idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
            img = img[idxs]
            slim_images = []
            for i in img:
                i = self.get_foreground(i)
                slim_images.append(cv2.resize(i, (resize_sz, resize_sz), cv2.INTER_AREA))
            del img
            glued_image = self.glue_tiles(slim_images, y_tiles_num, x_tiles_num, resize_sz)
            return glued_image/255.0

    def prediction_decode(y_preds):

        for i, pred in enumerate(y_preds):
            if pred < 0.5:
                y_preds[i] = 0
            elif pred >= 0.5 and pred < 1.5:
                y_preds[i] = 1
            elif pred >= 1.5 and pred < 2.5:
                y_preds[i] = 2
            elif pred >= 2.5 and pred < 3.5:
                y_preds[i] = 3
            elif pred >= 3.5 and pred < 4.5:
                y_preds[i] = 4
            else:
                y_preds[i] = 5

        return y_preds.astype(np.int32)

    @tf.function
    def inference_step(images):
        preds = model(images, training=False)
        return preds


    def inference(model, test_generator):

        prediction = []

        for step in range(test_generator.__len__()):
            print('=', end='', flush=True)
            images = test_generator.__getitem__(step)
            preds = model(images, training=False)
            preds = preds[0].numpy()
            for y_pred in preds:
                prediction.append(y_pred)
        print('')
        return np.array(prediction, dtype=np.float32)



    get_custom_objects().update({'swish': tf.keras.layers.Activation(tf.nn.swish)})
    get_custom_objects().update({'FixedDropout':FixedDropout})
    get_custom_objects().update({'Generalized_mean_pooling2D':Generalized_mean_pooling2D})

    with open('./SETTINGS.json') as f:
        CONFIG = json.load(f)

    MAIN_DIR = CONFIG['RAW_DATA_TEST']
    sample_csv = pd.read_csv(os.path.join(MAIN_DIR, 'sample_submission.csv'))

    #Hyper parameters
    ########################################
    x_tiles_num = 12
    y_tiles_num = 12
    sz = 128
    resize_sz = 128
    IMG_DIM = (int(resize_sz*y_tiles_num), int(resize_sz*x_tiles_num))
    CLASSES_NUM = 1
    BATCH_SIZE = 32
    N= 144
    PRETRAIN_PATH = [os.path.join(CONFIG['MODEL_CHECKPOINT_DIR'], 'fold0_b3_144tiles_128tilesize_mse_huber_0905kcv_0853rcv_0706.h5'),
                     os.path.join(CONFIG['MODEL_CHECKPOINT_DIR'], 'fold2_b3_144tiles_128tilesize_mse_huber_09174kcv_0841rcv_0719.h5')
                    ]

    raw_prediction = sample_csv['isup_grade'].values

    test_generator = TestGenerator(image_shape=IMG_DIM,
                                   batch_size=1,
                                   load_dir=os.path.join(MAIN_DIR, 'test_images'),
                                   test_df=sample_csv)

    for idx in range(len(PRETRAIN_PATH)):
        if PRETRAIN_PATH[idx]:
            print('load xies model pretrain weights..')
            model = tf.keras.models.load_model(PRETRAIN_PATH[idx],
                                               custom_objects={
                                                   'Generalized_mean_pooling2D' : Generalized_mean_pooling2D
                                                })

            if os.path.exists(os.path.join(MAIN_DIR, 'test_images')):
                if idx == 0:
                    raw_prediction = inference(model, test_generator)
                else:
                    raw_prediction += inference(model, test_generator)

            print('Clean usage memory of xie29 ... ')

            del model
            tf.keras.backend.clear_session()
            gc.collect()

    #averaging raw prediction
    raw_prediction = raw_prediction / len(PRETRAIN_PATH)
    decoded_prediction = prediction_decode(raw_prediction)

    xie29_only_csv = copy.deepcopy(sample_csv)
    ensemble_csv = copy.deepcopy(sample_csv)

    xie29_only_csv['isup_grade'] = prediction_decode(raw_prediction)
    ensemble_csv['isup_grade'] = raw_prediction

    xie29_only_csv.to_csv(os.path.join(CONFIG['SUBMISSION_DIR'], 'submission.csv'), index=False)
    ensemble_csv.to_csv(os.path.join(CONFIG['SUBMISSION_DIR'], 'xie29_ensemble_submission.csv'), index=False)


get_xie29_raw_prediction()
