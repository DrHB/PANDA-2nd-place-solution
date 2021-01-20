!pip install -U git+https://github.com/qubvel/efficientnet/

import os
import PIL
import time
import json
import math
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as albu
import matplotlib.pyplot as plt
import efficientnet.tfkeras as efn
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score,confusion_matrix

SEED = 329
warnings.filterwarnings('ignore')
AUTO = tf.data.experimental.AUTOTUNE
print('Tensorflow version : {}'.format(tf.__version__))

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU : {}'.format(tpu.master()))
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("Replicas : {}".format(strategy.num_replicas_in_sync))

with open('../input/settings_json/SETTINGS.json') as f:
    CONFIG = json.load(f)

TRAIN_FLAG = True
TRAIN_FOLD = CONFIG['FOLD']
MODEL_TRAIN_CHECKPOINT_DIR = CONFIG['MODEL_TRAIN_CHECKPOINT_DIR']

if not os.path.exists(MODEL_TRAIN_CHECKPOINT_DIR):
    os.mkdir(MODEL_TRAIN_CHECKPOINT_DIR)


RADBOUD_FILE_NUMS = [
    918,
    896,
    910,
    872,
    910
]
KAROLINSKA_FILE_NUMS = [
    1076,
    1098,
    1082,
    1111,
    1081
]

gcs_path_csv = pd.read_csv('../input/gcs-path/gcs_paths.csv')

GCS_DS_PATH1 = gcs_path_csv.loc[gcs_path_csv['path_title'] == 'GCS_DS_PATH1', 'path'].values[0]
GCS_DS_PATH2 = gcs_path_csv.loc[gcs_path_csv['path_title'] == 'GCS_DS_PATH2', 'path'].values[0]
GCS_DS_PATH3 = gcs_path_csv.loc[gcs_path_csv['path_title'] == 'GCS_DS_PATH3', 'path'].values[0]
GCS_DS_PATH4 = gcs_path_csv.loc[gcs_path_csv['path_title'] == 'GCS_DS_PATH4', 'path'].values[0]
GCS_DS_PATH5 = gcs_path_csv.loc[gcs_path_csv['path_title'] == 'GCS_DS_PATH5', 'path'].values[0]

print(GCS_DS_PATH1)
print(GCS_DS_PATH2)
print(GCS_DS_PATH3)
print(GCS_DS_PATH4)
print(GCS_DS_PATH5)

total_fns = tf.io.gfile.glob(os.path.join(GCS_DS_PATH1, 'fold1/*.tfrec')) + tf.io.gfile.glob(os.path.join(GCS_DS_PATH2, 'fold2/*.tfrec')) + \
            tf.io.gfile.glob(os.path.join(GCS_DS_PATH3, 'fold3/*.tfrec')) + tf.io.gfile.glob(os.path.join(GCS_DS_PATH4, 'fold4/*.tfrec')) + \
            tf.io.gfile.glob(os.path.join(GCS_DS_PATH5, 'fold5/*.tfrec'))

train_fns = []
valid_fns = []

for fn in total_fns:
    if 'val_%d'%TRAIN_FOLD in fn:
        valid_fns.append(fn)
    else:
        train_fns.append(fn)

FOLDED_NUM_TRAIN_IMAGES = np.sum(RADBOUD_FILE_NUMS) - RADBOUD_FILE_NUMS[TRAIN_FOLD] + np.sum(KAROLINSKA_FILE_NUMS) - KAROLINSKA_FILE_NUMS[TRAIN_FOLD]
FOLDED_NUM_VALID_IMAGES =  KAROLINSKA_FILE_NUMS[TRAIN_FOLD]  + RADBOUD_FILE_NUMS[TRAIN_FOLD]

val_radboud_fns = []
val_kar_fns = []

for fn in total_fns:
    if 'val_%d'%TRAIN_FOLD in fn and 'radboud' in fn:
        val_radboud_fns.append(fn)
    elif 'val_%d'%TRAIN_FOLD in fn and 'radboud' not in fn:
        val_kar_fns.append(fn)

print(len(train_fns))
print(len(valid_fns))

x_tiles_num = 12
y_tiles_num = 12
sz = 128
N = 144
IMG_DIM = (int(sz*y_tiles_num), int(sz*x_tiles_num))
CLASSES_NUM = 1
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
EPOCHS = 40
TRAIN_EPOCHS = 25
START_EPOCH = 0
LEARNING_RATE = 5e-4
STEPS_PER_EPOCH = FOLDED_NUM_TRAIN_IMAGES // BATCH_SIZE
VALIDATION_STEPS = FOLDED_NUM_VALID_IMAGES // BATCH_SIZE
PRETRAIN_PATH = None#'./data/panda-best-weights/fold0_b3_144tiles_128tilesize_mse_huber_0905kcv_0853rcv_0706.h5'

print('*'*20)
print('Notebook info')
print('Training data : {}'.format(FOLDED_NUM_TRAIN_IMAGES))
print('Validing data : {}'.format(FOLDED_NUM_VALID_IMAGES))
print('Categorical classes : {}'.format(CLASSES_NUM))
print('Training image size : {}'.format(IMG_DIM))
print('Training epochs : {}'.format(EPOCHS))
print('Batch size : {}'.format(BATCH_SIZE))
print('*'*20)

class Dataset:

    def __init__(self,
                 filenames,
                 data_len,
                 image_shape,
                 mode = 'train',
                 batch_size=BATCH_SIZE,
                 shuffle=4096,
                 repeat=True,
                 augmentation=True,
                 rot_prob=0.25,
                 shear_prob=0.0,
                 shift_prob=0.25,
                 scale_prob=0.25,
                 rot_range=10,
                 shear_range=5,
                 shift_range=50,
                 scale_range=0.05
                 ):

        self.filenames = filenames
        self.data_len = data_len
        self.image_shape = image_shape
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.aug = augmentation

        self.rot_prob = 1.0 - rot_prob
        self.shift_prob = 1.0 - shift_prob
        self.scale_prob = 1.0 - scale_prob
        self.shear_prob = 1.0 - shear_prob
        self.rot_range = rot_range
        self.shear_range = shear_range
        self.shift_range = shift_range
        self.scale_range = scale_range

        self.log_adjust = tf.math.log(1e-6)
        self.rgb_from_hed = tf.constant([[0.65, 0.7 , 0.29],
                                         [0.07, 0.99, 0.11],
                                         [0.27, 0.57, 0.78]], dtype=tf.float32)

        self.hed_from_rgb = tf.linalg.inv(self.rgb_from_hed)

    def get_dataset(self, order=False, drop_remainder=True):

        data_options = tf.data.Options()
        if not order:
            data_options.experimental_deterministic = False

        ds = tf.data.TFRecordDataset(self.filenames, num_parallel_reads=AUTO)
        ds = ds.with_options(data_options)
        ds = ds.map(self.decode_example, num_parallel_calls=AUTO)

        if self.shuffle:
            ds = ds.shuffle(self.shuffle, seed=SEED)
        if self.repeat:
            ds = ds.repeat()
        if self.batch_size:
            ds = ds.batch(self.batch_size, drop_remainder=drop_remainder)
        if self.aug:
            ds = ds.map(self.batch_data_augmentation, num_parallel_calls=AUTO)

        ds = ds.map(self.encode_labels, num_parallel_calls=AUTO)
        ds = ds.prefetch(AUTO)

        return ds

    def encode_labels(self, images, isup_grade, data_providers):

        isup_grade = tf.expand_dims(isup_grade, axis=-1)
        data_providers = tf.expand_dims(data_providers, axis=-1)
        encoded_labels = tf.concat([isup_grade, data_providers], axis=-1)

        return images, encoded_labels

    def rgb2hed(self, images):
        images = tf.math.maximum(images, 1e-6)
        hed = (tf.math.log(images) / self.log_adjust) @ self.hed_from_rgb
        return hed

    def hed2rgb(self, heds):
        log_rgb = -(heds * -self.log_adjust) @ self.rgb_from_hed
        rgb = tf.math.exp(log_rgb)
        return tf.clip_by_value(rgb, 0.0, 1.0)

    def hedjitter(self, heds, alpha, beta):
        auged_hed = tf.identity(heds)
        alpha = tf.random.uniform(shape=[self.batch_size, 1, 1, 3], minval=1-alpha, maxval=1+alpha, dtype=tf.float32)
        beta = tf.random.uniform(shape=[self.batch_size, 1, 1, 3], minval=-beta, maxval=beta, dtype=tf.float32)
        auged_hed = tf.multiply(auged_hed, alpha) + beta
        return auged_hed

    def ColorStain(self, images, alpha=0.05, beta=0.01, return_heds=False):
        heds = self.rgb2hed(images)
        heds = self.hedjitter(heds, alpha, beta)

        if return_heds:
            return self.hed2rgb(heds), heds
        else:
            return self.hed2rgb(heds)

    def batch_data_augmentation(self, images, isup_grade, data_provider):

        select_prob = self.random_prob()
        select_prob2 = self.random_prob()

        if select_prob < 0.5:
            if self.random_prob() >= 0.5:
                images = self.ColorStain(images)
        else:
            if self.random_prob() >= 0.5:
                if select_prob2 < 0.5:
                    images = tf.image.random_brightness(images, 0.1)
                else:
                    images = tf.image.random_contrast(images, 0.7, 1.3)

            if self.random_prob() >= 0.7:
                images = tf.image.random_saturation(images, 0.8, 1.2)

        if self.random_prob() >=0.5:
            images = tf.image.rot90(images)

        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        images = self.shift_rotate_scale_shear(images)
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images, isup_grade, data_provider

    def decode_example(self, example):
        feature_format = {
            "image" : tf.io.FixedLenFeature([], tf.string),
            "isup_grade" : tf.io.FixedLenFeature([], tf.int64),
            "data_provider" : tf.io.FixedLenFeature([], tf.int64)
        }

        example = tf.io.parse_single_example(example, feature_format)

        image = tf.image.decode_jpeg(example['image'], channels=3)
        image = tf.reshape(image, [*self.image_shape, 3])

        image = tf.cast(image, tf.float32) / 255.0

        isup_grade = example['isup_grade']
        data_provider = example['data_provider'] # 0 for radboud , 1 for kar

        return image, isup_grade, data_provider

    def random_prob(self, shape=1, minval=0.0, maxval=1.0):
        return tf.random.uniform(shape=[shape], minval=minval, maxval=maxval)

    def shift_rotate_scale_shear(self, images):

        angles  = self.rot_range * tf.random.normal([self.batch_size],dtype='float32')
        shears  = self.shear_range * tf.random.normal([self.batch_size],dtype='float32')
        hshifts = self.shift_range *  tf.random.normal([self.batch_size],dtype='float32')
        vshifts = self.shift_range *  tf.random.normal([self.batch_size],dtype='float32')
        hscales = tf.random.uniform(shape=[self.batch_size], minval=-self.scale_range, maxval=self.scale_range, dtype=tf.float32) + 1.0
        wscales = tf.random.uniform(shape=[self.batch_size], minval=-self.scale_range, maxval=self.scale_range, dtype=tf.float32) + 1.0

        shear_do_cond = self.random_prob(self.batch_size) >= self.shear_prob
        shift_do_cond = self.random_prob(self.batch_size) >= self.shift_prob
        scale_do_cond = self.random_prob(self.batch_size) >= self.scale_prob
        rotate_do_cond = self.random_prob(self.batch_size) >= self.rot_prob

        zeros = tf.zeros_like(angles)
        ones = tf.ones_like(angles)

        angles = tf.where(rotate_do_cond, angles,  zeros)
        shears = tf.where(shear_do_cond,  shears,  zeros)
        hshifts = tf.where(shift_do_cond, hshifts, zeros)
        vshifts = tf.where(shift_do_cond, vshifts, zeros)
        hscales = tf.where(scale_do_cond, hscales, ones)
        wscales = tf.where(scale_do_cond, wscales, ones)

        transform_mat = self.get_batch_transform_inv_matrix(angles, shears, hshifts, vshifts, hscales, wscales)

        images = self.batch_transform(images, transform_mat, (*self.image_shape, 3), self.batch_size)

        return images

    def get_batch_transform_inv_matrix(self, angles, shears, hshifts, vshifts, hscales, wscales):

        angles = math.pi * angles / 180
        shears = math.pi * shears / 180

        ones = tf.ones_like(angles)
        zeros = tf.zeros_like(angles)

        rot_cs = tf.math.cos(angles)
        rot_ss = tf.math.sin(angles)

        she_cs = tf.math.cos(shears)
        she_ss = tf.math.sin(shears)

        inv_rot_mat = tf.stack([ rot_cs, rot_ss, zeros, -rot_ss, rot_cs, zeros,  zeros, zeros, ones], axis=0)
        inv_rot_mat = tf.reshape(tf.transpose(inv_rot_mat), shape=(self.batch_size, 3, 3))

        inv_she_mat = tf.stack([ ones, she_ss, zeros, zeros, she_cs, zeros,  zeros, zeros, ones], axis=0)
        inv_she_mat = tf.reshape(tf.transpose(inv_she_mat), shape=(self.batch_size, 3, 3))

        inv_shi_mat = tf.stack([ones, zeros, vshifts, zeros, ones, hshifts, zeros, zeros, ones], axis=0)
        inv_shi_mat = tf.reshape(tf.transpose(inv_shi_mat), shape=(self.batch_size, 3, 3))

        inv_sca_mat = tf.stack([ones/hscales, zeros, zeros, zeros, ones/wscales, zeros, zeros, zeros, ones], axis=0)
        inv_sca_mat = tf.reshape(tf.transpose(inv_sca_mat), shape=(self.batch_size, 3, 3))
        transform_mat = tf.matmul(tf.matmul(inv_rot_mat, inv_she_mat), tf.matmul(inv_shi_mat, inv_sca_mat))

        return transform_mat

    def batch_transform(self, images, inv_mat, image_shape=(512, 512, 3), batch_size=16):

        images = tf.pad(images, [[0,0], [1,1], [1,1], [0,0]])
        h, w, c = image_shape
        h, w = h+2, w+2
        cx, cy = w//2, h//2

        new_xs = tf.repeat( tf.range(-cx, cx, 1), h)
        new_ys = tf.tile( tf.range(-cy, cy, 1), [w])
        new_zs = tf.ones([h*w], dtype=tf.int32)

        #5x3x3 dot 3 * (h*w) => 5 x 3 x (h*w)
        old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))
        old_coords_x, old_coords_y = old_coords[:, 0, ]+cx, old_coords[:, 1, ]+cy

        old_coords_x = tf.clip_by_value(tf.cast(old_coords_x, tf.int32), 0, w-1)
        old_coords_y = tf.clip_by_value(tf.cast(old_coords_y, tf.int32), 0, h-1)
        gather_coords = tf.stack([old_coords_y, old_coords_x], axis=1)
        d = tf.gather_nd(images, tf.transpose(gather_coords, perm=[0, 2, 1]), batch_dims=1)

        new_images = tf.reshape(d, (batch_size, w, h, 3))
        new_images = tf.transpose(new_images, perm=[0,2,1,3])
        new_images = tf.image.crop_to_bounding_box(new_images, 1, 1, h-2, w-2)
        return new_images

train_ds = Dataset(filenames=train_fns,
                   data_len=FOLDED_NUM_TRAIN_IMAGES,
                   image_shape=IMG_DIM,
                   batch_size=BATCH_SIZE,
                   shuffle=2048,
                   ).get_dataset()

valid_ds = Dataset(filenames=valid_fns,
                   data_len=FOLDED_NUM_VALID_IMAGES,
                   image_shape=IMG_DIM,
                   batch_size=BATCH_SIZE,
                   shuffle=0,
                   augmentation=False,
                   repeat=False
                   ).get_dataset(order=True, drop_remainder=True)

class Generalized_mean_pooling2D(tf.keras.layers.Layer):
    def __init__(self, p=3, epsilon=1e-6, name='', **kwargs):
        super(Generalized_mean_pooling2D, self).__init__(name, **kwargs)
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
            'init_p': self.init_p,
            'epsilon': self.epsilon,
        })
        return config

def build_model():

    Input_layer = tf.keras.layers.Input(shape=(*IMG_DIM,3), name='image_input')
    base_model = efn.EfficientNetB3(weights='noisy-student', input_shape=(*IMG_DIM,3), include_top=False, input_tensor=Input_layer)
    x = Generalized_mean_pooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', activation='relu')(x)
    Output_layer = tf.keras.layers.Dense(units=1)(x)
    model = tf.keras.models.Model(inputs=[Input_layer], outputs=[Output_layer])

    return model


with strategy.scope():

    @tf.function
    def cosine_annealing(epoch, total_epochs=EPOCHS, cycles=1, max_lr=LEARNING_RATE, min_lr=5e-6):
        epoch += START_EPOCH
        epoch_per_cycle = int(total_epochs/cycles)
        cos_inner = ( 3.14 * (epoch % epoch_per_cycle)) / (epoch_per_cycle)
        lr = max_lr/2 * ( tf.math.cos(cos_inner) + 1)
        return tf.where(lr > min_lr, lr, min_lr)

    class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            return cosine_annealing(epoch=step//STEPS_PER_EPOCH)

    @tf.function
    def Huber_Loss(y_true, y_pred, delta=1.0):
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        abs_diff = abs(y_true-y_pred)
        loss = tf.where(abs_diff < delta , 0.5 * tf.pow(y_true-y_pred, 2), delta * (abs_diff - 0.5*delta))
        return loss

    def radboud_loss(y_trues, y_preds):
        y_trues, center_mask = tf.split(y_trues, [1,1], axis=-1)
        center_mask = tf.reshape(center_mask, (-1,))
        mae_loss = tf.keras.losses.mean_absolute_error(y_trues, y_preds)
        huber_loss = Huber_Loss(y_trues, y_preds)
        rad_loss = tf.where(center_mask==0, huber_loss, tf.zeros_like(huber_loss))
        return rad_loss

    @tf.function
    def Karolinska_loss(y_trues, y_preds):
        y_trues, center_mask = tf.split(y_trues, [1,1], axis=-1)
        center_mask = tf.reshape(center_mask, (-1,))
        mse_loss = tf.keras.losses.mean_squared_error(y_trues, y_preds)
        kar_loss = tf.where(center_mask==1, mse_loss, tf.zeros_like(mse_loss))
        return kar_loss

    rad_loss_object = radboud_loss
    kar_loss_object = Karolinska_loss

    def compute_radboud_loss(labels, predictions):
        per_example_loss = rad_loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

    def compute_karolinska_loss(labels, predictions):
        per_example_loss = kar_loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)


    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LRSchedule())

    train_loss = tf.keras.metrics.Sum()
    valid_loss = tf.keras.metrics.Sum()
    train_rad_loss = tf.keras.metrics.Sum()
    valid_rad_loss = tf.keras.metrics.Sum()
    train_kar_loss = tf.keras.metrics.Sum()
    valid_kar_loss = tf.keras.metrics.Sum()

if PRETRAIN_PATH:
    print('load model pretrain weights..')
    model.load_weights(PRETRAIN_PATH)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as grad_tape:
        preds = model(images, training=True)

        rad_loss = compute_radboud_loss(labels, preds)
        kar_loss = compute_karolinska_loss(labels, preds)
        loss = rad_loss + kar_loss

    grads = grad_tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss)
    train_rad_loss.update_state(rad_loss)
    train_kar_loss.update_state(kar_loss)

@tf.function
def valid_step(images, labels):
    preds = model(images, training=False)

    rad_loss = compute_radboud_loss(labels, preds)
    kar_loss = compute_karolinska_loss(labels, preds)
    loss = rad_loss + kar_loss

    valid_loss.update_state(loss)
    valid_rad_loss.update_state(rad_loss)
    valid_kar_loss.update_state(kar_loss)


@tf.function
def inference_step(images):
    preds = model(images, training=False)
    return preds

best_valid_loss = 10000
best_rad_valid_loss = 10000
best_kar_valid_loss = 10000
history = {
    'train_loss' : [],
    'valid_loss' : [],
    'train_rad_loss' : [],
    'valid_rad_loss' : [],
    'train_kar_loss' : [],
    'valid_kar_loss' : [],
    'qwk' : []
}

if TRAIN_FLAG:
    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    valid_dist_ds = strategy.experimental_distribute_dataset(valid_ds)
    print("Steps per epoch:", STEPS_PER_EPOCH, "Valid steps per epoch:", VALIDATION_STEPS)
    epoch = 0
    start_time = time.time()
    for step, (images, labels) in enumerate(train_dist_ds):

        strategy.run(train_step, args=(images, labels))
        print('=', end='', flush=True)

        if ((step+1) // STEPS_PER_EPOCH) > epoch:
            print('|', end='',flush=True)

            for images, labels in valid_dist_ds:
                print('=', end='', flush=True)
                strategy.run(valid_step, args=(images, labels))

            history['train_loss'].append(train_loss.result().numpy() / STEPS_PER_EPOCH)
            history['valid_loss'].append(valid_loss.result().numpy() / VALIDATION_STEPS)

            history['train_rad_loss'].append(train_rad_loss.result().numpy() / STEPS_PER_EPOCH)
            history['train_kar_loss'].append(train_kar_loss.result().numpy() / STEPS_PER_EPOCH)

            history['valid_rad_loss'].append(valid_rad_loss.result().numpy() / VALIDATION_STEPS)
            history['valid_kar_loss'].append(valid_kar_loss.result().numpy() / VALIDATION_STEPS)

            print('\nEPOCH {:d}/{:d}'.format(epoch+1, EPOCHS))
            print('loss: {:0.4f}'.format(history['train_loss'][-1]), 'valid_loss: {:0.4f}'.format(history['valid_loss'][-1]))
            print('radboud loss : {:0.4f}'.format(history['train_rad_loss'][-1]), 'valid radboud loss : {:0.4f}'.format(history['valid_rad_loss'][-1]))
            print('karolinska loss : {:0.4f}'.format(history['train_kar_loss'][-1]), 'valid karolinska loss : {:0.4f}'.format(history['valid_kar_loss'][-1]))

            train_loss.reset_states()
            valid_loss.reset_states()
            train_rad_loss.reset_states()
            valid_rad_loss.reset_states()
            train_kar_loss.reset_states()
            valid_kar_loss.reset_states()

            if history['valid_loss'][-1] < best_valid_loss:
                print('Validation loss improve from {} to {}, save model checkpoint'.format(best_valid_loss, history['valid_loss'][-1]))
                model.save(os.path.join(CONFIG['MODEL_TRAIN_CHECKPOINT_DIR'], 'model.h5'))
                best_valid_loss = history['valid_loss'][-1]
                patience=0

            if history['valid_rad_loss'][-1] < best_rad_valid_loss:
                print('Validation rad loss improve from {:0.4f} to {:0.4f}, save rad checkpoint'.format(best_rad_valid_loss, history['valid_rad_loss'][-1]))
                model.save(os.path.join(CONFIG['MODEL_TRAIN_CHECKPOINT_DIR'], 'rad_model.h5'))
                best_rad_valid_loss = history['valid_rad_loss'][-1]

            if history['valid_kar_loss'][-1] < best_kar_valid_loss:
                print('Validation kar loss improve from {:0.4f} to {:0.4f}, save kar checkpoint'.format(best_kar_valid_loss, history['valid_kar_loss'][-1]))
                model.save(os.path.join(CONFIG['MODEL_TRAIN_CHECKPOINT_DIR'], 'kar_model.h5'))
                best_kar_valid_loss = history['valid_kar_loss'][-1]

            print('Spending time : {} second...'.format(time.time()-start_time))
            start_time = time.time()
            epoch = (step+1) // STEPS_PER_EPOCH
            if epoch >= TRAIN_EPOCHS:
                break

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


rad_valid_ds = Dataset(filenames=val_radboud_fns,
                      data_len=FOLDED_NUM_VALID_IMAGES,
                      image_shape=IMG_DIM,
                      batch_size=BATCH_SIZE,
                      shuffle=0,
                      augmentation=False,
                      repeat=False,
                      ).get_dataset(order=True, drop_remainder=False)

kar_valid_ds = Dataset(filenames=val_kar_fns,
                      data_len=FOLDED_NUM_VALID_IMAGES,
                      image_shape=IMG_DIM,
                      batch_size=BATCH_SIZE,
                      shuffle=0,
                      augmentation=False,
                      repeat=False,
                      ).get_dataset(order=True, drop_remainder=False)


def valid_qwk(valid_ds, data_center='Both'):
    predictions = []
    groundtruths = []
    for images, labels in valid_ds:
        print('=', end='', flush=True)
        preds = inference_step(images)
        preds = prediction_decode(preds.numpy())
        labels = prediction_decode(labels.numpy()[:,0])

        groundtruths += list(labels)
        predictions += list(preds)

    qwk = cohen_kappa_score(groundtruths, predictions, labels=None, weights= 'quadratic', sample_weight=None)
    confusion_mat = confusion_matrix(groundtruths, predictions)
    print('\n {} data center validation qwk : {:0.4f}'.format(data_center, qwk))

    return confusion_mat, groundtruths, predictions

if TRAIN_FLAG:
    model.save(os.path.join(CONFIG['MODEL_TRAIN_CHECKPOINT_DIR'], 'model_finalcheckpoint.h5'))
    model.load_weights(os.path.join(CONFIG['MODEL_TRAIN_CHECKPOINT_DIR'], 'model.h5'))

print('Validation qwk on overall validation loss checkpoint :')
print('-'*20)
rad_confusion_mat, rad_groundtruths, rad_predictions = valid_qwk(rad_valid_ds, 'Radboud')
kar_confusion_mat, kar_groundtruths, kar_predictions = valid_qwk(kar_valid_ds, 'Karolinska')

overall_groundtruths = list(rad_groundtruths) + list(kar_groundtruths)
overall_predictions = list(rad_predictions) + list(kar_predictions)
overall_confusion_mat = confusion_matrix(overall_groundtruths, overall_predictions)
overall_qwk = cohen_kappa_score(overall_groundtruths, overall_predictions, labels=None, weights= 'quadratic', sample_weight=None)

print(' both data center validation qwk : {:0.4f}'.format(overall_qwk))

print('overall confusion matrix : ')
print(overall_confusion_mat)
print('Radboud confusion matrix : ')
print(rad_confusion_mat)
print('Karolinska confusion matrix : ')
print(kar_confusion_mat)
print('-'*20)
