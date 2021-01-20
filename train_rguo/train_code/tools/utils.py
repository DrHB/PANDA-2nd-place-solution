from numba import jit
import numpy as np
from functools import partial
import scipy.optimize
import pandas as pd
import cv2
import torch

def qwk3(a1, a2, max_rat=5):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3,4,5])

        return -qwk3(y, X_p)

    def fit(self, X, y, initial_coef):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = scipy.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3,4,5])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


def save_parameters(options,filename):
    with open(filename,"w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key,options[key]))


def classify(x,thres=(0.5,1.5,2.5,3.5,4.5)):
    if x<thres[0]:
        return 0
    elif x<thres[1]:
        return 1
    elif x<thres[2]:
        return 2
    elif x<thres[3]:
        return 3
    elif x<thres[4]:
        return 4
    else:
        return 5

def get_prediction(pred):
    return np.array([classify(x) for x in pred])

def tensor_to_image(image,mean=(0.90949707,0.8188697,0.87795304), std=(0.36357649,0.49984502,0.40477625)):
    mean=np.array(mean,dtype=np.float32)
    image=image.detach().cpu().numpy().transpose(1,2,0)
    image=image*std+mean
    image=(image*255).astype(np.uint8)
    return cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


import albumentations as A
from albumentations.core.transforms_interface import DualTransform,ImageOnlyTransform
import cv2
import random
def HETransform(img,factor):
    shape=img.shape
    #img=img.astype(np.float32)
    white_mask=(img.mean(axis=2,keepdims=True)>=250).astype(np.uint8)
    DeconvMatrix = np.array([[1.88, -0.07, -0.6],
                             [-1.02, 1.13, -0.48],
                             [-0.55, -0.13, 1.57]])

    # decompose: he = D*rgb
    HEcomponent = np.dot(DeconvMatrix, img.reshape(-1, 3).T).T.reshape(img.shape[0], img.shape[1], 3)
    # enhance Esoin
    HEcomponent[:, :, 1] = HEcomponent[:, :, 1] * factor
    # recons
    recons = np.dot(np.linalg.inv(DeconvMatrix), HEcomponent.reshape(-1, 3).T).T.reshape(*shape)
    recons = np.maximum(recons, 0)
    recons = recons / recons.max()*255
    recons = recons.astype(np.uint8)
    return recons*(1-white_mask)+img*white_mask

class RandomHETransform(ImageOnlyTransform):

    def __init__(self,sigma=0.1,always_apply=False, p=0.5):
        super(RandomHETransform, self).__init__(always_apply, p)
        self.sigma=sigma


    def apply(self, img, factor,**params):
        return HETransform(img,factor)

    def get_params(self):
        return {"factor":np.random.normal(1.,self.sigma)}

    def get_transform_init_args_names(self):
        return ("factor",)

