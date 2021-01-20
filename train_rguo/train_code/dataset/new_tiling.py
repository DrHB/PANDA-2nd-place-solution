'''
    Tiling method from https://www.kaggle.com/akensert/panda-optimized-tiling-tf-data-dataset
'''

import numpy as np
import math
import warnings
import cv2
from numba import jit
def _mask_tissue(image, kernel_size=(7, 7), gray_threshold=220):
    """Masks tissue in image. Uses gray-scaled image, as well as
    dilation kernels and 'gap filling'
    """
    # Define elliptic kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Convert rgb to gray scale for easier masking
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Now mask the gray-scaled image (capturing tissue in biopsy)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
    # Use dilation and findContours to fill in gaps/holes in masked tissue
    mask = cv2.dilate(mask, kernel, iterations=1)
    contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 1, -1)
    return mask

def _pad_image(image, pad_len, pad_val):
    """Pads inputted image, accepts both
    2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len)), pad_val)
    elif image.ndim == 3:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), pad_val)
    return None


def _transpose_image(image):
    """Inputs an image and transposes it, accepts
    both 2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.transpose(image, (1, 0)).copy()
    elif image.ndim == 3:
        return np.transpose(image, (1, 0, 2)).copy()
    return None


def _get_tissue_parts_indices(tissue, min_consec_info):
    """If there are multiple tissue parts in 'tissue', 'tissue' will be
    split it. Each tissue part will be taken care of separately, and if
    the tissue part is less than min_consec_info, it's considered to small
    and won't be returned.
    """
    split_points = np.where(np.diff(tissue) != 1)[0] + 1
    tissue_parts = np.split(tissue, split_points)
    return [
        tp for tp in tissue_parts if len(tp) >= min_consec_info
    ]


def _get_tissue_subparts_coords(subtissue, patch_size, min_decimal_keep):
    """Inputs a tissue part resulting from '_get_tissue_parts_indices'.
    This tissue part is divided into N subparts and returned.
    Argument min_decimal_keep basically decides if we should squeeze in the
    N subparts in an area bigger than the sum of the N subparts or not.
    """
    start, end = subtissue[0], subtissue[-1]
    num_subparts = (end - start) / patch_size
    if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
        num_subparts = math.floor(num_subparts)
    else:
        num_subparts = math.ceil(num_subparts)

    excess = (num_subparts * patch_size) - (end - start)
    shift = excess // 2

    return [
        i * patch_size + start - shift
        for i in range(num_subparts)
    ]


def _eval_and_append_xy_coords(coords,
                               image,
                               mask,
                               patch_size,
                               x, y,
                               min_patch_info,
                               transposed,
                               precompute):
    """Based on computed x and y coordinates of patch:
    slices out patch from original image, flattens it,
    preprocesses it, and finally evaluates its mask.
    If patch contains more info than min_patch_info,
    the patch coordinates are kept, along with a
    value 'val1' that estimates how much information
    there is in the patch.
    """
    patch_1d = (
        image[y: y + patch_size, x:x + patch_size, :]
            .mean(axis=2)
            .reshape(-1)
    )
    idx_tissue = np.where(patch_1d <= 210)[0]
    idx_black = np.where(patch_1d < 5)[0]
    idx_background = np.where(patch_1d > 210)[0]

    if len(idx_tissue) > 0:
        patch_1d[idx_black] = 210
        patch_1d[idx_background] = 210
        val1 = int(patch_1d.mean())
        val2 = mask[y:y + patch_size, x:x + patch_size].mean()
        if val2 > min_patch_info:
            if precompute:
                if transposed:
                    coords = np.concatenate([
                        coords, [[val1, x - patch_size, y - patch_size]]
                    ])
                else:
                    coords = np.concatenate([
                        coords, [[val1, y - patch_size, x - patch_size]]
                    ])
            else:
                coords = np.concatenate([
                    coords, [[val1, y, x]]
                ])

    return coords

def compute_coords(image,
                   patch_size=256,
                   precompute=False,
                   min_patch_info=0.35,
                   min_axis_info=0.35,
                   min_consec_axis_info=0.35,
                   min_decimal_keep=0.7):
    """
    Input:
        image : 3-d np.ndarray
        patch_size : size of patches/tiles, will be of
            size (patch_size x patch_size x 3)
        precompute : If True, only coordinates will be returned,
            these coordinates match the inputted 'original' image.
            If False, both an image and coordinates will be returned,
            the coordinates does not match the inputted image but the
            image that it is returned with.
        min_patch_info : Minimum required information in patch
            (see '_eval_and_append_xy_coords')
        min_axis_info : Minimum fraction of on-bits in x/y dimension to be
            considered enough information. For x, this would be fraction of
            on-bits in x-dimension of a y:y+patch_size slice. For y, this would
            be the fraction of on-bits for the whole image in y-dimension
        min_consec_axis_info : Minimum consecutive x/y on-bits from 'min_axis_info'
            (see '_get_tissue_parts_indices')
        min_decimal_keep : Threshold for decimal point for removing "excessive" patch
            (see '_get_tissue_subparts_coords')

    Output:
        image [only if precompute is True] : similar to input image, but fits
            to the computed coordinates
        coords : the coordinates that will be used to compute the patches later on
    """

    if type(image) != np.ndarray:
        # if image is a Tensor
        image = image.numpy()

    # masked tissue will be used to compute the coordinates
    mask = _mask_tissue(image)

    # initialize coordinate accumulator
    coords = np.zeros([0, 3], dtype=int)

    # pad image and mask to make sure no tissue is potentially missed out
    image = _pad_image(image, patch_size, 'maximum')
    mask = _pad_image(mask, patch_size, 'minimum')

    y_sum = mask.sum(axis=1)
    x_sum = mask.sum(axis=0)
    # if on bits in x_sum is greater than in y_sum, the tissue is
    # likely aligned horizontally. The algorithm works better if
    # the image is aligned vertically, thus the image will be transposed
    if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
        image = _transpose_image(image)
        mask = _transpose_image(mask)
        y_sum, _ = x_sum, y_sum
        transposed = True
    else:
        transposed = False

    # where y_sum is more than the minimum number of on-bits
    y_tissue = np.where(y_sum >= (patch_size * min_axis_info))[0]

    if len(y_tissue) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute:
            return [(0, 0, 0)]
        else:
            return image, [(0, 0, 0)]

    y_tissue_parts_indices = _get_tissue_parts_indices(
        y_tissue, patch_size * min_consec_axis_info)

    if len(y_tissue_parts_indices) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute:
            return [(0, 0, 0)]
        else:
            return image, [(0, 0, 0)]

    # loop over the tissues in y-dimension
    for yidx in y_tissue_parts_indices:
        y_tissue_subparts_coords = _get_tissue_subparts_coords(
            yidx, patch_size, min_decimal_keep)

        for y in y_tissue_subparts_coords:
            # in y_slice, where x_slice_sum is more than the minimum number of on-bits
            x_slice_sum = mask[y:y + patch_size, :].sum(axis=0)
            x_tissue = np.where(x_slice_sum >= (patch_size * min_axis_info))[0]

            x_tissue_parts_indices = _get_tissue_parts_indices(
                x_tissue, patch_size * min_consec_axis_info)

            # loop over tissues in x-dimension (inside y_slice 'y:y+patch_size')
            for xidx in x_tissue_parts_indices:
                x_tissue_subparts_coords = _get_tissue_subparts_coords(
                    xidx, patch_size, min_decimal_keep)

                for x in x_tissue_subparts_coords:
                    coords = _eval_and_append_xy_coords(
                        coords, image, mask, patch_size, x, y,
                        min_patch_info, transposed, precompute
                    )

    if len(coords) < 1:
        warnings.warn("Not enough tissue in image (x-dim)", RuntimeWarning)
        if precompute:
            return [(0, 0, 0)]
        else:
            return image, [(0, 0, 0)]

    if precompute:
        return coords
    else:
        return image, coords