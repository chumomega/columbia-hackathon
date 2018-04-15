"""
This code is to test NN model and visualize output

References:
1. https://github.com/ellisdg/3DUnetCNN/blob/master/unet3d/model.py
2. https://github.com/jocicmarko/ultrasound-nerve-segmentation
"""
#!/usr/bin/python
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.layers import Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D, BatchNormalization
from keras import backend as K
#import tensorflow as tf
#from utils import *

K.set_image_data_format('channels_last')  # Tensorflow dimension ordering

data_path  = sys.argv[1] + "/"
model_path = data_path + "models/"

# dir for storing results that contains
rst_path = data_path + "test-records/"
if not os.path.exists(rst_path):
    os.makedirs(rst_path)

model_to_test = sys.argv[2]
cur_fold = sys.argv[3]
plane = sys.argv[4]
im_z = int(sys.argv[5])
im_y = int(sys.argv[6])
im_x = int(sys.argv[7])
high_range = float(sys.argv[8])
low_range = float(sys.argv[9])
margin = int(sys.argv[10])
input_image = sys.argv[11]

# prediction of trained model
pred_path = os.path.join(rst_path, "pred-%s/"%cur_fold)
if not os.path.exists(pred_path):
    os.makedirs(pred_path)

# path to save images
testim_path = os.path.join(rst_path, "testim-%s/"%cur_fold)
if not os.path.exists(testim_path):
    os.makedirs(testim_path)

def preprocess_front(imgs):
    imgs = imgs[np.newaxis, ...]
    return imgs

def preprocess(imgs):
    """add one more axis as tf require"""
    imgs = imgs[..., np.newaxis]
    return imgs


def pad_2d(image, plane, padval, xmax, ymax, zmax):
    """pad image with zeros to reach dimension as (row_max, col_max)

    Params
    -----
    image : 2D numpy array
        image to pad
    dim : char
        X / Y / Z
    padval : int
        value to pad around
    xmax, ymax, zmax : int
        dimension to reach in x/y/z axis
    """

    if plane == 'X':
        npad = ((0, ymax - image.shape[1]), (0, zmax - image.shape[2]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)
    elif plane =='Z':
        npad = ((0, xmax - image.shape[0]), (0, ymax - image.shape[1]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)

    return padded

"""
Dice Ceofficient and Cost functions for training
"""
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return  -dice_coef(y_true, y_pred)


def test(model_to_test, input_im, current_fold, plane, rst_dir):
    print("-"*50)
    print("loading model ", model_to_test)
    print("-"*50)

    model = load_model(model_path + model_to_test + '.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})

    input_image = np.load(input_im)

	# standardize test data
    input_image[input_image < low_range] = low_range
    input_image[input_image > high_range] = high_range
    input_image = (input_image - low_range) / float(high_range - low_range)

    # for creating final prediction visualization
    pred = np.zeros_like(input_image)

    try:
        # crop each slice according to smallest bounding box of each slice
        image_padded_ = pad_2d(input_image, plane, 0, im_x, im_y, im_z)
        padded_prep = preprocess_front(preprocess(image_padded_))
        pred_padded = (model.predict(padded_prep) > 0.5).astype(np.uint8).reshape(image_padded_.shape)
        pred = pred_padded[0: input_image.shape[0], 0: input_image.shape[1]]

        fig = plt.figure()
        plt.axis('off')
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title("input test image")
        ax.imshow(input_image, cmap=plt.cm.gray)

        ax = fig.add_subplot(1, 2, 2)
        ax.set_title("prediction")
        ax.imshow(pred, cmap=plt.cm.gray)

        fig.canvas.set_window_title("%s"%input_im)
        fig.savefig(pred_path + 'seg-output.jpg')
        plt.show()

    except KeyboardInterrupt:
        print('KeyboardInterrupt caught')
        raise ValueError("terminate because of keyboard interruption")


if __name__ == "__main__":

    start_time = time.time()

    test(model_to_test, input_image, cur_fold, plane, rst_path)

    print("-----------test done, total time used: %s ------------"% (time.time() - start_time))
