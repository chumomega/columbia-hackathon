#import os
#import numpy as np
#import matplotlib.pyplot as plt
#, methods = ['GET', 'POST']
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

#
#
#UPLOAD_FOLDER = os.path.basename('uploads')
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
#from keras.models import load_model
#from keras import backend as K
#
#K.set_image_data_format('channels_last')  # Tensorflow dimension ordering
#
#cwd = os.getcwd()
#print (cwd)
#data_path  =  cwd + "/segmentation/data/"
#model_path = data_path + "models/"
#
#model_to_test = "unet_fd0_Z_ep10_lr1e-5"
#current_fold = 0
#plane = "Z"
## model to test
#model_test="unet_fd0_Z_ep$10_lr1e-5.h5"
## now just fix the input image
#input_image="uploads/*.npy"
#im_z = int(160)
#im_y = int(256)
#im_x = int(192)
#high_range = float(240)
#low_range = float(-100)
#margin = int(20)
#
## dir for storing results that contains
#rst_path = data_path + "test-records/"
#if not os.path.exists(rst_path):
#    os.makedirs(rst_path)
#
## prediction of trained model
#pred_path = os.path.join(rst_path, "pred-%s/"%current_fold)
#if not os.path.exists(pred_path):
#    os.makedirs(pred_path)
#
## path to save images
#testim_path = os.path.join(rst_path, "testim-%s/"%current_fold)
#if not os.path.exists(testim_path):
#    os.makedirs(testim_path)
#
#def preprocess_front(imgs):
#    imgs = imgs[np.newaxis, ...]
#    return imgs
#
#def preprocess(imgs):
#    """add one more axis as tf require"""
#    imgs = imgs[..., np.newaxis]
#    return imgs
#
#
#def pad_2d(image, plane, padval, xmax, ymax, zmax):
#    """pad image with zeros to reach dimension as (row_max, col_max)
#
#    Params
#    -----
#    image : 2D numpy array
#        image to pad
#    dim : char
#        X / Y / Z
#    padval : int
#        value to pad around
#    xmax, ymax, zmax : int
#        dimension to reach in x/y/z axis
#    """
#
#    if plane == 'X':
#        npad = ((0, ymax - image.shape[1]), (0, zmax - image.shape[2]))
#        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)
#    elif plane =='Z':
#        npad = ((0, xmax - image.shape[0]), (0, ymax - image.shape[1]))
#        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)
#
#    return padded
#
#
#"""
#Dice Ceofficient and Cost functions for training
#"""
#smooth = 1.
#
#def dice_coef(y_true, y_pred):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#def dice_coef_loss(y_true, y_pred):
#    return  -dice_coef(y_true, y_pred)
#
## def test(model_to_test, input_im, current_fold, plane, rst_dir):
#
#
#@app.route('/upload', methods=['POST'])
#def upload_file():
#    file = request.files['image']
#    input_im = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#    
#    print ("-"*50)
#    print ("loading model ", model_to_test)
#    print ("-"*50)
#
#    model = load_model(model_path + model_to_test + '.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})
#
#    input_image = np.load(input_im)
#
#	# standardize test data
#    input_image[input_image < low_range] = low_range
#    input_image[input_image > high_range] = high_range
#    input_image = (input_image - low_range) / float(high_range - low_range)
#
#    # for creating final prediction visualization
#    pred = np.zeros_like(input_image)
#
#    try:
#        # crop each slice according to smallest bounding box of each slice
#        image_padded_ = pad_2d(input_image, plane, 0, im_x, im_y, im_z)
#        padded_prep = preprocess_front(preprocess(image_padded_))
#        pred_padded = (model.predict(padded_prep) > 0.5).astype(np.uint8).reshape(image_padded_.shape)
#        pred = pred_padded[0: input_image.shape[0], 0: input_image.shape[1]]
#
#        fig = plt.figure()
#        ax = fig.add_subplot(1, 2, 1)
#        ax.set_title("input test image")
#        ax.imshow(input_image, cmap=plt.cm.gray)
#
#        ax = fig.add_subplot(1, 2, 2)
#        ax.set_title("prediction")
#        ax.imshow(pred, cmap=plt.cm.gray)
#
#        fig.canvas.set_window_title("%s"%input_im)
#        fig.savefig(pred_path + 'seg-output.jpg')
#        plt.show()
#
#    except KeyboardInterrupt:
#        raise ValueError("terminate because of keyboard interruption")
#    
#    # save uploaded file
#    file.save(input_im)
#
#    return render_template('index.html')

if __name__ =='__main__':
	app.run(debug=True, use_reloader=True)
    
    