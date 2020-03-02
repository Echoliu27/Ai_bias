import numpy as np
import pandas as pd

import os
from os import listdir
from os.path import isfile, join

from scipy.ndimage import interpolation
from scipy.misc import imread, imsave, imresize
import cv2
import PIL
from PIL import Image

import argparse
import pickle
import h5py

from keras.models import load_model



def predict(model1,model2,DATA_FILE,most_frequent_class = 4, new_size = 128):
    ethnicity = ["Asian",	"Black", "Hispanic","Indian",	"White"]
    X = []
    id_img = []
    pred = []
    
    # for this specific dataset only
    with h5py.File(DATA_FILE, 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        train_pixel_key = list(f.keys())[5]
        publicT_pixel_key = list(f.keys())[3]
        privateT_pixel_key = list(f.keys())[1]

        # Get the data
        train_pixel = list(f[train_pixel_key])
        publicT_pixel = list(f[publicT_pixel_key])
        privateT_pixel = list(f[privateT_pixel_key])
    
    pixel_wanted = train_pixel
    
    for t in range(len(pixel_wanted)):
        img = pixel_wanted[t].reshape(48,48)
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        basewidth = 128
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        im = img.resize((basewidth,hsize), Image.ANTIALIAS)

        # discard image if too small or has some problem
        if len(np.array(im).shape)<3:
            pred.append(ethnicity[most_frequent_class])
            id_img.append(t)
#             print(files[t])
#             print(ethnicity[most_frequent_class])
#             print("...")
            continue
        if np.array(im).shape[0] < new_size or np.array(im).shape[1] < new_size or np.array(im).shape[2] != 3:
            pred.append(ethnicity[most_frequent_class])
            id_img.append(t)
#             print(files[t])
#             print(ethnicity[most_frequent_class])
#             print("...")
            continue

        # preprocess image with openCV
        image = np.array(im)
        image_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=1,minSize=(MIN_FACE_SIZE,MIN_FACE_SIZE),flags=0)
        num_faces = len(faces)
        height, width, channels = image.shape

        # Predict with the faces model if faces are detected
        if num_faces >0:
            pred_temp = np.zeros([1,5])
            for i,(x,y,w,h) in enumerate(faces):
                y0 = (y-PADDING) if y-PADDING>=0 else 0
                y1 = (y+h+PADDING) if y+h+PADDING <= height else height
                x0 = (x-PADDING) if x-PADDING>=0 else 0
                x1 = (x+w+PADDING) if (x+w+PADDING) <= width else width
                sub_img = image[y0:y1, x0:x1]
                sub_img = cv2.resize(sub_img, (MIN_FACE_SIZE+PADDING, MIN_FACE_SIZE+PADDING))
                x = np.expand_dims(sub_img, axis=0)
                #print(y_test[t,:])
                pred_ = np.round(model1.predict( (x- 127.5) / 255),2)
                pred_temp += pred_
#             pred.append(ethnicity[np.argmax(pred_temp)])
            pred.append(np.argmax(pred_temp))
#             print(files[t])
#             print(ethnicity[np.argmax(pred_temp)])
#             print("...")    # Predict with the full picture model if faces are not detected
        else:
            reduced_size = int(new_size), int(new_size)
            imr = im.resize(reduced_size, Image.ANTIALIAS)
            imr = np.array(imr)
            x = np.expand_dims(imr, axis=0)
            pred_ = np.round(model2.predict( (x- 127.5) / 255),2)
#             pred.append(ethnicity[np.argmax(pred_)])
            pred.append(np.argmax(pred_))
#             print(files[t])
#             print(ethnicity[np.argmax(pred_)])
#             print("...")
        # save id images for getting labels
        id_img.append(t)
    return np.vstack([id_img,np.array(pred)])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', default="~/cluster/", type=str)
    parser.add_argument('--labels_data_dir', default="None", type=str)
    args = parser.parse_args()

    print("import models...")
    model1 = load_model('./models/run20-model.h5')
    model1.load_weights('./models/run20-model.h5')
    model2 = load_model('./models/run22-model.h5')
    model2.load_weights('./models/run22-model.h5')
    print("models imported...")
    CASCADE='./utils/haarcascade_frontalface_alt_tree.xml'
    FACE_CASCADE=cv2.CascadeClassifier(CASCADE)
    MIN_FACE_SIZE = 32
    PADDING = 32

    DATA_FILE = args.test_data_dir
    predictions = predict(model1,model2,DATA_FILE="data.h5",most_frequent_class = 4, new_size = 128)
    
#     datapath = os.path.join('data','data.h5')
#     if not os.path.exists(os.path.dirname(datapath)):
#         os.makedirs(os.path.dirname(datapath))
#     datafile = h5py.File(datapath, 'w')
#     datafile.create_dataset("Training_pixel", dtype = 'uint8', data=predictions[1,:])   
#     print(predictions[1,:])
 
    with open('predictions_train.pkl', 'wb') as f:
        pickle.dump(predictions[1,:], f)