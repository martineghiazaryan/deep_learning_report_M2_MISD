import os
import cv2
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

img_dir = 'd:/Profils/myeghiazaryan/Downloads/train_v2/' # change the path to your training data folder

border = 5
im_chan = 3
n_classes = 2 

# Decoding the ship coordinates 
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


#This function preprocesses the image and mask data

def preprocess_data(img_ids, img_dir, df, train=True):

    X = np.zeros((len(img_ids), 256, 256, im_chan), dtype=np.uint8)  # changed dimensions here
    y = np.zeros((len(img_ids), 256, 256, n_classes), dtype=np.uint8)  # changed dimensions here
    for n, id_ in enumerate(img_ids):
        
        img_path = img_dir + id_
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))  
#         print(f"Resized image shape: {img.shape}")  # Print the dimensions of the resized image
        X[n] = img

        if train:

            mask = np.zeros((768, 768))
            masks = df.loc[df['ImageId'] == id_, 'EncodedPixels'].tolist()


            if masks[0] != masks[0]:

                pass
            else:
                for mask_ in masks:
                    mask += rle_decode(mask_)
            

            mask = cv2.resize(mask, (256, 256))  # added resizing here
#             print(f"Resized mask shape: {mask.shape}") 

            mask = np.expand_dims(mask, axis=-1)
#             print(f"Expanded mask shape: {mask.shape}") 
            mask_cat = to_categorical(mask, num_classes=n_classes)
    
#             print(f"Categorical mask shape: {mask_cat.shape}")
            y[n, ...] = mask_cat.squeeze()

    return X, y


# This function preprocesses the test image data
def preprocess_test_data(img_ids, img_dir):

    X = np.zeros((len(img_ids), 256, 256, im_chan), dtype=np.uint8)

    for n, id_ in enumerate(img_ids):
        img_path = os.path.join(img_dir, id_)
        if not os.path.exists(img_path):
            print(f"Image path does not exist: {img_path}")
            continue

        # Loading image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        img = cv2.resize(img, (256, 256))  
        X[n] = img

    return X
