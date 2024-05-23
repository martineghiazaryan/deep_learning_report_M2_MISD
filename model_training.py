import os
import numpy as np
import tensorflow as tf
import pandas as pd
from keras import backend as K
from model_creation import build_unet
from data_preprocessing import preprocess_data
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth=1e-7):
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return dice

def train_model(model=None):
    img_dir = 'd:/Profils/myeghiazaryan/Downloads/train_v2/'

    # Importing the train and validation dataframes

    train_df = pd.read_csv('train_df.csv')
    valid_df = pd.read_csv('valid_df.csv')

    train_ids = train_df['ImageId'].values
    valid_ids = valid_df['ImageId'].values


    if os.path.exists('X_train.npy') and os.path.exists('y_train.npy'):
        print("Loading data...")
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')

    else:
        print("Preprocessing data...")
        X_train, y_train = preprocess_data(train_ids, img_dir, train_df)

        print("Saving data...")
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)

    if os.path.exists('X_valid.npy') and os.path.exists('y_valid.npy'):
        print("Loading data...")
        X_valid = np.load('X_valid.npy')
        y_valid = np.load('y_valid.npy')
    else:
        print("Preprocessing data...")
        X_valid, y_valid = preprocess_data(valid_ids, img_dir, valid_df)

        print("Saving data...")
        np.save('X_valid.npy', X_valid)
        np.save('y_valid.npy', y_valid)

    total_ship_pixels = np.sum(y_train[:, :, :, 1])
    total_background_pixels = np.sum(y_train[:, :, :, 0])

    class_weight = {0: 1.0, 1: total_background_pixels / total_ship_pixels}

    print("Class Weights:")
    print(class_weight)


    input_shape = (256, 256, 3)
    n_classes = 2  # number of classes 


    # Create a new model if none is provided
    if model is None:
        model = build_unet(input_shape, n_classes)


    # opt = Adadelta(learning_rate=0.01, rho=0.95)
    # adadelta optimizer rho = 0.95 

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])


    # model.compile(optimizer='adam', loss=lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, ship_weight),  metrics=[iou_coef])

    callbacks = [
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001),
        ModelCheckpoint('model_best_checkpoint.h5', verbose=1, save_best_only=True)
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_valid, y_valid),
                        batch_size=16,
                        epochs=20,
                        callbacks=callbacks)

    return model, history