import numpy as np
import pandas as pd
import model_creation as mc
import model_training as mt
from model_training import dice_coef
import model_inference as mi
import data_preprocessing as dp
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


def main(test_img_dir, model_path):
    # Load the dataset
    df = pd.read_csv('d:/Profils/myeghiazaryan/Desktop/train_ship_segmentations_v2.csv') # Change the path to the datafarme
    
    # Filter the dataset to keep the first 5000 images
    df = df.head(5000)
    print(df)

    # Splitting the dataset into ships and no_ships dataframes
    ships_df = df[df['EncodedPixels'].notnull()]
    no_ships_df = df[df['EncodedPixels'].isnull()]
    
    # Splitting the ships and no_ships datasets into training and validation datasets
    train_ships_df, valid_ships_df = train_test_split(ships_df, test_size=0.2, random_state=42)
    train_no_ships_df, valid_no_ships_df = train_test_split(no_ships_df, test_size=0.2, random_state=42)

    # Concatenating ships and no_ships dataframes
    train_df = pd.concat([train_ships_df, train_no_ships_df])
    valid_df = pd.concat([valid_ships_df, valid_no_ships_df])


    # Saving the dataframe for using it in the model_training.py file 

    train_df.to_csv('train_df.csv', index=False)
    valid_df.to_csv('valid_df.csv', index=False)

    
    # Loading the model
    model = load_model(model_path, custom_objects={'dice_coef': dice_coef})

    # You can continue the training process using the loaded model if needed. So simply uncomment the line below.

    # Train the model
    # trained_model, history = mt.train_model(model)

    # If you want to train a new model, simply call mt.train_model() without any arguments.
    
    # Run for the model inferences 
    mi.run_inference(test_img_dir, model_path)

if __name__ == '__main__':

    test_img_dir = 'd:/Profils/myeghiazaryan/Downloads/test_v2/'  # The path to your test images directory
    model_path = r'D:\Profils\myeghiazaryan\Desktop\airbus_case_study\models\model_best_checkpoint.h5' # The path to your trained model
    
    main(test_img_dir, model_path)
