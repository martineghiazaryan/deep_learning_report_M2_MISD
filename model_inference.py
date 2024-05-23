import cv2
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from data_preprocessing import preprocess_test_data

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def plot_predictions(predicted_segmentation, original_image):
    # Plot original image and predicted mask
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_segmentation[0, ..., 0], cmap='gray')  
    plt.title('Predicted Segmentation')
    plt.show()


def run_inference(test_img_dir, model_path):
    # Load the model
    model = load_trained_model(model_path)

        # Get the list of image names in the test directory
    test_ids = os.listdir(test_img_dir)

    random_test_img_names = random.sample(test_ids,  k=20)
    

    for img_name in random_test_img_names:
        img_path = os.path.join(test_img_dir, img_name)
        X_test = preprocess_test_data([img_path], test_img_dir)
        original_image = X_test[0]

        # Make predictions on the test image
        predicted_segmentation = model.predict(X_test)

        # Might apply threshold to convert probabilities to binary predictions
        # predicted_segmentation = (predicted_segmentation > 0.8).astype(np.uint8)

        # Plot the predictions
        plot_predictions(predicted_segmentation, original_image)


if __name__ == "__main__":
    # Define the path to your test images directory
    test_img_dir = 'd:/Profils/myeghiazaryan/Downloads/test_v2/'
    model_path = 'd:/Profils/myeghiazaryan/Desktop/airbus_case_study/models/model_best_checkpoint.h5' # the path to the model

    run_inference(test_img_dir, model_path)