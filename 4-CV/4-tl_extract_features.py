from pathlib import Path
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception

# Empty lists to hold the images and labels for each each image
x_train = []
y_train = []

# Load the training data set by looping over every image file
for image_file in Path("training_dataset").glob("**/*.png"):

    # Load the current image file
    image_data = image.load_img(image_file, target_size=(73, 73))
    # Convert the loaded image file to a numpy array
    image_array = image.img_to_array(image_data)
    print("%s %s"%(image_file,image_array.shape))

    # Add the current image to our list of training images
    x_train.append(image_array)

    # Add a label for this image. If it was a not_bird image, label it 0. If it was a bird, label it 1.
    if "not_bird" in image_file.stem:
        y_train.append(0)
    else:
        y_train.append(1)

# Convert the list of separate images into a single 4D numpy array. This is what Keras expects.
x_train = np.array(x_train)

# Normalize image data to 0-to-1 range
x_train = xception.preprocess_input(x_train)

# Load the pre-trained neural network to use as a feature extractor
feature_extractor = xception.Xception(weights='imagenet', include_top=False, input_shape=(73, 73, 3))

# Extract features for each image (all in one pass)
features_x = feature_extractor.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Convert the list a numpy array. TensorFlow 2.0 doesn't like Python lists.
y_train = np.array(y_train)

# Save the matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")
