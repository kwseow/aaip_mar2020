import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


# Empty lists to hold the images and labels for each each image
x_test = []
y_test = []

# Load the test data set by looping over every image file
for image_file in Path("test_dataset").glob("**/*.png"):

    # Load the current image file
    image_data = image.load_img(image_file, target_size=(73, 73))

    # Convert the loaded image file to a numpy array
    image_array = image.img_to_array(image_data)

    # Add the current image to our list of test images
    x_test.append(image_array)

    # Add an expected label for this image. If it was a not_bird image, label it 0. If it was a bird, label it 1.
    if "not_bird" in image_file.stem:
        y_test.append(0)
    else:
        y_test.append(1)

# Convert the list of test images to a numpy array
x_test = np.array(x_test)

# Normalize test data set to 0-to-1 range
x_test = xception.preprocess_input(x_test)

# Load the Xception neural network to use as a feature extractor
feature_extractor = xception.Xception(weights='imagenet', include_top=False, input_shape=(73, 73, 3))

# Load our trained classifier model
model = load_model("bird_feature_classifier_model.h5")

# Extract features for each image (all in one pass)
features_x = feature_extractor.predict(x_test)

# Given the extracted features, make a final prediction using our own model
predictions = model.predict(features_x)

# If the model is more than 50% sure the object is a bird, call it a bird.
# Otherwise, call it "not a bird".
predictions = predictions > 0.5

# Calculate how many mis-classifications the model makes
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

# Calculate Precision and Recall for each class
report = classification_report(y_test, predictions)
print(report)