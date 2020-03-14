from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import xception

image_to_test = "bird1.png"

# Load the model we trained
model = load_model('bird_feature_classifier_model.h5')

# Load image to test, resizing it to 73 pixels (as required by this model)
img = image.load_img(image_to_test, target_size=(73, 73))

# Convert the image to a numpy array
image_array = image.img_to_array(img)

# Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
images = np.expand_dims(image_array, axis=0)

# Normalize the data
images = xception.preprocess_input(images)

# Use the pre-trained neural network to extract features from our test image (the same way we did to train the model)
feature_extraction_model = xception.Xception(weights='imagenet', include_top=False, input_shape=(73, 73, 3))
features = feature_extraction_model.predict(images)

# Given the extracted features, make a final prediction using our own model
results = model.predict(features)

# Since we are only testing one image with possible class, we only need to check the first result's first element
single_result = results[0][0]

# Print the result
print(f"Likelihood that {image_to_test} is a bird: {single_result * 100}%")