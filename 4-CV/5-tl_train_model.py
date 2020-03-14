from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import joblib

# Load data set of extracted features
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

# Create a model and add layers
model = Sequential()

# Add layers to our model
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    validation_split=0.05,
    epochs=10,
    shuffle=True,
    verbose=2
)

# Save the trained model to a file so we can use it to make predictions later
model.save("bird_feature_classifier_model.h5")