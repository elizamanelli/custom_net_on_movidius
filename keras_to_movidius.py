import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
import keras2onnx
import onnx
# installed: scikit-learn, keras2onnx, onnx, keras, pandas

# Prepare the dataset
############################################################################################
# Load the iris dataset
iris = load_iris()

# Create X and y data
X = iris.data
y = iris.target
# make targets one hot encoded
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Model Definition
###########################################################################################
model = Sequential(name='iris_keras')
# Add the input layer
model.add(Dense(50, activation='relu', input_shape=(4,)))
# A second hidden layer
model.add(Dense(50, activation='relu'))
# And the output layer - nodes: 4, 50, 50, 3
model.add(Dense(3, activation='softmax'))
# Add the loss, function, optimizer and accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Training
###########################################################################################
model.fit(X_train, y_train, batch_size=5, epochs=50, verbose=0, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model as ONNX
###########################################################################################
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, 'keras_iris_model.onnx')

