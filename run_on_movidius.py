try: from openvino.inference_engine import IECore, IENetwork
except ImportError: print('Make sure you activated setupvars.bat!')
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


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

# Set up the core -> Finding your device and loading the model
############################################################################################
ie = IECore()
device_list = ie.available_devices

# Load any network from file
model_xml = "IR_models/keras_iris_model.xml"
model_bin = "IR_models/keras_iris_model.bin"
net = IENetwork(model=model_xml, weights=model_bin)


# create some kind of blob
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

# Input Shape
print('Input Shape: ' + str(net.inputs[input_blob].shape))

# Run the model on the device
##############################################################################################
# load model to device
exec_net = ie.load_network(network=net, device_name='MYRIAD')

# execute the model and read the output
res = exec_net.infer(inputs={input_blob: X_test[:10,:]})
res = res[out_blob]
# The result is the softmax output corresponding to the probability distribution in one hot encoding
# Except for the pytorch model - here we don't do the softmax. The highest value corresponds to the class.
print(res)



