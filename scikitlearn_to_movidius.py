import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import onnx
# installed: scikit-learn, skl2onnx, onnx, pandas

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

# Create the model
############################################################################################
mlp = MLPClassifier(hidden_layer_sizes=(50,50))

# Train the model
############################################################################################
mlp.fit(X_train, y_train)

score = mlp.score(X_test, y_test)
print('Test Accuracy: ', score)

# Save ONNX model
###########################################################################################
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(mlp, initial_types=initial_type)
onnx.save_model(onnx_model, 'sklearn_iris_model_before_pruning.onnx')

# Prune the ONNX model
###########################################################################################
onnx_model = onnx.load('sklearn_iris_model_before_pruning.onnx')
graph = onnx_model.graph
# Print a model overview
# print('The model is:\n{}'.format(onnx_model))

# Just remove all nodes after Identity
remove_list = []
end_reached = False
for x in graph.node:
    # print(x.name)
    if x.name == "Identity":
        end_reached = True
    if end_reached:
        remove_list.append(x)

for entry in remove_list:
    graph.node.remove(entry)
# Save the pruned model
onnx.save(onnx_model, 'sklearn_iris_model_after_pruning.onnx')