
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

# Represents a hidden or output layer in our neural network.
class Layer:
    """
    no_inputs : size of input from previous layer
    no_neurons : number of neurons in current layer
    weights : weights of current layer
    """

    def __init__(self, no_inputs, no_neurons, weights = None, bias = None):
        self.weights = weights if weights is not None else np.random.rand(no_inputs, no_neurons)
        self.bias = bias if bias is not None else np.random.rand(no_neurons)


    # Calculates the dot product of current layer.
    def activate(self, x):
        y = np.dot(x, self.weights) + self.bias
        self.last_activation = self.apply_activation(y)
        return self.last_activation


    # Applies the sigmoid activation function
    def apply_activation(self, x):
        return 1 / (1 + np.exp(-x))

# Represents a neural network.
class NeuralNetwork:
    def __init__(self):
        # List of layers in this neural network
        self.layers = []


    # Adds a layer to the neural network.
    def add_layer(self, layer):
        self.layers.append(layer)


    #Feed forward the input through the layers.
    def feed_forward(self, X):
        """
        :param X: The input values.
        :return: The result.
        """

        for layer in self.layers:
            X = layer.activate(X)

        return X


    # Predicts a class or classes.
    def predict(self, X):

        ff = self.feed_forward(X)
        # One row

        if ff.ndim == 1:
            return np.argmax(ff)

        # Multiple rows

        return np.argmax(ff, axis=1)

nn = NeuralNetwork()
nn.add_layer(Layer(2, 3))
nn.add_layer(Layer(3, 3))
nn.add_layer(Layer(3, 2))
nn.predict([[0, 0], [0, 1], [1, 0], [1, 1]])
