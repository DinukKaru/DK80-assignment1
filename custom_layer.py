__author__ = 'DK80'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        # Initialize weights and biases for this layer
        self.W = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation

    def actFun(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))

    def diff_actFun(self, z):
        if self.activation == "relu":
            return np.where(z > 0, 1, 0)
        elif self.activation == "tanh":
            return 1 - np.tanh(z)**2
        elif self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)

    def feedforward(self, X):
        # Feedforward step for the layer
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z)
        return self.a

    def backprop(self, delta, a_prev, reg_lambda):
        # Backpropagation for the layer
        dW = np.dot(a_prev.T, delta) + reg_lambda * self.W
        db = np.sum(delta, axis=0, keepdims=True)
        
        # Make sure the shapes match for delta_prev
        delta_prev = np.dot(delta, self.W.T) * self.diff_actFun(a_prev)  # Change from self.z to a_prev
        
        return dW, db, delta_prev

