__author__ = 'DK80'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork
from custom_layer import Layer

class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, actFun_type='relu', reg_lambda=0.01, seed=0):
        '''
        :param layer_sizes: List containing the sizes of each layer (including input and output layers)
        :param actFun_type: Activation function for the network
        :param reg_lambda: Regularization strength
        :param seed: Random seed for reproducibility
        '''
        np.random.seed(seed)
        self.layers = []
        self.reg_lambda = reg_lambda
        self.actFun_type = actFun_type
        
        # Create layers based on provided layer sizes
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i - 1], layer_sizes[i], actFun_type))

    def feedforward(self, X):
        '''
        Feedforward for all layers using Layer.feedforward
        :param X: input data
        :return: None
        '''
        a = X
        for layer in self.layers:
            a = layer.feedforward(a)
        # Final probabilities using softmax
        self.probs = np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)

    def backprop(self, X, y):
        '''
        Backpropagation through all layers using Layer.backprop
        :param X: input data
        :param y: true labels
        :return: gradients for weights and biases
        '''
        num_examples = len(X)
        delta = self.probs
        delta[range(num_examples), y] -= 1

        dW_list = []
        db_list = []
        delta_prev = delta

        # Backpropagation through each layer
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = X if i == 0 else self.layers[i - 1].a
            dW, db, delta_prev = layer.backprop(delta_prev, a_prev, self.reg_lambda)
            dW_list.append(dW)
            db_list.append(db)

        return dW_list[::-1], db_list[::-1]  # Reverse to match layer order

    def calculate_loss(self, X, y):
        '''
        Compute the loss function with L2 regularization
        :param X: input data
        :param y: true labels
        :return: total loss
        '''
        num_examples = len(X)
        self.feedforward(X)
        correct_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)

        # Add L2 regularization
        data_loss += self.reg_lambda / 2 * sum(np.sum(np.square(layer.W)) for layer in self.layers)
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        Train the model using gradient descent
        :param X: input data
        :param y: true labels
        :param epsilon: learning rate
        :param num_passes: number of training iterations
        :param print_loss: whether to print the loss or not
        '''
        for i in range(num_passes):
            # Forward propagation
            self.feedforward(X)

            # Backpropagation
            dW_list, db_list = self.backprop(X, y)

            # Gradient descent parameter update
            for j, layer in enumerate(self.layers):
                layer.W += -epsilon * dW_list[j]
                layer.b += -epsilon * db_list[j]

            # Optionally print the loss
            if print_loss and i % 1000 == 0:
                print(f"Loss after iteration {i}: {self.calculate_loss(X, y)}")

    def predict(self, X):
        '''
        Predict the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)