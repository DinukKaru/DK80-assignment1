import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from DeepNeuralNetwork import DeepNeuralNetwork

def generate_moons_data():
    '''
    Generate the Make-Moons dataset
    :return: input data X, labels y
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def generate_other_data():
    '''
    Generate another dataset from sklearn (e.g., circles dataset)
    :return: input data X, labels y
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(200, noise=0.20, factor=0.5)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    Plot the decision boundary for a given model
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: labels
    '''
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def main():
    # Example 1: Train on Make Moons Dataset
    X, y = generate_moons_data()
    
    # Define some different configurations for the deep neural network
    configs = [
        {"layer_sizes": [2, 5, 2], "actFun_type": "relu"},
        {"layer_sizes": [2, 10, 5, 2], "actFun_type": "tanh"},
        {"layer_sizes": [2, 20, 10, 5, 2], "actFun_type": "sigmoid"}
    ]
    
    for config in configs:
        print(f"Training network with config: {config}")
        model = DeepNeuralNetwork(layer_sizes=config["layer_sizes"], actFun_type=config["actFun_type"])
        model.fit_model(X, y, epsilon=0.01, num_passes=10000, print_loss=True)
        plot_decision_boundary(lambda x: model.predict(x), X, y)
    
    # Example 2: Train on another dataset (circles dataset)
    X2, y2 = generate_other_data()
    
    # Try a different configuration for this dataset
    print("Training network on a different dataset (circles)")
    model = DeepNeuralNetwork(layer_sizes=[2, 10, 5, 2], actFun_type='relu')
    model.fit_model(X2, y2, epsilon=0.01, num_passes=10000, print_loss=True)
    plot_decision_boundary(lambda x: model.predict(x), X2, y2)

if __name__ == "__main__":
    main()
