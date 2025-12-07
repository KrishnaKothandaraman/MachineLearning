import numpy as np
from typing import Callable

def sigmoid(inp: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-inp))

def sigmoid_deriv(inp: np.ndarray) -> np.ndarray:
    return sigmoid(inp) * (1 - sigmoid(inp))

def relu(inp: np.ndarray) -> np.ndarray:
    return np.maximum(0, inp)

def relu_deriv(inp: np.ndarray) -> np.ndarray:
    return (inp > 0).astype(float)

class Layer:
    def __init__(self, input_dim: int, output_dim: int, activation_fn: Callable[[np.ndarray], np.ndarray], activation_fn_deriv: Callable[[np.ndarray], np.ndarray]):
        """
        Docstring for __init__
        
        :param input_dim: Number of inputs to this layer
        :type input_dim: int
        :param output_dim: Number of outputs from this layer
        :type output_dim: int
        :param activation_fn: Function to squish activation to (0,1) Relu or Sigmoid
        """

        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.zeros(output_dim)
        self.activation_fn = activation_fn
        self.activation_fn_deriv = activation_fn_deriv
        # forward pass values
        self.input_activation = np.zeros(0)
        self.pre_activation = np.zeros(0)
        self.post_activation = np.zeros(0)

        # gradient placeholders
        self.dW = np.zeros(0)
        self.db = np.zeros(0)

    
    def forward(self, input_activation: np.ndarray) -> np.ndarray:
        self.input_activation = input_activation # a(l-1)

        self.pre_activation = np.dot(self.weights, input_activation) + self.biases # z

        self.post_activation = self.activation_fn(self.pre_activation) # a

        return self.post_activation    

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        "output gradient is the result of gradient calculation from the next layer having shape (output_dim,)"
        "Think of this bascically as a vector of desired `nudges` you want to make to the neurons in the current layer"

        dZ = output_gradient * self.activation_fn_deriv(self.pre_activation) # shape is (output_dim,)

        self.dW = np.outer(dZ, self.input_activation)
        self.db = dZ

        return self.weights.T @ dZ
    
    def update(self, learning_rate: float):
        self.weights -= learning_rate * self.dW
        self.biases  -= learning_rate * self.db

class MLPModel:
    
    def __init__(self, neuron_counts: list[int], activation_fn: Callable[[np.ndarray], np.ndarray], activation_fn_deriv: Callable[[np.ndarray], np.ndarray]):
        self.layers: list[Layer] = []
        for i, neuron_count in enumerate(neuron_counts[:-1]): # 784 16 16 10
            if i == len(neuron_counts) - 2:
                self.layers.append(Layer(neuron_count, neuron_counts[i+1], lambda x: x, lambda x: np.ones_like(x)))
            else:
                self.layers.append(Layer(neuron_count, neuron_counts[i+1], activation_fn, activation_fn_deriv))
   
    def forward(self, activation: np.ndarray):

        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backprop(self, gradient):

        for layer in self.layers[-1::-1]:
            gradient = layer.backward(gradient)
        return gradient
    
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
            


def main():
    print("Hello world!!")

if __name__ == "__main__":
    main()