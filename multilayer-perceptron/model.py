import numpy as np
from typing import Callable
from imager import read_image_chunks, read_labels_chunks, RunType
import json

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
            # if i == len(neuron_counts) - 2:
            #     self.layers.append(Layer(neuron_count, neuron_counts[i+1], activation_fn, activation_fn_deriv))#lambda x: x, lambda x: np.ones_like(x)))
            # else:
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
    
    def answer_from_activation(self, activation: np.ndarray) -> np.intp:
        return np.argmax(activation)

    def save_model_to_file(self):
        model_json = {}
        with open('./model_config/model_dump.json', "w") as f:
            for i, layer in enumerate(self.layers):
                model_json[i] = {
                    "weights": layer.weights.tolist(),
                    "biases":  layer.biases.tolist()
                }
            json.dump(model_json, f, indent=3)

    def load_model_from_file(self, file_path: str):
        self.layers = []
        with open(file_path, "r") as f:
            model_json = json.load(f)
            for layer_rep in model_json:
                weights = model_json[layer_rep]["weights"]
                biases = model_json[layer_rep]["biases"]

                # the constructor should take the weights if offered
                layer = Layer(len(weights), len(biases), sigmoid, sigmoid_deriv)
                layer.weights = weights
                layer.biases= biases

                self.layers.append(layer)
                
        
def load_pixels_from_image():
    images = read_image_chunks(RunType.TRAIN)
    img_arrays = []
    for image in images[:1]:
        flattened_img = [pixel for row in image for pixel in row]
        img_arrays.append(np.array(flattened_img))
    return img_arrays

def loss_fn(activation: np.ndarray, y: np.ndarray):
    return sum((activation - y) ** 2)

def loss_deriv(activation: np.ndarray, y: np.ndarray):
    return 2 * (activation - y)

def main():
    epochs = 1
    images = load_pixels_from_image()
    labels = read_labels_chunks(RunType.TRAIN)

    model = MLPModel([784, 16, 16, 10], sigmoid, sigmoid_deriv)
    print(f"Started training for {epochs} epoch")
    total_loss = 0
    for _ in range(epochs):
        for (initial_activation, label) in zip(images, labels):
            print(f"Training on Label: {label}")

            output = model.forward(initial_activation)
            
            y = np.zeros(10)
            y[label] = 1.0
            loss = loss_fn(output, y)
            total_loss += loss

            output_gradient = loss_deriv(output, y)

            model.backprop(output_gradient)
            model.update(0.001)

    print(f"Training complete. Total Loss: {total_loss}")
    model.save_model_to_file()


def process(model: MLPModel, image: np.ndarray) -> np.intp:
    guess = model.forward(image)

    return model.answer_from_activation(guess)


if __name__ == "__main__":
    model = MLPModel([784, 16, 16, 10], sigmoid, sigmoid_deriv)
    model.load_model_from_file("./model_config/model_dump.json")

    image = load_pixels_from_image()

    print(process(model, image[0]))