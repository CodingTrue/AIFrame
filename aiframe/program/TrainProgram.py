import numpy as np

from aiframe import NeuralNetwork
from aiframe.program import Program

class TrainProgram(Program):
    def train(self, nn: NeuralNetwork, training_data: list, learn_rate: float = 0.05, iterations: int = 1):
        train_function = self.get_function(function_name="train_function", globals={"np": np})

        weights = nn.get_weights()
        biases = nn.get_biases()
        layer_count = nn.get_layer_count()

        gradientW = [np.zeros(layer) for layer in nn.get_network_structure()]
        gradientB = [np.zeros(neuron_count) for _, neuron_count in nn.get_network_structure()]

        for i in range(iterations):
            train_function(inputs=training_data[0], expected=training_data[1], gradientW=gradientW, gradientB=gradientB)

            for i in range(layer_count):
                weights[i] -= gradientW[i] * learn_rate
                biases[i] -= biases[i] * learn_rate

        for i in range(layer_count):
            layer = nn.get_layer_at_index(index=i)
            layer._weights = weights[i]
            layer._biases = biases[i]