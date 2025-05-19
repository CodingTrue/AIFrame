import numpy as np

from aiframe import NeuralNetwork
from aiframe.program import Program

class TrainProgram(Program):
    def train(self, nn: NeuralNetwork, training_data: list, learn_rate: float = 0.05):
        train_function = self.get_function(function_name="train_function", globals={"np": np})

        gradientW = [np.zeros(layer) for layer in nn.get_network_structure()]
        gradientB = [np.zeros(neuron_count) for _, neuron_count in nn.get_network_structure()]

        inputs = np.array([[1, 2, 1], [2, 2, 1]])
        expected = np.array([2, 2])

        train_function(inputs=inputs, expected=expected, gradientW=gradientW, gradientB=gradientB)