import numpy as np

from aiframe import NeuralNetwork
from aiframe.program import Program

class TrainProgram(Program):
    def __init__(self):
        super().__init__(nn=None)
        self._function_name = "train_function"

    def train(self, nn: NeuralNetwork, training_data: list, learn_rate: float = 0.05, iterations: int = 1, learn_rate_decay: float = 1):
        train_function = self.get_function(function_name="train_function", globals={"np": np})

        weights = nn.get_weights()
        biases = nn.get_biases()
        layer_count = nn.get_layer_count()

        gradientW = [np.zeros(layer) for layer in nn.get_network_structure()]
        gradientB = [np.zeros(neuron_count) for _, neuron_count in nn.get_network_structure()]

        batch_count = training_data[0].shape[0]
        batch_size = training_data[0].shape[1]

        inputs = training_data[0]
        expected = training_data[1]

        lr = learn_rate
        for e in range(iterations):
            for idx in np.random.permutation(batch_count):
                train_function(inputs=inputs[idx], expected=expected[idx], gradientW=gradientW, gradientB=gradientB)

                for i in range(layer_count):
                    weights[i] -= gradientW[i] / batch_size * lr
                    biases[i] -= gradientB[i] / batch_size * lr
            lr = learn_rate * (learn_rate_decay**e)

        for i in range(layer_count):
            layer = nn.get_layer_at_index(index=i)
            layer._weights = weights[i]
            layer._biases = biases[i]