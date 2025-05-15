import numpy as np

from aiframe import NeuralNetwork
from aiframe.program import Program

class TrainProgram(Program):
    def train(self, nn: NeuralNetwork, training_data: list, learn_rate: float = 0.05):
        train_function = self.get_function(function_name="train_function", globals={"np": np})

        train_function(inputs=np.array([1, 2]), expected=np.array([2, 2]))