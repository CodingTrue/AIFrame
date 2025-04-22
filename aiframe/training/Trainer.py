from aiframe import NeuralNetwork
from aiframe.training import TrainProgram

import numpy as np

class Trainer():
    def train(self, nn: NeuralNetwork, train_program: TrainProgram):
        train_program.assamble()

        # dispatch
        exec(train_program.program, train_program.parameters)
