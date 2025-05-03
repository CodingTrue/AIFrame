from aiframe import NeuralNetwork
from aiframe.program import Program

import numpy as np

class Trainer():
    def train(self, nn: NeuralNetwork, train_program: Program):
        #print("\n".join(train_program.program_elements))

        train_program.assamble()

        # dispatch
        inputs = np.array([1, 2])
        expected = np.array([1, 0])

        parameters = {
            "expected": expected,
            "pass_values": inputs,
        }
        parameters.update(train_program.parameters)
        exec(train_program.program, parameters)

