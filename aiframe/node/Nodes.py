from abc import ABC, abstractmethod
from aiframe.PropertyUtils import private

import numpy as np

class BaseNode(ABC):
    def __init__(self):
        self._input = None
        self._output = None

    @abstractmethod
    def evaluate(self):
        return

    def _info(self):
        return id(self)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._info()})"

    @private
    def input(self):
        return self._input

    @private
    def output(self):
        return self._output

class ActivationNode(BaseNode):
    @abstractmethod
    def backward(self):
        return

class HiddenLayerNode(BaseNode):
    def __init__(self, neuron_count: int = 0, input_size: int = 0):
        super().__init__()

        self._neuron_count = neuron_count
        self._input_size = input_size
        self._weights = None
        self._biases = None

    def evaluate(self):
        self._output = np.add(self._biases, np.dot(self._input, self._weights.T))

    def _info(self):
        return f"neuron_count={self._neuron_count} input_size={self._input_size}"

    @private
    def neuron_count(self):
        return self._neuron_count

    @private
    def input_size(self):
        return self._input_size

    @private
    def weights(self):
        return self._weights

    @private
    def biases(self):
        return self._biases

class ReLUNode(ActivationNode):
    def evaluate(self):
        self._output = np.maximum(0, self._input)

    def backward(self):
        return (self._input > 0).astype(int)

class TanHNode(ActivationNode):
    def evaluate(self):
        self._output = np.tanh(self._input)

    def backward(self):
        return 1 - np.square(np.tanh(self._input))

class SigmoidNode(ActivationNode):
    def evaluate(self):
        self._output = 1 / (1 + np.exp(-self._input))

    def backward(self):
        return (1 / (1 + np.exp(-self._input))) * (1 - 1 / (1 + np.exp(-self._input)))