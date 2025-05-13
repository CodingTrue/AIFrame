from abc import ABC, abstractmethod
import numpy as np

class BaseNode(ABC):
    def __init__(self):
        self._input = None
        self._output = None

    @abstractmethod
    def evaluate(self):
        return

class BaseActivationNode(BaseNode):
    @abstractmethod
    def backward(self):
        return

class HiddenLayerNode(BaseNode):
    def __init__(self, neuron_count: int = 0, input_size: int = 0):
        self._neuron_count = neuron_count
        self._input_size = input_size

        self._weights = None
        self._biases = None

    def evaluate(self):
        self._output = np.dot(self._input, self._weights.T) + self._biases

class ReluActivation(BaseActivationNode):
    def evaluate(self):
        self._output = np.max(self._input)

    def backward(self):
        return self._output > 0