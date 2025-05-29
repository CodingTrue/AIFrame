from abc import ABC, abstractmethod
import numpy as np

class BaseNode(ABC):
    def __init__(self):
        self._input = None
        self._output = None

    @abstractmethod
    def evaluate(self):
        return

    @abstractmethod
    def backward():
        return

class HiddenLayerNode(BaseNode):
    def __init__(self, neuron_count: int = 0, input_size: int = 0):
        self._neuron_count = neuron_count
        self._input_size = input_size

        self._weights = None
        self._biases = None

    def evaluate(self):
        self._output = np.dot(self._input, self._weights) + self._biases

    def backward(self, activation, backward_values):
        return np.dot(backward_values, self._weights.T) * activation

class ReluActivationNode(BaseNode):
    def evaluate(self):
        self._output = np.maximum(self._input, 0)

    def backward(self):
        return (self._output > 0).astype(float)

class SoftmaxActivationNode(BaseNode):
    def evaluate(self):
        expos = np.exp(self._input - np.max(self._input, axis=-1, keepdims=True))
        self._output = expos / np.sum(expos, axis=-1, keepdims=True)

    def backward(self):
        expos = np.exp(self._input - np.max(self._input))
        s = expos / np.sum(expos, axis=-1, keepdims=True)
        return np.diag(s) - np.outer(s, s)