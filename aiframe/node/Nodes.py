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