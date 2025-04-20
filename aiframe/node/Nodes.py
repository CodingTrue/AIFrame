from abc import ABC, abstractmethod
from aiframe.PropertyUtils import private

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

class HiddenLayerNode(BaseNode):
    def __init__(self, neuron_count: int = 0, input_size: int = 0):
        self._neuron_count = neuron_count
        self._input_size = input_size

    def evaluate(self):
        ...

    def _info(self):
        return f"neuron_count={self._neuron_count} input_size={self._input_size}"

    @private
    def neuron_count(self):
        return self._neuron_count

    @private
    def input_size(self):
        return self._input_size