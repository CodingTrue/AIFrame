from aiframe.PropertyUtils import private

class NeuralNetwork():
    def __init__(self, network_nodes: list = [], input_size: int = -1):
        self._network_nodes = network_nodes
        self._input_size = input_size

    def init(self, random_weights: bool = False, random_biases: bool = False):
        last_input_size = self._input_size
        for node in self._network_nodes:
            node._input_size = last_input_size
            last_input_size = node._neuron_count
        return self

    @private
    def network_nodes(self):
        return self._network_nodes

    @private
    def input_size(self):
        return self._input_size