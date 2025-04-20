from aiframe.PropertyUtils import private
from aiframe.node.NodeLoader import *

import numpy as np

class NeuralNetwork():
    def __init__(self, network_nodes: list = [], input_size: int = -1, nodeloader: NodeLoader = BASIC_NODE_LOADER):
        self._network_nodes = network_nodes
        self._input_size = input_size
        self._nodeloader = nodeloader

    def init(self, random_weights: bool = False, random_biases: bool = False):
        last_input_size = self._input_size
        for node in self._network_nodes:
            if not self._nodeloader.is_layer(node.__class__): continue

            node._input_size = last_input_size
            last_input_size = node._neuron_count

            node._weights = np.random.uniform(-1, 1, (node._neuron_count, node._input_size)) if random_weights else np.zeros((node._neuron_count, node._input_size))
            node._biases = np.random.uniform(-1, 1, (node._neuron_count,)) if random_biases else np.zeros((node._neuron_count))
        return self

    def forward(self, input_data: np.ndarray):
        output_data = input_data
        for node in self._network_nodes:
            node._input = output_data
            node.evaluate()
            output_data = node._output
        return output_data

    @private
    def network_nodes(self):
        return self._network_nodes

    @private
    def input_size(self):
        return self._input_size

    @private
    def nodeloader(self):
        return self._nodeloader