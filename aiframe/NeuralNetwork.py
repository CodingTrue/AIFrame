from aiframe.node import NodeLoader, BASIC_NODE_LOADER
from aiframe.node.Nodes import *

import numpy as np

class NeuralNetwork():
    def __init__(self, network_nodes: list[BaseNode] = [], max_input_size: int = 0, nodeloader: NodeLoader = BASIC_NODE_LOADER):
        self._network_nodes = network_nodes
        self._max_input_size = max_input_size
        self._nodeloader = nodeloader

    def allocate_input_dimensions(self, random: bool = True):
        input_size = self._max_input_size

        for node in self._network_nodes:
            if not self._nodeloader.is_layer(target=node): continue

            node._input_size = input_size
            input_size = node._neuron_count

            node._weights = np.random.uniform(-1, 1, (node._input_size, node._neuron_count)) if random else np.zeros((node._input_size, node._neuron_count))
            node._biases = np.random.uniform(-1, 1, (node._neuron_count,)) if random else np.zeros((node._neuron_count,))
        return self

    def get_network_structure(self) -> list[(int, int)]:
        return [(layer._input_size, layer._neuron_count) for layer in self._network_nodes if self._nodeloader.is_layer(target=layer)]

    def get_layer_count(self) -> int:
        return len([node for node in self._network_nodes if self._nodeloader.is_layer(target=node)])