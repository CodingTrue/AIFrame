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

        for node in self.get_layers():
            node._input_size = input_size
            input_size = node._neuron_count

            node._weights = np.random.uniform(-1, 1, (node._input_size, node._neuron_count)) if random else np.zeros((node._input_size, node._neuron_count))
            node._biases = np.random.uniform(-1, 1, (node._neuron_count,)) if random else np.zeros((node._neuron_count,))
        return self

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        out = input_data
        for node in self._network_nodes:
            node._input = out
            node.evaluate()
            out = node._output
        return out

    def get_layer_at_index(self, index: int) -> BaseNode:
        return self.get_layers()[index]

    def get_weights(self) -> np.ndarray:
        return [node._weights for node in self.get_layers()]

    def get_biases(self) -> np.ndarray:
        return [node._biases for node in self.get_layers()]

    def get_network_structure(self) -> list[(int, int)]:
        return [(layer._input_size, layer._neuron_count) for layer in self._network_nodes if self._nodeloader.is_layer(target=layer)]

    def get_layers(self) -> np.ndarray:
        return [node for node in self._network_nodes if self._nodeloader.is_layer(target=node)]

    def get_layer_count(self) -> int:
        return len(self.get_layers())