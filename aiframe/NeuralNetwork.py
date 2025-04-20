from aiframe.PropertyUtils import private
from aiframe.node.NodeLoader import *

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
        return self

    @private
    def network_nodes(self):
        return self._network_nodes

    @private
    def input_size(self):
        return self._input_size

    @private
    def nodeloader(self):
        return self._nodeloader