from aiframe.node import NodeLoader, BASIC_NODE_LOADER
from aiframe.node.Nodes import *

class NeuralNetwork():
    def __init__(self, network_nodes: list[BaseNode] = [], max_input_size: int = 0, nodeloader: NodeLoader = BASIC_NODE_LOADER):
        self._network_nodes = network_nodes
        self._nodeloader = nodeloader