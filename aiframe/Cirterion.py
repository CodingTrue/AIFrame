from abc import ABC, abstractmethod

from aiframe.Loss import BaseLoss, SquaredLoss, CrossEntropyLoss
from aiframe.node.Nodes import BaseNode, SoftmaxActivationNode, SigmoidActivationNode


class BaseCriterion(ABC):
    @abstractmethod
    def preparse_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        return nodes

    @abstractmethod
    def get_loss(self) -> BaseLoss:
        return

    @abstractmethod
    def add_node(self, index: int, node: BaseNode) -> bool:
        return

class SquaredLossCriterion(ABC):
    def preparse_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        return nodes

    def get_loss(self) -> BaseLoss:
        return SquaredLoss()

    def add_node(self, index: int, node: BaseNode) -> bool:
        return True

class CrossEntropyCriterion(ABC):
    def preparse_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        return nodes if nodes[-1].__class__ == SoftmaxActivationNode else [*nodes, SoftmaxActivationNode()]

    def get_loss(self) -> BaseLoss:
        return CrossEntropyLoss()

    def add_node(self, index: int, node: BaseNode) -> bool:
        return not (index == 0 and node.__class__ in [SoftmaxActivationNode, SigmoidActivationNode])