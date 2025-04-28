from abc import ABC, abstractmethod

class BaseCriterion(ABC):
    @abstractmethod
    def loss(self, predicted, expected):
        return

    @abstractmethod
    def backward_loss(self, predicted, expected):
        return

class SquaredLoss(BaseCriterion):
    def loss(self, predicted, expected):
        return (predicted - expected)**2

    def backward_loss(self, predicted, expected):
        return 2 * (predicted - expected)