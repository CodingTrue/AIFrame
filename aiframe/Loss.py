from abc import ABC, abstractmethod

import numpy as np

class BaseLoss(ABC):
    @abstractmethod
    def loss(self, predicted, expected, *args, **kwargs) -> float:
        return

    @abstractmethod
    def loss_derivative(self, predicted, expected, *args, **kwargs):
        return

class SquaredLoss(ABC):
    def loss(self, predicted, expected, *args, **kwargs) -> float:
        return ((predicted - expected) ** 2).mean()

    def loss_derivative(self, predicted, expected, *args, **kwargs):
        return 2 * (predicted - expected)

class CrossEntropyLoss(ABC):
    def loss(self, predicted, *args, **kwargs) -> float:
        return np.sum(-np.log(predicted + 0e09))

    def loss_derivative(self, predicted, expected, *args, **kwargs):
        return predicted - expected