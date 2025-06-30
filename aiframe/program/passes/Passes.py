from abc import ABC, abstractmethod

from aiframe import NeuralNetwork
from aiframe.program.passes import PassInfo, PassInfoParameter

class BasePass(ABC):
    def run_pass(self):
        return

    @abstractmethod
    def get_pass_info(self, nn: NeuralNetwork) -> PassInfo:
        return