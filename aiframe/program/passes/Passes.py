from abc import ABC, abstractmethod

from aiframe.program.passes import PassInfo, PassInfoParameter

class BasePass(ABC):
    @abstractmethod
    def get_pass_info(self) -> PassInfo:
        return