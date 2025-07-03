from abc import ABC, abstractmethod

from aiframe import NeuralNetwork
from aiframe.program.passes import PassInfo, PassInfoParameter

def get_closest_keys(start: str, reference: dict) -> list[str]:
    result = []
    for key in reference.keys():
        if not key.startswith(start): continue
        result.append(key)

    return result

def select_by_group_path(path: str, groups: dict) -> any:
    keys = path.split("/")
    reference = groups

    for i in range(len(keys[:-1])):
        reference = reference.get(keys[i])

    if isinstance(reference.get(keys[-1]), dict): raise Exception(f"Target reference '{path}' can't be group!")
    return reference, get_closest_keys(start=keys[-1], reference=reference)

class PassInstruction(ABC):
    def __init__(self, target: str):
        self._target = target

    @abstractmethod
    def modify(self, groups: dict = {}, hard_pass: bool = False):
        return

class ReplacePassInstruction(PassInstruction):
    def __init__(self, target: str, find: str, replace_with: str):
        super().__init__(target=target)

        self._find = find
        self._replace_with = replace_with

    def modify(self, groups: dict = {}, hard_pass: bool = False):
        ref, keys = select_by_group_path(path=self._target, groups=groups)

        for key in keys:
            if not hard_pass and not ref.get(key): return
            ref[key] = ref[key].replace(self._find, self._replace_with)

class AppendPassInstruction(PassInstruction):
    def __init__(self, target: str, append_str: str):
        super().__init__(target=target)

        self._append_str = append_str

    def modify(self, groups: dict = {}, hard_pass: bool = False):
        ref, keys = select_by_group_path(path=self._target, groups=groups)

        for key in keys:
            if not hard_pass and not ref.get(key): return
            ref[key] += self._append_str

class RemovePassInstruction(PassInstruction):
    def __init__(self, target: str):
        super().__init__(target=target)

    def modify(self, groups: dict = {}, hard_pass: bool = False):
        ref, keys = select_by_group_path(path=self._target, groups=groups)

        for key in keys:
            if not hard_pass and not ref.get(key): return
            ref[key] = ""

class BasePass(ABC):
    def run_pass(self) -> list:
        return []

    @abstractmethod
    def get_pass_info(self, nn: NeuralNetwork) -> PassInfo:
        return