from typing import Self

class PassInfoParameter():
    def __init__(self, name: str, value: any, is_argument: bool = False):
        self._name = name
        self._value = value
        self._is_argument = is_argument

class PassInfo():
    def __init__(self):
        self._parameters = []

    def add(self, target: PassInfoParameter) -> Self:
        self._parameters.append(target)
        return self