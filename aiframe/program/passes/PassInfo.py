from typing import Self

class PassInfoParameter():
    def __init__(self, name: str, value: any, is_argument: bool = False):
        self._name = name
        self._value = value
        self._is_argument = is_argument

class PassInfo():
    def __init__(self, for_key: str = ""):
        self._parameters = []
        self._is_finalized = False

    def add(self, target: PassInfoParameter) -> Self:
        if self._is_finalized: raise Exception(f"PassInfo '{self}' is already finalized!")

        self._parameters.append(target)

        return self

    def finalize(self) -> Self:
        self._is_finalized = True
        return self