from abc import ABC, abstractclassmethod

from aiframe.PropertyUtils import private

from aiframe.program.Program import Program
from aiframe.node import BaseNode, NodeLoader, BASIC_NODE_LOADER
from aiframe.Criterion import BaseCriterion

from aiframe.Utils import mass_replace, mass_strip
from aiframe.training.TrainingFunctions import calculate_layer_backward_values, update_gradient

import inspect

class ProgramBuilder(ABC):
    def __init__(self, criterion: BaseCriterion = None, nodeloader: NodeLoader = BASIC_NODE_LOADER):
        self._nodeloader = nodeloader
        self._code_definitions = {}
        self._function_definitions = {}
        self._import_definitions = []
        self._criterion = criterion

    def add_program_import(self, module_name: str, import_as: str = ""):
        self._import_definitions.append([f"import {module_name}{f' as {import_as}' if import_as else ''}"])

    def add_program_function_definition(self, function, function_name: str, strip_info: list = [], replace_info: dict = {}):
        sourcecode = inspect.getsource(function)
        sourcecode = mass_strip(sourcecode, strip_info)
        sourcecode = mass_replace(sourcecode, replace_info)

        sourcecode_definitions = []
        for i, line in enumerate(sourcecode.split('\n')):
            if not line: continue
            sourcecode_definitions.append(("\t" if i > 0 else "") + line.strip())

        self._function_definitions[function_name] = sourcecode_definitions

    def add_node_code_definitions(self, function, pass_name: str = "", strip_info: list = [], replace_info: dict = {}, override_definitions: dict = {}, add_definitions: dict = {}):
        for classname in self._nodeloader._classes:
            classinfo = self._nodeloader._classes[classname]
            path_name = f"{classname}_{pass_name}"

            sourcecode_definition = []

            if classname in override_definitions:
                sourcecode_definition = override_definitions[classname]
            else:
                sourcecode = inspect.getsource(function(classinfo["class"]))
                sourcecode = mass_strip(sourcecode, strip_info)
                sourcecode = mass_replace(sourcecode, replace_info)

                for line in sourcecode.split('\n')[1:]:
                    line = line.strip()

                    if not line: continue
                    sourcecode_definition.append(line)
            if classname in add_definitions:
                for definition in add_definitions[classname]:
                    sourcecode_definition.append(definition)
            self._code_definitions[path_name] = sourcecode_definition

    def get_node_definition(self, node: BaseNode, pass_direction: str = "") -> list:
        return self._code_definitions[f"{node.__class__.__name__}_{pass_direction}"] if self._nodeloader.is_node_registered(target=node) else []

    @abstractclassmethod
    def build_program_from_nodes(self):
        return


    @private
    def nodeloader(self):
        return self._nodeloader

    @private
    def code_definitions(self):
        return self._code_definitions

    @private
    def function_definitions(self):
        return self._function_definitions

    @private
    def import_definitions(self):
        return self._import_definitions

    @private
    def criterion(self):
        return self._criterion