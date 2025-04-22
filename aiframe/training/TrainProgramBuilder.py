from aiframe import private, mass_replace, mass_strip
from aiframe.training import TrainProgram
from aiframe.node import BaseNode, NodeLoader, BASIC_NODE_LOADER

import inspect

class TrainProgramBuilder():
    def __init__(self, nodeloader: NodeLoader = BASIC_NODE_LOADER):
        self._nodeloader = nodeloader
        self._code_definitions = {}

        strip_info = [
            "self."
        ]
        replace_info = {
            "_weights": "w_{layer_position}",
            "_biases": "b_{layer_position}",
            "_input": "pass_values",
            "_output": "pass_values",
            "return": "pass_values ="
        }

        self.add_code_definitions(
            function=lambda x: x.evaluate,
            pass_name="forward",
            strip_info=strip_info,
            replace_info=replace_info,
            override_definitions={}
        )

        self.add_code_definitions(
            function=lambda x: x.backward,
            pass_name="backward",
            strip_info=strip_info,
            replace_info=replace_info,
            override_definitions={
                "HiddenLayerNode": ["backward_values(w_{layer_position}, b_{layer_position})"]
            }
        )

    def add_code_definitions(self, function, pass_name: str = "", strip_info: list = [], replace_info: dict = {}, override_definitions: dict = {}):
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
            self._code_definitions[path_name] = sourcecode_definition

    def build_program_from_nodes(self, nodes: list[BaseNode]) -> TrainProgram:
        train_program = TrainProgram()

        layers = [node for node in nodes if self._nodeloader.is_layer(node.__class__)]
        layer_count = len(layers)

        layer_position = 0
        for node in nodes:
            nodeclass = node.__class__

            class_definitions = self._code_definitions[f"{nodeclass.__name__}_forward"]

            for line in class_definitions:
                train_program.add_program_element(element=line.replace("{layer_position}", f"{layer_position}"))
            if self._nodeloader.is_layer(nodeclass): layer_position += 1

        layer_position = 0
        for node in nodes[::-1]:
            nodeclass = node.__class__

            class_definitions = self._code_definitions[f"{nodeclass.__name__}_backward"]

            for line in class_definitions:
                train_program.add_program_element(element=line.replace("{layer_position}", f"{layer_count - layer_position - 1}"))
            if self._nodeloader.is_layer(nodeclass): layer_position += 1

        train_program.set_parameters(parameters={
            k: v for i, layer in enumerate(layers) for k, v in {
                f"w_{i}": layer.weights,
                f"b_{i}": layer.biases
            }.items()
        })
        return train_program

    @private
    def nodeloader(self):
        return self._nodeloader

    @private
    def code_definitions(self):
        return self._code_definitions