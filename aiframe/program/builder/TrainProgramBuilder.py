from aiframe.program import Program
from aiframe.program.builder import ProgramBuilder
from aiframe.program.dispatch import DisptachInfo, DeviceType

from aiframe.node.NodeLoader import NodeLoader, BASIC_NODE_LOADER
from aiframe.node.Nodes import BaseNode

from aiframe.Criterion import BaseCriterion

from aiframe.Utils import mass_replace_all
from aiframe.training.TrainingFunctions import calculate_layer_backward_values, update_gradient

import numpy as np

class TrainProgramBuilder(ProgramBuilder):
    def __init__(self, criterion: BaseCriterion = None, nodeloader: NodeLoader = BASIC_NODE_LOADER):
        super().__init__(criterion, nodeloader)

        strip_info = [
            "self.",
            "_"
        ]
        replace_info = {
            "weights": "w_{layer_position}",
            "biases": "b_{layer_position}",
            "input": "pass_values",
            "output": "pass_values",
            "return": "pass_values =",
            "numpy": "np"
        }

        self.add_node_code_definitions(
            function=lambda x: x.evaluate,
            pass_name="forward",
            strip_info=strip_info,
            replace_info=replace_info,
            override_definitions={},
            add_definitions={
                layer: ["layer_output_{layer_position} = pass_values"] for layer, info in self._nodeloader._classes.items() if info["is_layer"]
            }
        )

        replace_info = {
            "weights": "w_{layer_position}",
            "biases": "b_{layer_position}",
            "input": "layer_output_{layer_position}",
            "return": "activation_deriv_{layer_position} =",
            "numpy": "np"
        }
        add_definitions = {
            layer: [
                "pass_values *= activation_deriv_{layer_position}",
                "gw_{layer_position}, gb_{layer_position} = update_gradient(pass_values, layer_output_{layer_position})"
            ]
            for layer, info in self._nodeloader._classes.items() if not info["is_layer"]
        }

        self.add_node_code_definitions(
            function=lambda x: x.backward,
            pass_name="backward",
            strip_info=strip_info,
            replace_info=replace_info,
            override_definitions={
                "HiddenLayerNode": ["pass_values = calculate_layer_backward_values(w_{layer_position}, pass_values, activation_deriv_{layer_position})"]
            },
            add_definitions=add_definitions
        )
        self.add_program_import(module_name="numpy", import_as="np")
        self.add_program_function_definition(function=calculate_layer_backward_values, function_name="calculate_layer_backward_values", strip_info=[], replace_info={"numpy": "np"})
        self.add_program_function_definition(function=update_gradient, function_name="update_gradient", strip_info=[], replace_info={"numpy": "np"})
        if self._criterion: self.add_program_function_definition(function=self._criterion.backward_loss, function_name="backward_loss", strip_info=["self, "], replace_info={"numpy": "np"})

    def build_program_from_nodes(self, nodes: list[BaseNode]) -> Program:
        train_program = Program(dispatch_info=DisptachInfo(run_device=DeviceType.CPU, can_be_parallelized=False))

        for import_definition in self._import_definitions:
            train_program.add_program_elements(elements=import_definition)

        for function_definition in self._function_definitions:
            train_program.add_program_elements(elements=self._function_definitions[function_definition])

        layers = [node for node in nodes if self._nodeloader.is_node_layer(node)]
        layer_count = len(layers)
        node_count = len(nodes)

        layer_position = 0
        for node in nodes:
            code_definition = self.get_node_definition(node=node, pass_direction="forward")

            code_definition = mass_replace_all(target_list=code_definition, info={
                "{layer_position}": str(layer_position)
            })
            train_program.add_program_elements(elements=code_definition)
            if self._nodeloader.is_node_layer(target=node): layer_position += 1

        train_program.add_program_element(element="pass_values = backward_loss(pass_values, expected)")

        layer_position = 0
        node_position = 0
        for node in nodes[::-1]:
            if node_position == node_count - 1: break

            code_definition = self.get_node_definition(node=node, pass_direction="backward")

            code_definition = mass_replace_all(target_list=code_definition, info={
                "{layer_position}": str(layer_count - layer_position - 1)
            })
            train_program.add_program_elements(elements=code_definition)
            if self._nodeloader.is_node_layer(target=node): layer_position += 1
            node_position += 1

        train_program.set_parameters(parameters={
            k: v for i, layer in enumerate(layers) for k, v in {
                f"w_{i}": layer.weights,
                f"b_{i}": layer.biases,
                f"gw_{i}": np.zeros((layer.neuron_count, layer.input_size)),
                f"gb_{i}": np.zeros((layer.neuron_count,))
            }.items()
        })
        return train_program