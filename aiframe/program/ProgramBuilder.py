import numpy as np

from aiframe import NeuralNetwork
from aiframe.Criterion import BaseCriterion
from aiframe.Utils import mass_strip_list, mass_remove_list, mass_replace_list
from aiframe.node.Nodes import BaseNode
from aiframe.program import TrainProgram
from aiframe.program.ProgramConstants import *
from aiframe.program.passes import BasePass, PassInfo, PassInfoParameter

from uuid import uuid4

import inspect
import textwrap
import ast

FORWARD_DESCRIPTOR_NAMES = {
    "HiddenLayerNode": {
        "line_index + 1 == source_len": "f'layer_forward_{node_index}'"
    },
    "ReluActivationNode": {
        "line_index + 1 == source_len": "f'relu_{node_index}'"
    },
    "SigmoidActivationNode": {
        "line_index + 1 == source_len": "f'sigmoid_{node_index}'"
    },
    "TanHActivationNode": {
        "line_index + 1 == source_len": "f'tanh_{node_index}'"
    },
    "SoftmaxActivationNode": {
        "line_index + 1 == source_len": "f'softmax_{node_index}'"
    }
}

BACKWARD_DESCRIPTOR_NAMES = {
    "HiddenLayerNode": {
        "line_index + 1 == source_len": "f'layer_backward_{node_index}'"
    },
    "ReluActivationNode": {
        "line_index + 1 == source_len": "f'relu_derivative_{node_index}'"
    },
    "SigmoidActivationNode": {
        "line_index + 1 == source_len": "f'sigmoid_derivative_{node_index}'"
    },
    "TanHActivationNode": {
        "line_index + 1 == source_len": "f'tanh_derivative_{node_index}'"
    },
    "SoftmaxActivationNode": {
        "line_index + 1 == source_len": "f'softmax_derivative_{node_index}'"
    }
}

def get_descriptors(source: list[str], descriptor_names: dict, set_locals: dict = {}) -> list[str]:
    descriptors = []
    node, node_index = set_locals.get("node"), set_locals.get("node_index")

    for line_index in range(len(source)):
        set_locals.update(locals())

        definition = descriptor_names.get(node.__class__.__name__)
        if definition:
            for expression, value in definition.items():
                descriptors.append(eval(value, {}, set_locals) if eval(expression, {}, set_locals) else NO_DEFINITION + f"_{str(uuid4())}")
        else:
            descriptors.append(UNDEFINED_DESCRIPTOR + f"_{str(uuid4())}")
    return descriptors

class DefaultProgramBuilderPass(BasePass):
    def get_pass_info(self, nn: NeuralNetwork) -> PassInfo:
        return PassInfo().add(target=PassInfoParameter(name="backward_values", value=np.zeros(np.max([neuron_count for neuron_count, _ in nn.get_network_structure()])))).finalize()

def format_source(source, replace_info: list[dict]|dict = []) -> list:
    source_string = textwrap.dedent(inspect.getsource(source))
    source_lines = source_string.splitlines()

    body_ast = ast.parse(source_string)
    line_start, lined_end = body_ast.body[0].lineno, body_ast.body[0].end_lineno

    body_lines = source_lines[line_start:lined_end]

    if type(replace_info) == dict: replace_info = [replace_info]

    body_lines = mass_strip_list(targets=body_lines)
    body_lines = mass_remove_list(targets=body_lines, remove_info=["self."])
    for info in replace_info:
        body_lines = mass_replace_list(targets=body_lines, replace_info=info)

    return body_lines

class ProgramBuilder():
    @staticmethod
    def create_train_program(nn: NeuralNetwork, criterion: BaseCriterion) -> TrainProgram:
        train_program = TrainProgram()
        train_program.add_pass(target=DefaultProgramBuilderPass())
        train_program._nn = nn

        network_nodes = criterion.preparse_nodes(nodes=nn._network_nodes)
        node_count = len(network_nodes)
        layer_count = nn.get_layer_count()

        train_program.set_active_group(name=FORWARD_PASS)
        for node_index, node in enumerate(network_nodes):
            source = format_source(source=node.evaluate, replace_info={
                "_output": CURRENT_LAYER_RESULT,
                "_input": LAST_LAYER_RESULT,
                "_weights": LAYER_WEIGHTS,
                "_biases": LAYER_BIASES,
                "np": "numpy"
            })
            source_len = len(source)

            descriptors = get_descriptors(source=source, descriptor_names=FORWARD_DESCRIPTOR_NAMES, set_locals=locals())
            train_program.add_lines(lines=source, descriptors=descriptors)

        train_program.set_active_group(name=BACKWARD_PASS)
        for node_index, node in enumerate(network_nodes[::-1]):
            source = format_source(source=node.backward, replace_info=[
                {
                    "return": "return =",
                },
                {
                    "return": CURRENT_LAYER_RESULT,
                    "_input": LAST_LAYER_RESULT,
                    "np": "numpy"
                }
            ])
            source_len = len(source)

            descriptors = get_descriptors(source=source, descriptor_names=FORWARD_DESCRIPTOR_NAMES, set_locals=locals())
            train_program.add_lines(lines=source, descriptors=descriptors)
        return train_program