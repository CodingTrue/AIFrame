import numpy as np

from aiframe import NeuralNetwork
from aiframe.Criterion import BaseCriterion
from aiframe.Utils import mass_strip_list, mass_remove_list, mass_replace_list
from aiframe.program import TrainProgram
from aiframe.program.ProgramConstants import *
from aiframe.program.passes import BasePass, PassInfo, PassInfoParameter

import inspect

class DefaultProgramBuilderPass(BasePass):
    def get_pass_info(self, nn: NeuralNetwork) -> PassInfo:
        return PassInfo().add(target=PassInfoParameter(name="backward_values", value=np.zeros(np.max([neuron_count for neuron_count, _ in nn.get_network_structure()])))).finalize()

def format_source(source: list = [], replace_info: list[dict]|dict = []) -> list:
    if type(replace_info) == dict: replace_info = [replace_info]

    source = mass_strip_list(targets=source)
    source = mass_remove_list(targets=source, remove_info=["self."])
    for info in replace_info:
        source = mass_replace_list(targets=source, replace_info=info)

    return source

class ProgramBuilder():
    @staticmethod
    def create_train_program(nn: NeuralNetwork, criterion: BaseCriterion) -> TrainProgram:
        train_program = TrainProgram()
        train_program.add_pass(target=DefaultProgramBuilderPass())
        train_program._nn = nn

        network_nodes = criterion.preparse_nodes(nodes=nn._network_nodes)
        layer_count = nn.get_layer_count()

        train_program.set_active_group(name=FORWARD_PASS)
        for node in network_nodes:
            source = format_source(source=inspect.getsource(node.evaluate).split(':', 1)[1:], replace_info={
                "_output": CURRENT_LAYER_RESULT,
                "_input": LAST_LAYER_RESULT,
                "_weights": LAYER_WEIGHTS,
                "_biases": LAYER_BIASES,
                "np": "numpy"
            })
            train_program.add_lines(lines=source)

        train_program.set_active_group(name=BACKWARD_PASS)
        for node in network_nodes:
            source = format_source(source=inspect.getsource(node.backward).split(':', 1)[1:], replace_info=[
                {
                    "return": "return =",
                },
                {
                    "return": CURRENT_LAYER_RESULT,
                    "_input": LAST_LAYER_RESULT,
                    "np": "numpy"
                }
            ])
            train_program.add_lines(lines=source)
        return train_program