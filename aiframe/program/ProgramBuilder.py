from aiframe import NeuralNetwork
from aiframe.Utils import mass_strip_list, mass_remove_list, mass_replace_list
from aiframe.program import Program

import inspect

class ProgramBuilder():
    @staticmethod
    def create_train_program(nn: NeuralNetwork):
        train_program = Program()

        layer_position = -1
        was_last_layer = False
        for node in nn._network_nodes:
            is_node_layer = nn._nodeloader.is_layer(target=node)
            if is_node_layer: layer_position += 1

            source = inspect.getsource(node.evaluate).split(":")[1:]
            source = mass_strip_list(targets=source)
            source = mass_remove_list(targets=source, remove_info=["self."])
            source = mass_replace_list(targets=source, replace_info={
                "_output": f"x_{layer_position}",
                "_weights": f"w_{layer_position}",
                "_biases": f"b_{layer_position}",
                "_input": "INPUTS" if layer_position == 0 and is_node_layer else f"x_{layer_position}"
            })

            source.insert(0, "#BEGIN_LAYER_TOKEN" if is_node_layer else "#BEGIN_TOKEN")
            source.append("#END_LAYER_TOKEN" if is_node_layer else "#END_TOKEN")

            train_program.add_lines(lines=source)

        print('\n'.join(train_program._program_lines))