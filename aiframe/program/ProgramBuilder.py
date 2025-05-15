from aiframe import NeuralNetwork
from aiframe.Utils import mass_strip_list, mass_remove_list, mass_replace_list
from aiframe.program import TrainProgram

import inspect


class ProgramBuilder():
    @staticmethod
    def create_train_program(nn: NeuralNetwork) -> TrainProgram:
        train_program = TrainProgram()

        layer_count = nn.get_layer_count()

        layer_position = 0
        offset = 0
        was_last_layer = False
        for i, node in enumerate(nn._network_nodes):
            is_node_layer = nn._nodeloader.is_layer(target=node)

            if (not was_last_layer and is_node_layer) and i > 0: offset += 1

            source = inspect.getsource(node.evaluate).split(":")[1:]
            source = mass_strip_list(targets=source)
            source = mass_remove_list(targets=source, remove_info=["self."])
            source = mass_replace_list(targets=source, replace_info={
                "_output": f"x_{offset}",
                "_weights": f"w_{layer_position}",
                "_biases": f"b_{layer_position}",
                "_input": "inputs" if i == 0 else f"x_{offset - is_node_layer}"
            })

            if is_node_layer: layer_position += 1
            was_last_layer = is_node_layer

            train_program.add_lines(lines=source)

        layer_position -= 1
        train_program.add_line(line=f"cost_values = (x_{layer_position} - expected)**2")

        for i, node in enumerate(nn._network_nodes[::-1]):
            is_node_layer = nn._nodeloader.is_layer(target=node)
            if is_node_layer: layer_position -= 1

            source = []
            if is_node_layer:
                if layer_position + 2 == layer_count: continue

                source = mass_replace_list(targets=[
                    f"backward_values = np.sum((w_{layer_position + 1} * z_{layer_position + 1}) * backward_values[:, None], axis=1)"
                ], replace_info={})
            else:
                source = inspect.getsource(node.backward).split(":")[1:]
                source = mass_strip_list(targets=source)
                source = mass_remove_list(targets=source, remove_info=["self."])
                source = mass_replace_list(targets=source, replace_info={
                    "return": f"z_{layer_position} =",
                    "_output": f"x_{layer_position}",
                    "_input": "INPUTS" if layer_position == 0 and is_node_layer else f"x_{layer_position}"
                })
            if i == 0:
                source.append(f"backward_values = cost_values * z_{layer_position}")
            train_program.add_lines(lines=source)

        layer_map = [node for node in nn._network_nodes if nn._nodeloader.is_layer(target=node)]
        parameters = {
            k: v for i in range(layer_count) for k, v in {
                f"w_{i}": layer_map[i]._weights,
                f"b_{i}": layer_map[i]._biases,
            }.items()
        }
        train_program.set_parameters(parameters=parameters)

        train_program._program_lines = [f"\t{line}" for line in train_program._program_lines]
        train_program._program_lines.insert(0, "def train_function(inputs, expected):")
        return train_program