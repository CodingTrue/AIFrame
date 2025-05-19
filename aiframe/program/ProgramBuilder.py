from aiframe import NeuralNetwork
from aiframe.Utils import mass_strip_list, mass_remove_list, mass_replace_list
from aiframe.program import TrainProgram

import inspect


class ProgramBuilder():
    @staticmethod
    def create_train_program(nn: NeuralNetwork) -> TrainProgram:
        train_program = TrainProgram()
        layer_count = nn.get_layer_count()

        train_program.add_line(line="def train_function(inputs, expected, gradientW, gradientB):")

        layer_position = 0
        offset = 0
        was_last_layer = False
        for i, node in enumerate(nn._network_nodes):
            is_node_layer = nn._nodeloader.is_layer(target=node)

            if (not was_last_layer and is_node_layer) and i > 0: offset += 1

            source = inspect.getsource(node.evaluate).split(":", 1)[1:]
            source = mass_strip_list(targets=source)
            source = mass_remove_list(targets=source, remove_info=["self."])
            source = mass_replace_list(targets=source, replace_info={
                "_output": f"x_{offset}",
                "_weights": f"w_{layer_position}",
                "_biases": f"b_{layer_position}",
                "_input": "inputs" if i == 0 else f"x_{offset - is_node_layer}"
            })

            if is_node_layer:
                source.append(f"layer_cache_{layer_position} = x_{offset}")
                layer_position += 1
            was_last_layer = is_node_layer

            train_program.add_lines(lines=source, prefix="\t")

        layer_position = layer_count - 1
        train_program.add_line(line=f"cost_values = 2 * (x_{layer_position} - expected)", prefix="\t")

        for i, node in enumerate(nn._network_nodes[::-1]):
            is_node_layer = nn._nodeloader.is_layer(target=node)

            source = inspect.getsource(node.backward).split(":", 1)[1:]
            source = mass_strip_list(targets=source)
            source = mass_remove_list(targets=source, remove_info=["self."])
            source = mass_replace_list(targets=source, replace_info={
                "_weights": f"w_{layer_position + 1}",
                "activation": f"z_{layer_position}",
                "return": f"{'backward_values' if is_node_layer else f'z_{layer_position}'} =",
                "_output": f"x_{layer_position}"
            })

            if is_node_layer:
                if layer_position == layer_count - 1:
                    source = [f"backward_values = z_{layer_position} * cost_values"]
                source.append(f"gradientW[{layer_position}] += np.dot({f'x_{layer_position - 1}' if layer_position > 0 else 'inputs'}.T, backward_values)")
                source.append(f"gradientB[{layer_position}] += np.sum(backward_values, axis=0)")
                layer_position -= 1

            train_program.add_lines(lines=source, prefix="\t")

        layer_map = [node for node in nn._network_nodes if nn._nodeloader.is_layer(target=node)]
        parameters = {
            k: v for i in range(layer_count) for k, v in {
                f"w_{i}": layer_map[i]._weights,
                f"b_{i}": layer_map[i]._biases,
            }.items()
        }

        train_program.set_parameters(parameters=parameters)
        return train_program