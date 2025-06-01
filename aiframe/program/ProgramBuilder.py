from aiframe import NeuralNetwork
from aiframe.Criterion import BaseCriterion
from aiframe.Utils import mass_strip_list, mass_remove_list, mass_replace_list
from aiframe.program import TrainProgram

import inspect


class ProgramBuilder():
    @staticmethod
    def create_train_program(nn: NeuralNetwork, criterion: BaseCriterion) -> TrainProgram:
        train_program = TrainProgram()

        network_nodes = criterion.preparse_nodes(nodes=nn._network_nodes)
        layer_count = nn.get_layer_count()

        train_program.add_line(line="def train_function(inputs, expected, gradientW, gradientB):")

        layer_position = 0
        offset = 0
        was_last_layer = False
        for i, node in enumerate(network_nodes):
            is_node_layer = nn._nodeloader.is_layer(target=node)

            if (not was_last_layer and is_node_layer) and i > 0: offset += 1

            source = [s for s in inspect.getsource(node.evaluate).split('\n') if not '):' in s and s]
            source = mass_strip_list(targets=source)
            source = mass_remove_list(targets=source, remove_info=["self."])
            source = mass_replace_list(targets=source, replace_info={
                "_output": f"x_{offset}",
                "_weights": f"w_{layer_position}",
                "_biases": f"b_{layer_position}",
                "_input": "inputs" if i == 0 else f"x_{offset - is_node_layer}"
            })

            if len(source[-1]) <= len(f"x_{layer_position}") + 2: continue  # skip empty lines

            if is_node_layer:
                source.append(f"layer_cache_{layer_position} = x_{offset}")
                layer_position += 1
            was_last_layer = is_node_layer

            train_program.add_lines(lines=source, prefix="\t")

        layer_position = layer_count - 1

        source = [s for s in inspect.getsource(criterion.get_loss().loss_derivative).split('\n') if not '):' in s and s]
        source = mass_strip_list(targets=source)
        source = mass_replace_list(targets=source, replace_info={
            "return": "cost_values =",
            "predicted": f"x_{layer_position}"
        })
        train_program.add_lines(lines=source, prefix="\t")

        skipped_last = False
        for i, node in enumerate(network_nodes[::-1]):
            if not criterion.add_node(index=i, node=node):
                skipped_last = True
                continue
            is_node_layer = nn._nodeloader.is_layer(target=node)

            source = [s for s in inspect.getsource(node.backward).split('\n') if not '):' in s and s]
            source = mass_strip_list(targets=source)
            source = mass_remove_list(targets=source, remove_info=["self."])
            source = mass_replace_list(targets=source, replace_info={
                "_weights": f"w_{layer_position + 1}",
                "activation": f"z_{layer_position}",
                "return": f"{'backward_values' if is_node_layer else f'z_{layer_position}'} =",
                "_output": f"x_{layer_position}",
                "_input": f"layer_cache_{layer_position}"
            })

            if len(source[-1]) <= len(f"z_{layer_position}") + 2: continue # skip empty lines

            if is_node_layer:
                if layer_position == layer_count - 1:
                    source = ["backward_values = cost_values"] if skipped_last else [f"backward_values = z_{layer_position} * cost_values"]
                source.append(f"gradientW[{layer_position}] = np.dot({f'x_{layer_position - 1}' if layer_position > 0 else 'inputs'}.T, backward_values)")
                source.append(f"gradientB[{layer_position}] = np.sum(backward_values, axis=0)")
                layer_position -= 1

            train_program.add_lines(lines=source, prefix="\t")
            skipped_last = False

        layer_map = [node for node in network_nodes if nn._nodeloader.is_layer(target=node)]
        parameters = {
            k: v for i in range(layer_count) for k, v in {
                f"w_{i}": layer_map[i]._weights,
                f"b_{i}": layer_map[i]._biases,
            }.items()
        }

        #print('\n'.join(train_program._program_lines))
        #exit()

        train_program.set_parameters(parameters=parameters)
        return train_program