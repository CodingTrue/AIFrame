from aiframe import NeuralNetwork
from aiframe.program.passes import BasePass, PassInfo, PassInfoParameter
from aiframe.program.ProgramConstants import *
from aiframe.Utils import mass_replace

from uuid import uuid4

import ast
import inspect
import textwrap
from types import GenericAlias

def generate_function_source(name: str, body_content: list, named_arguments: dict) -> str:
    undefined_arguments = []
    defined_arguments = []
    for arg, info in named_arguments.items():
        pass_info = info["pass_info"]
        if not pass_info._is_argument: continue

        value = pass_info._value
        value_is_class = inspect.isclass(value) or value.__class__ == GenericAlias

        value_type = (value if value.__module__ == "builtins" else f"{value.__module__}.{value.__name__}") if value_is_class else value.__class__.__name__

        if value_is_class: undefined_arguments.append(f"{arg}: {value_type}")
        else: defined_arguments.append(f"{arg}: {value_type} = {value}")
    function_arguments = undefined_arguments + defined_arguments

    function_header = f"def {name}({', '.join(function_arguments)}):\n"
    function_body = textwrap.indent(text="\n".join(body_content), prefix="\t")
    return function_header + function_body

def parse_function_body(groups: dict, layer_count: int) -> list[str]:
    body_content = []
    last_line = INPUTS
    for group_name, group_body in groups.items():
        line_index = 0
        layer_index = layer_count - 1 if group_name == BACKWARD_PASS else 0

        for descriptor, content in group_body.items():
            projected_layer_index = layer_index if group_name == FORWARD_PASS else layer_count - layer_index - 1

            if group_name == FORWARD_PASS:
                current_line = f"_{group_name}_" + str(line_index)
            elif group_name == BACKWARD_PASS:
                current_line = BACKWARD_VALUES

            body_content.append(mass_replace(target=content, replace_info={
                LAYER_WEIGHTS: f"{LAYER_WEIGHTS}_{layer_index}",
                LAYER_BIASES: f"{LAYER_BIASES}_{layer_index}",
                CURRENT_LAYER_RESULT: current_line,
                LAST_LAYER_RESULT: last_line,
                CURRENT_LAYER_INDEX: f"{layer_index}",
                GROUP_NAME: group_name,
                LAYER_ACTIVATION_DERIVATIVE: f"{LAYER_ACTIVATION_DERIVATIVE}_{layer_index}"
            }))
            if not descriptor.startswith(DEPENDENT_CODE_DESCRIPTOR): line_index = line_index + 1
            if descriptor.startswith("layer_"):
                layer_index = layer_count - layer_index - 1 if group_name == BACKWARD_PASS else layer_index + 1
            last_line = current_line
    return body_content

class Program():
    def __init__(self, nn: NeuralNetwork = None):
        self._program = None

        self._nn = nn
        self._function_name = DEFAULT_FUNCTION_NAME

        self._groups = {}
        self._active_group = ""

        self._passes = []
        self._named_arguments = {}

    def set_active_group(self, name: str):
        self._active_group = name
        if not self._groups.get(self._active_group): self._groups[self._active_group] = {}

    def add_line(self, line: str = "", descriptor: str = ""):
        if not descriptor: descriptor = str(uuid4())
        self._groups[self._active_group].update({descriptor: line})

    def add_lines(self, lines: list = [], descriptors: list|str = ""):
        for i, line in enumerate(lines):
            self.add_line(line=line, descriptor=descriptors[i] if isinstance(descriptors, list) else descriptors)

    def set_parameters(self, parameters: dict):
        self._program_parameters = parameters

    def assamble(self, nn: NeuralNetwork = None, hard_passes: bool = False, function_name: str = DEFAULT_FUNCTION_NAME):
        if not self._nn and nn: self._nn = nn

        self.assemble_passes()
        self.run_passes(hard_passes=hard_passes)

        body_content = parse_function_body(groups=self._groups, layer_count=self._nn.get_layer_count())
        source_string = generate_function_source(name=self._function_name if self._function_name else function_name, body_content=body_content, named_arguments=self._named_arguments)

        self._program = compile(source=source_string, filename="<string>", mode="exec")
        return self

    def assemble_passes(self):
        for target in self._passes:
            parameters = target.get_pass_info(nn=self._nn)._parameters

            for param in parameters:
                if param._name in self._named_arguments: raise Exception(f"'{param._name}' was already regsitered by '{self._named_arguments[param._name]['associated_class'].__name__}'!")

                self._named_arguments[param._name] = {
                    "pass_info": param,
                    "associated_class": target.__class__
                }

    def get_function(self, function_name: str, program_globals: dict = {}):
        for arg, info in self._named_arguments.items():
            pass_info = info["pass_info"]
            if pass_info._is_argument: continue

            program_globals.update({arg: pass_info._value})

        program_locals = {}
        exec(self._program, program_globals, program_locals)
        return program_locals.get(function_name)

    def run_passes(self, hard_passes: bool = False):
        for _pass in self._passes:
            modifiers = _pass.run_pass()
            modifiers = modifiers if modifiers else []

            for modifier in modifiers:
                modifier.modify(self._groups, hard_pass=hard_passes)

    def add_pass(self, target: BasePass):
        self._passes.append(target)
        return self

    def _debug_log_groups(self):
        for group, lines in self._groups.items():
            print(f"{group}: " + '{\n\t' + '\n\t'.join(
                [f"{v} | {k}" for k, v in lines.items()]
            ) + '\n}')