from aiframe import NeuralNetwork
from aiframe.program.passes import BasePass, PassInfo, PassInfoParameter
from uuid import uuid4

class Program():
    def __init__(self, nn: NeuralNetwork = None):
        self._program_lines = []
        self._program_parameters = {}
        self._program = None

        self._nn = nn

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

    def assamble(self, nn: NeuralNetwork = None, hard_passes: bool = False):
        #self._program = compile('\n'.join(self._program_lines), "<string>", "exec")
        #self._debug_log_groups()
        if not self._nn and nn: self._nn = nn

        self.assemble_passes()
        self.run_passes(hard_passes=hard_passes)
        self._debug_log_groups()

        exit(-44)
        return self

    def assemble_passes(self):
        for target in self._passes:
            parameters = target.get_pass_info(nn=self._nn)._parameters

            for param in parameters:
                if param._name in self._named_arguments: raise Exception(f"'{param._name}' was already regsitered by '{self._named_arguments[param._name]}'!")

                if param._is_argument:
                    ...
                else:
                    ...
                self._named_arguments[param._name] = target.__class__.__name__

    def get_function(self, function_name: str, globals: dict = {}):
        globals.update(self._program_parameters)

        locals = {}
        exec(self._program, globals, locals)
        return locals.get(function_name)

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