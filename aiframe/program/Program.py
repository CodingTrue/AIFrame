from aiframe.program.dispatch import DispatchInfo
from aiframe.PropertyUtils import private

class Program():
    def __init__(self, dispatch_info: DispatchInfo):
        self._dispatch_info = dispatch_info

        self._program_elements = []
        self._program = None
        self._parameters = {}

    def assamble(self):
        self._program = compile('\n'.join(self._program_elements), "<string>", "exec")
        return self

    def set_parameters(self, parameters: dict = {}):
        self._parameters = parameters

    def set_program(self, program_elements: list):
        self._program_elements = program_elements

    def add_program_element(self, element: str):
        self._program_elements.append(element)

    def add_program_elements(self, elements: list):
        for element in elements:
            self.add_program_element(element=element)

    @private
    def dispatch_info(self):
        return self._dispatch_info

    @private
    def program_elements(self):
        return self._program_elements

    @private
    def program(self):
        return self._program

    @private
    def parameters(self):
        return self._parameters