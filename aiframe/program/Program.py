class Program():
    def __init__(self):
        self._program_lines = []
        self._program_parameters = {}
        self._program = None

    def add_line(self, line: str = "", prefix: str = ""):
        self._program_lines.append(prefix + line)

    def add_lines(self, lines: list[str] = [], prefix: str = ""):
        for line in lines:
            self._program_lines.append(prefix + line)

    def set_parameters(self, parameters: dict):
        self._program_parameters = parameters

    def assamble(self):
        self._program = compile('\n'.join(self._program_lines), "<string>", "exec")
        return self

    def get_function(self, function_name: str, globals: dict = {}):
        globals.update(self._program_parameters)

        locals = {}
        exec(self._program, globals, locals)
        return locals.get(function_name)

    def add_pass(self):
        return