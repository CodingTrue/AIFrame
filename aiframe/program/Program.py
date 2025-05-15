class Program():
    def __init__(self):
        self._program_lines = []
        self._program_parameters = {}
        self._program = None

    def add_line(self, line: str = ""):
        self._program_lines.append(line)

    def add_lines(self, lines: list[str] = []):
        for line in lines:
            self._program_lines.append(line)

    def set_parameters(self, parameters: dict):
        self._program_parameters = parameters

    def assamble(self):
        return

    def add_pass(self):
        return