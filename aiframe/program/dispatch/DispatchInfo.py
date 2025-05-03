from aiframe.PropertyUtils import private

from enum import Enum

class DeviceType(Enum):
    CPU = 0
    GPU = 1
    ACCELERATOR = 2

class DisptachInfo():
    def __init__(self, run_device: DeviceType, can_be_parallelized: bool = False):
        self._run_device = run_device
        self._can_be_parallelized = can_be_parallelized

    @private
    def run_device(self):
        return self._run_device

    @private
    def can_be_parallized(self):
        return self._can_be_parallelized