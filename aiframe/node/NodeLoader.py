from aiframe.PropertyUtils import private
from aiframe.node.Nodes import *

class NodeLoader():
    def __init__(self):
        self._classes = {}

    def register_default_classes(self):
        self.register_class(target=HiddenLayerNode, is_layer=True, path="", author="aiframe")

        return self

    def register_class(self, target, is_layer: bool = False, path: str = "", author: str = ""):
        if self.is_registered(target=target): raise KeyError(f"'{target.__name__}' is already registered in this NodeLoader!")

        path += target.__name__
        self._classes[target.__name__] = {
            "class": target,
            "is_layer": is_layer,
            "path": f"{author}:{path}"
        }
        return self

    def is_registered(self, target) -> bool:
        return target.__name__ in self._classes

    def is_layer(self, target):
        if not self.is_registered(target=target): raise KeyError(f"'{target.__name__}' is not registered in this NodeLoader!")
        return self._classes[target.__name__]["is_layer"]

BASIC_NODE_LOADER = NodeLoader().register_default_classes()