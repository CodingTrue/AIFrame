from aiframe.node.Nodes import BaseNode, HiddenLayerNode, ReluActivationNode
from aiframe.Utils import get_class

class NodeLoader():
    def __init__(self):
        self._registry = {}

    def register_defualts(self):
        self.register(target=HiddenLayerNode, is_layer=True, author="builtin", path="HiddenLayerNode")
        self.register(target=ReluActivationNode, is_layer=False, author="builtin", path="ReluActivationNode")

        return self

    def is_registered(self, target: BaseNode | type) -> bool:
        return get_class(target) in self._registry

    def is_layer(self, target: BaseNode | type):
        target_class = get_class(target=target)
        if not self.is_registered(target=target_class): raise TypeError(f"'{target_class.__name__}' is not registered!")
        return self._registry[target_class]["is_layer"]

    def register(self, target: BaseNode | type, is_layer: bool = False, author: str = "", path: str = ""):
        target_class = get_class(target=target)
        if self.is_registered(target=target_class): raise TypeError(f"'{target_class.__name__}' is already registered!")

        path = f"{author}:{path}"
        self._registry[target_class] = {
            "is_layer": is_layer,
            "path": path
        }
        return self

BASIC_NODE_LOADER = NodeLoader().register_defualts()