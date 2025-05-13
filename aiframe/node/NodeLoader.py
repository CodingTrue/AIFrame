from aiframe.node.Nodes import BaseNode, HiddenLayerNode

def get_class(target: BaseNode | type) -> type:
    return target if isinstance(target, type) else target.__class__

class NodeLoader():
    def __init__(self):
        self._registry = {}

    def register_defualts(self):
        self.register(target=HiddenLayerNode, is_layer=True, author="builtin", path="HiddenLayerNode")

        return self

    def is_registered(self, target: BaseNode | type) -> bool:
        return get_class(target) in self._registry

    def is_layer(self, target: BaseNode | type):
        if not self.is_registered(target=target_class): raise TypeError(f"'{target_class.__name__}' is not registered!")
        return self._registry[get_class(target=target)]["is_layer"]

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