def get_class(target: object | type) -> type:
    return target if isinstance(target, type) else target.__class__