def get_class(target: object | type) -> type:
    return target if isinstance(target, type) else target.__class__

def mass_replace(target: str, replace_info: dict = {}) -> str:
    for k, v in replace_info.items():
        target = target.replace(k, v)
    return target

def mass_replace_list(targets: list[str], replace_info: dict = {}) -> list[str]:
    result = []
    for target in targets:
        result.append(mass_replace(target=target, replace_info=replace_info))
    return result

def mass_strip_list(targets: list[str]) -> list[str]:
    result = []
    for target in targets:
        result.append(target.strip())
    return result

def mass_remove(target: str, remove_info: list[str]) -> str:
    for k in remove_info:
        target = target.replace(k, "")
    return target

def mass_remove_list(targets: list[str], remove_info: list[str]) -> str:
    result = []
    for target in targets:
        result.append(mass_remove(target=target, remove_info=remove_info))
    return result