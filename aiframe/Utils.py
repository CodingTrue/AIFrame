import ast

def unparse_ast_list(node_List):
    return '\n'.join([ast.unparse(l) for l in node_List])

def mass_replace(target: str, info: dict) -> str:
    for replacement in info:
        target = target.replace(replacement, info[replacement])
    return target

def mass_strip(target: str, info: list):
    return mass_replace(target=target, info={x: '' for x in info})