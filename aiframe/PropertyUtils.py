def private(func):
    def setter(self, *args, **kwargs):
        raise AttributeError(f"'{func.__name__}' is a private property and can not be changed!")
    return property(func, setter)