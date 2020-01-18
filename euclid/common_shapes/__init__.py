# Dynamic loading of common shapes
import pkgutil

__all__ = []
for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)
    globals()[name] = module
    __all__.append(name)
del loader, name, module, is_pkg, pkgutil
