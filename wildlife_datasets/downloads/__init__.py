import sys
import pkgutil
import importlib

# Import all submodules
__all__ = []
package = sys.modules[__name__]
for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
    importlib.import_module(__name__ + '.' + name)
    __all__.append(name)
