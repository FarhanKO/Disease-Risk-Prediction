"""
artifacts.py — Loading the pickled tabular pipelines, whichever way they were saved.

Each pipeline's first step calls its module's `add_custom_features`, and pickle stores
that function by name: `__main__.add_custom_features` when a notebook saved it,
`src.data.add_custom_features` when `python -m src.train` saved it. All six modules
name their package `src`, so neither name is unique once several modules share a
process. While a pipeline is unpickled, both names are pointed at the loading
module's own data module, then restored.
"""

import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Union

import joblib

_alias_lock = threading.Lock()


@contextmanager
def _module_aliases(data_module: ModuleType):
    package = sys.modules[data_module.__package__]
    main = sys.modules["__main__"]
    saved_modules = {name: sys.modules.get(name) for name in ("src", "src.data")}
    saved_function = main.__dict__.get("add_custom_features")

    sys.modules["src"], sys.modules["src.data"] = package, data_module
    main.add_custom_features = data_module.add_custom_features
    try:
        yield
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if saved_function is None:
            del main.add_custom_features
        else:
            main.add_custom_features = saved_function


def load_pipeline(path: Union[str, Path], data_module: ModuleType):
    """joblib.load a pipeline whose feature-engineering step lives in `data_module`."""
    with _alias_lock, _module_aliases(data_module):
        return joblib.load(path)
