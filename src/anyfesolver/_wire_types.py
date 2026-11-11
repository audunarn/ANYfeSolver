"""Type identities used by checkpoint and detached-result format versions."""

import sys


def wire_class_name(cls: type) -> str:
    """Return the original wire namespace for a class defined by the solver.

    These identifiers are included in checksums. They are never imported from
    a serialized record. Foreign classes keep their original qualification.
    """
    module_name = cls.__module__
    if module_name.startswith("anyfesolver."):
        owner = sys.modules.get(module_name)
        if owner is not None and vars(owner).get(cls.__qualname__) is cls:
            module_name = "anysolver." + module_name.removeprefix("anyfesolver.")
    return f"{module_name}.{cls.__qualname__}"
