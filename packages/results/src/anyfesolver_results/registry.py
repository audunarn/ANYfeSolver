"""Closed derived-operation registry owned by the anyfesolver-results runtime.

ANYpost resolves dependencies and calls :func:`derive_field`; the physics and
the complete set of admitted operations live here.  No operation may be added
without an adopted solver contract, and no unqualified or fabricated quantity
(for example a zero-valued Hill utilisation) is registered.

Every registered operation returns one scalar value per spatial sample, so a
derived component never changes the spatial shape of its inputs.  The shell
principal-stress operation therefore takes an explicit ``rank`` parameter
(1, 2, 3; descending eigenvalues) instead of returning three values per
sample.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import formulas
from .errors import DerivationError

SHELL_INPUTS = formulas.SHELL_LINEAR_COMPONENTS
BEAM_INPUTS = formulas.BEAM_VON_MISES_COMPONENTS
PRINCIPAL_RANKS = (1, 2, 3)

#: operation name -> (required canonical input names, required parameters)
OPERATIONS = {
    'shell_von_mises': (tuple(SHELL_INPUTS), ()),
    'beam_von_mises': (tuple(BEAM_INPUTS), ()),
    'equivalent_stress': (('von_mises',), ()),
    'shell_principal_stress': (tuple(SHELL_INPUTS), ('surface', 'rank')),
}


def _require_arrays(operation: str, inputs: Mapping[str, Any], names) -> dict:
    resolved = {}
    shape = None
    for name in names:
        if name not in inputs:
            raise DerivationError(
                f"operation {operation!r} is missing input {name!r}")
        value = inputs[name]
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise DerivationError(
                f"input {name!r} of operation {operation!r} must be a 1-D array")
        array = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise DerivationError(
                f"input {name!r} of operation {operation!r} is not finite")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise DerivationError(
                f"inputs of operation {operation!r} must share one spatial shape")
        resolved[name] = array
    return resolved


DERIVATION_EXECUTION_API = 1


def derive_field(specification: Mapping[str, Any], inputs: Mapping[str, Any], *, workers: int = 1, execution=None) -> np.ndarray:
    """Compute one derived field from already reconstructed input arrays.

    ``specification`` is a JSON mapping with ``operation`` and ``inputs`` (a
    mapping of canonical recovery input names to component keys), plus
    operation parameters such as ``surface`` and ``rank``.  ``inputs`` maps the
    same canonical names to already reconstructed 1-D spatial arrays.  The
    returned array is float64, read-only and scalar per spatial sample.
    """
    if type(workers) is not int or workers not in (1, 2):
        raise DerivationError('derived kernel workers must be 1 or 2')
    if execution is not None and type(execution) is not dict:
        raise DerivationError('derived execution metadata must be a plain dict')
    if execution is not None:
        execution.update(workers=1)
    if not isinstance(specification, Mapping):
        raise DerivationError('derived specification must be a mapping')
    operation = specification.get('operation')
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise DerivationError(f'unknown derived operation: {operation!r}')
    names, parameters = OPERATIONS[operation]
    declared = specification.get('inputs')
    if not isinstance(declared, Mapping):
        raise DerivationError('derived specification requires an inputs mapping')
    surface = specification.get('surface')
    rank = specification.get('rank')
    if 'surface' in parameters and surface not in ('top', 'bottom'):
        raise DerivationError(
            f'operation {operation!r} requires surface "top" or "bottom"')
    if 'rank' in parameters:
        if type(rank) is not int or rank not in PRINCIPAL_RANKS:
            raise DerivationError(
                f'operation {operation!r} requires an integer rank in {list(PRINCIPAL_RANKS)}')
    resolved = _require_arrays(operation, inputs, names)
    if operation == 'shell_von_mises':
        result = formulas.shell_surface_von_mises(resolved)
    elif operation == 'beam_von_mises':
        result = formulas.beam_von_mises(resolved)
    elif operation == 'equivalent_stress':
        result = np.asarray(resolved['von_mises'], dtype=np.float64).copy()
    elif operation == 'shell_principal_stress':
        try:
            result = formulas.shell_principal_stresses(resolved, surface, workers=workers, execution=execution)[..., rank - 1]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            # Only expected numerical failures cross the public derived-field
            # boundary. Cancellation, resource and programmer errors propagate.
            raise DerivationError(f'principal stress derivation failed: {error}') from error
    else:  # pragma: no cover - registry is closed
        raise DerivationError(f'unknown derived operation: {operation!r}')
    array = np.asarray(result, dtype=np.float64).reshape(-1)
    array.setflags(write=False)
    return array
