"""Shared derived-measure formulas for ANYfeSolver recovery capsules.

This module is the single source of the derived stress measures that
ANYfeSolver's prepared response routes export and that the offline
``anyfesolver_results`` runtime recomputes.  The solver delegates to these
functions; the runtime imports them directly, so capsule reconstruction and
in-solver recovery share one formula set.

NumPy only: this package must not import ``anyfesolver``, scipy or any sibling
application package.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

PRINCIPAL_COMPONENTS = ('principal_stresses_top', 'principal_stresses_bottom')
SHELL_LINEAR_COMPONENTS = ('membrane_xx', 'membrane_yy', 'membrane_xy',
                           'bending_xx', 'bending_yy', 'bending_xy', 'shear_xz', 'shear_yz')
BEAM_VON_MISES_COMPONENTS = ('axial_stress', 'bending_stress_y', 'bending_stress_z',
                             'shear_stress_y', 'shear_stress_z', 'torsional_stress')


def von_mises_surface(sxx: Any, syy: Any, szz: Any, sxy: Any, syz: Any, sxz: Any) -> Any:
    """Plane-stress-state von Mises measure (shared reviewed expression)."""
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
    )


def shell_surface_von_mises(record: Mapping[str, Any]) -> Any:
    """Reviewed shell von Mises: max of the top/bottom plane-stress measures
    with the shared transverse shear.

    Source: ``elements.ShellElement.compute_stresses`` and
    ``e4_pl_s3_element.QualifiedE4PLS3ShellElement`` elastic recovery.
    """
    membrane = (record["membrane_xx"], record["membrane_yy"], record["membrane_xy"])
    bending = (record["bending_xx"], record["bending_yy"], record["bending_xy"])
    shear = (record["shear_xz"], record["shear_yz"])
    arrays = np.broadcast_arrays(*membrane, *bending, *shear)
    mx, my, mxy, bx, by, bxy, sxz, syz = arrays
    surfaces = [
        von_mises_surface(mx + sign * bx, my + sign * by, 0.0,
                          mxy + sign * bxy, syz, sxz)
        for sign in (1.0, -1.0)
    ]
    return np.maximum(surfaces[0], surfaces[1])


def beam_von_mises(record: Mapping[str, Any]) -> Any:
    """Reviewed beam von Mises from canonical section fields.

    Source: ``elements.BeamElement.compute_stresses`` elastic path.
    """
    sigma_x = (
        record["axial_stress"]
        + record["bending_stress_y"]
        + record["bending_stress_z"]
    )
    return np.sqrt(
        sigma_x * sigma_x
        + 3.0
        * (
            record["shear_stress_y"] ** 2
            + record["shear_stress_z"] ** 2
            + record["torsional_stress"] ** 2
        )
    )


def principal_stresses(tensor: Any, *, workers: int = 1, execution=None) -> np.ndarray:
    """Return descending eigenvalues along the final axis, one tensor per station."""
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError('principal kernel workers must be 1 or 2')
    if execution is not None and type(execution) is not dict:
        raise ValueError('principal execution metadata must be a plain dict')
    values = np.asarray(tensor, dtype=float)
    if values.shape[-2:] != (3, 3) or not np.all(np.isfinite(values)):
        raise ValueError('principal stresses require finite 3 by 3 tensors')
    transposed=np.swapaxes(values, -1, -2)
    # Finite exact symmetry implies the original tolerant predicate. Keep
    # its unchanged fallback for caller tensors with small asymmetry.
    if not np.array_equal(values, transposed) and not np.allclose(values, transposed, rtol=1e-12, atol=1e-12):
        raise ValueError('principal stresses require symmetric tensors')
    symmetric = .5 * (values + np.swapaxes(values, -1, -2))
    rows = int(np.prod(symmetric.shape[:-2]))
    actual_workers = workers if rows >= 4096 else 1
    if execution is not None:
        execution.update(workers=actual_workers, tensors=rows)
    if actual_workers == 1:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    else:
        # Each LAPACK invocation still receives one identical 3x3 tensor.
        # Partition only independent stations; keep validation, symmetrisation,
        # reduction order inside each tensor, and result order unchanged.
        from concurrent.futures import ThreadPoolExecutor
        flat = symmetric.reshape(rows, 3, 3)
        midpoint = rows // 2
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='anyfesolver-principal') as pool:
            parts = tuple(pool.map(np.linalg.eigvalsh, (flat[:midpoint], flat[midpoint:])))
        eigenvalues = np.concatenate(parts, axis=0).reshape(symmetric.shape[:-1])
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("principal stress eigenvalues must be finite")
    return eigenvalues[..., ::-1].copy()


def shell_surface_component(record: Mapping[str, Any], surface: str, component: str) -> np.ndarray:
    """Recover one local surface component from its membrane/bending pair.

    Pair-only records remain supported, including scalar and native-sample
    arrays. This compatibility operation preserves the original float64
    addition/subtraction order and missing values; it does not certify a
    complete stress tensor or a new response combination.
    """
    if surface not in ('top', 'bottom') or component not in ('xx', 'yy', 'xy'):
        raise ValueError('shell surface requires top/bottom and xx/yy/xy')
    membrane = record.get(f'membrane_{component}')
    bending = record.get(f'bending_{component}')
    if membrane is None or bending is None:
        raise ValueError('shell surface component requires its membrane/bending pair')
    membrane = np.asarray(membrane, dtype=np.float64)
    bending = np.asarray(bending, dtype=np.float64)
    return membrane + bending if surface == 'top' else membrane - bending


def shell_surface_tensor(record: Mapping[str, Any], surface: str) -> np.ndarray:
    """Physical local top/bottom tensors matching canonical shell surface recovery."""
    if surface not in ('top', 'bottom') or not all(name in record for name in SHELL_LINEAR_COMPONENTS):
        raise ValueError('physical shell surface stresses are unavailable; generalized resultants and beam envelopes cannot define a coherent tensor')
    mx, my, mxy, bx, by, bxy, sxz, syz = np.broadcast_arrays(*(np.asarray(record[name], dtype=float) for name in SHELL_LINEAR_COMPONENTS))
    sign = 1. if surface == 'top' else -1.
    tensor = np.zeros(mx.shape + (3, 3), dtype=float)
    tensor[..., 0, 0] = mx + sign*bx
    tensor[..., 1, 1] = my + sign*by
    tensor[..., 0, 1] = tensor[..., 1, 0] = mxy + sign*bxy
    tensor[..., 0, 2] = tensor[..., 2, 0] = sxz
    tensor[..., 1, 2] = tensor[..., 2, 1] = syz
    return tensor


def shell_principal_stresses(record: Mapping[str, Any], surface: str, *, workers: int = 1, execution=None) -> np.ndarray:
    return principal_stresses(shell_surface_tensor(record, surface), workers=workers, execution=execution)
