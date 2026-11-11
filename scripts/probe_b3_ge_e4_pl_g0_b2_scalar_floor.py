"""G0 F09-S: public B2 scalar shear-floor physical-range witness.

The physical comparator is an independent closed-form bending/shear entry.
This diagnostic leaves the public element and its historical identity intact.
"""

from __future__ import annotations

import json
import math

from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import Material


def case(area: float, direction: str) -> dict:
    modulus = 1.0
    poisson = 0.0
    inertia = 1.0e-12
    shear_factor = 1.0
    length = 1.0
    displacement = 0.01
    material = Material("steel", modulus, poisson, density=1.0)
    beam = BeamElement(
        1, [1, 2], "steel",
        {"area": area, "Iy": inertia, "Iz": inertia, "J": inertia,
         "shear_factor_y": shear_factor, "shear_factor_z": shear_factor},
    )
    public_stiffness = beam._local_linear_stiffness(length, material)
    index = {"y": 7, "z": 8}[direction]
    public_entry = float(public_stiffness[index, index])

    shear_modulus = modulus / (2.0 * (1.0 + poisson))
    physical_phi = 12.0 * modulus * inertia / (
        shear_modulus * area * shear_factor * length**2
    )
    physical_entry = 12.0 * modulus * inertia / (
        length**3 * (1.0 + physical_phi)
    )
    public_energy = 0.5 * public_entry * displacement**2
    physical_energy = 0.5 * physical_entry * displacement**2
    ratio = public_energy / physical_energy
    relative_difference = abs(public_energy - physical_energy) / physical_energy
    assert all(math.isfinite(value) and value > 0.0 for value in
               (public_entry, physical_entry, public_energy, physical_energy))
    if area == 1.0e-14:
        assert ratio > 100.0
    else:
        assert relative_difference <= 1.0e-12
    return {
        "area": area, "direction": direction, "public_entry": public_entry,
        "physical_entry": physical_entry, "public_energy": public_energy,
        "physical_energy": physical_energy, "public_over_physical": ratio,
        "relative_difference": relative_difference,
    }


def main() -> int:
    rows = [case(area, direction)
            for area in (1.0e-14, 2.0e-12, 1.0e-10)
            for direction in ("y", "z")]
    assert len(rows) == 6
    print(json.dumps({
        "profile": "G0-F09-S-v1",
        "status": "BOUNDED_PHYSICAL_RANGE_MISMATCH",
        "public_route": "BeamElement scalar local linear stiffness",
        "rows": rows,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
