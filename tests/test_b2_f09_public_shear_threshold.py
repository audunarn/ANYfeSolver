"""Exact-input F09 shear-floor distinction through the public physical route.

The historical scalar law remains unchanged. Agreement is required only at
and above its existing floor; below it the mismatch is retained as evidence.
"""

from fractions import Fraction as F
import json

import numpy as np
import pytest
from scipy import sparse

from anyfesolver import stiffened_shell
from anyfesolver.elements import BeamElement
from anyfesolver.fe_core import Material


@pytest.mark.parametrize("area", (1e-14, 2e-12, 1e-10))
@pytest.mark.parametrize("direction,index", (("y", 7), ("z", 8)))
def test_public_physical_threshold_preserves_unfloored_work(area, direction, index, tmp_path):
    # Same E, nu, L, section and displacement as the frozen F09-S-v1 witness.
    length, displacement, inertia = 1., .01, 1e-12
    section = stiffened_shell.LinearSection.isotropic(
        E=1., nu=0., area=area, Iy=inertia, Iz=inertia, J=inertia,
        ky=1., kz=1.)
    member = stiffened_shell.PhysicalB2Member(
        "threshold", "F09-S-v1-input", (1, 2),
        ((0., 0., 0.), (length, 0., 0.)), (0., 1., 0.),
        "scalar-threshold", section)
    # Independent unfloored Timoshenko entry. No local K/B/H is the oracle.
    ei, ga, ell = F(inertia), F(area)/2, F(length)
    physical_entry = float(12*ei*ga/(ell*(ga*ell**2+12*ei)))
    expected_energy = .5*physical_entry*displacement**2
    load = np.zeros(12)
    load[index] = physical_entry*displacement
    transform = sparse.csr_matrix(([1.], ([index], [0])), shape=(12, 1))
    owner = stiffened_shell.create_analysis(definition=stiffened_shell.B2LinearDefinition(
        stiffened_shell.B2_E4_PL_FAST_LINEAR_V1, (member,), transform,
        np.zeros(12), (index,), max_rhs_chunk=1))
    try:
        result = owner.solve_linear_many(
            (("threshold", load),), recover_members={"threshold": (0., .5, 1.)})
        assert result.status == "completed", result.failure_reason
        assert result.factorization_count == result.solve_count == 1
        assert not result.production_qualified
        case, = result.cases
        actual_displacement = float(case.values[0])
        recovered, = case.members
        # Half the integral of section resultant dot strain. Simpson is
        # exact here because this constant-section field is affine.
        densities = [float(np.asarray(wrench)@np.asarray(strain))
                     for wrench, strain in zip(recovered.resultants_local,
                                              recovered.strains_local, strict=True)]
        station_energy = length*(densities[0]+4*densities[1]+densities[2])/12
        public_energy = .5*load[index]*actual_displacement
        material = Material("steel", 1., 0., density=1.)
        historical = BeamElement(1, [1, 2], "steel", {
            "area": area, "Iy": inertia, "Iz": inertia, "J": inertia,
            "shear_factor_y": 1., "shear_factor_z": 1.})
        old_entry = float(historical._local_linear_stiffness(length, material)[index,index])
        old_energy = .5*old_entry*displacement**2
        observations = {
            "area_m2": area, "direction": direction,
            "physical_entry": physical_entry, "historical_entry": old_entry,
            "expected_energy": expected_energy, "public_energy": public_energy,
            "recovered_station_energy": station_energy,
            "actual_displacement": actual_displacement,
            "historical_over_physical": old_energy/expected_energy,
            "public_relative_energy_error": abs(public_energy-expected_energy)/expected_energy,
            "station_relative_energy_error": abs(station_energy-expected_energy)/expected_energy,
            "historical_relative_energy_error": abs(old_energy-expected_energy)/expected_energy,
        }
        (tmp_path/'public-shear-threshold.json').write_text(json.dumps(observations, indent=2))
        assert abs(actual_displacement-displacement)/displacement <= 1e-12
        assert observations["public_relative_energy_error"] <= 1e-12
        assert observations["station_relative_energy_error"] <= 1e-12
        if area == 1e-14:
            assert observations["historical_over_physical"] > 100.
        else:
            assert observations["historical_relative_energy_error"] <= 1e-12
            assert abs(public_energy-old_energy)/expected_energy <= 1e-12
    finally:
        owner.close()
