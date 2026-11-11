"""One-sided physical yield limits of the admitted full shell pullbacks.

Uniform in-plane dilation has an identity Procrustes fit for these centred
planar polygons, so d(epsilon*v)=epsilon*v exactly. In plane stress, with
epsilon_x=epsilon_y=epsilon, tau=sigma_x=sigma_y=B*(epsilon-p),
B=E/(1-nu). Native associated J2 flow gives alpha-alpha0=2*(p-p0).
On C208's first linear segment, sigma_flow=sigma_prop+H*alpha. Thus

    epsilon_b = p0 + (sigma_prop + H*alpha0)/B,
    d tau/d epsilon | elastic = B,
    d tau/d epsilon | plastic = B*H/(H+B/2).

These are independent constitutive/edge-traction oracles, not boundaries found
by asking the production return map to switch branches. The stencils approach
epsilon_b strictly from each side and never difference across yield. Their
resolved offsets lie outside the native 1e-10 yield-function guard; behaviour
at the guard itself is not asserted. These cases cover virgin first yield and
reloading yield from one fixed yielded predecessor, not reversal or a general
nonproportional plastic history. No plastic potential or symmetry is assumed.
"""

from dataclasses import dataclass
import hashlib
import inspect
import json

import numpy as np
import pytest

from anyfesolver import _stiffened_shell_q4_algorithmic_pullback as q4
from anyfesolver import _stiffened_shell_s3_algorithmic_pullback as s3
from anyfesolver.e4_pl_s3_state import canonical_json_bytes
from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEModel
from anyfesolver.material_curves import DNVC208MaterialCurve
from anyfesolver.plasticity import plane_stress_return_map


@dataclass(frozen=True)
class _YieldCase:
    model: object
    element: object
    adapter: object
    origin: dict
    direction: np.ndarray
    unit_stress_force: np.ndarray
    boundary: float
    step_scale: float
    biaxial_modulus: float
    hardening: float
    plastic0: float
    alpha0: float
    layers: int
    curve: object


def _edge_traction(reference, thickness):
    """Exact linear nodal work of uniform biaxial unit stress on each edge."""
    result = np.zeros((len(reference), 6))
    for i, point in enumerate(reference):
        j = (i+1) % len(reference)
        dx, dy = (reference[j]-point)[:2]
        # For a CCW polygon, outward normal times edge length is (dy,-dx).
        traction = thickness*np.array((dy, -dx))/2.
        result[i, :2] += traction
        result[j, :2] += traction
    return result.ravel()


@pytest.fixture(scope="module", params=(
    ("q4", "virgin"), ("q4", "accepted"),
    ("s3", "virgin"), ("s3", "accepted"),
), ids=("q4-virgin", "q4-reload", "s3-virgin", "s3-reload"))
def yield_case(request):
    family, predecessor = request.param
    reference = np.array(
        ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))
        if family == "q4" else
        ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    reference -= reference.mean(axis=0)
    young, poisson, thickness = 210e9, .3, .02
    curve = DNVC208MaterialCurve(100e6, 105e6, 110e6,
                                .005, .01, 400e6, .2)
    model = FEModel(f"{family}-one-sided-{predecessor}-yield")
    material = model.add_material("shell", young, poisson, density=7850.,
                                  hardening_curve=curve)
    for node, point in enumerate(reference, 1):
        model.add_node(node, *point)
    element = create_shell_element(
        1, tuple(range(1, len(reference)+1)), "shell",
        formulation="e4-pl" if family == "q4" else "e4-pl-s3-v2d",
        thickness=thickness, reference_normal=(0., 0., 1.))
    model.add_element(1, element)
    adapter = q4 if family == "q4" else s3
    layers = 3 if family == "q4" else 5
    origin = (element.init_nonlinear_state(layers) if family == "q4" else
              element.init_model_bound_nonlinear_state(model.mesh, material, layers))
    direction = np.zeros((len(reference), 6))
    direction[:, :2] = reference[:, :2]
    direction = direction.ravel()
    biaxial = young/(1.-poisson)
    hardening = (curve.sigma_yield-curve.sigma_prop)/curve.eps_p_y1
    first_yield = curve.sigma_prop/biaxial
    if predecessor == "accepted":
        accepted_strain = 1.2*first_yield
        before = canonical_json_bytes(origin)
        _, _, candidate = adapter.response(
            model, element, accepted_strain*direction, origin, layers)
        assert canonical_json_bytes(origin) == before
        # Independent first-segment solution for this accepted history.
        expected_alpha = ((biaxial*accepted_strain-curve.sigma_prop) /
                          (hardening+biaxial/2.))
        assert 0. < expected_alpha < curve.eps_p_y1
        np.testing.assert_allclose(candidate["alpha"], expected_alpha,
                                   rtol=1e-8, atol=1e-12)
        expected_plastic = np.zeros_like(candidate["plastic_strain"])
        expected_plastic[:, :2] = expected_alpha/2.
        np.testing.assert_allclose(candidate["plastic_strain"], expected_plastic,
                                   rtol=1e-8, atol=1e-12)
        local = adapter.local_coordinates(model, element, accepted_strain*direction)
        if family == "q4":
            origin = element.seal_committed_current_tangent_state(
                model.mesh, material, local, candidate, layers)
        else:
            origin = element.seal_solver_integrated_nonlinear_state(
                model.mesh, material, candidate, layers, local, kinematics="LINEAR")
    plastic = np.asarray(origin["plastic_strain"])
    alpha = np.asarray(origin["alpha"])
    plastic0, alpha0 = float(plastic[0, 0]), float(alpha[0])
    np.testing.assert_allclose(plastic[:, :2], plastic0, rtol=0., atol=1e-12)
    np.testing.assert_allclose(plastic[:, 2], 0., rtol=0., atol=1e-12)
    np.testing.assert_allclose(alpha, alpha0, rtol=0., atol=1e-12)
    assert abs(alpha0-2.*plastic0) <= 1e-12
    # Derive the reloading boundary from the fixed accepted history; no search
    # of production branch flags, residuals or candidate alpha defines it.
    boundary = plastic0+(curve.sigma_prop+hardening*alpha0)/biaxial
    return _YieldCase(model, element, adapter, origin, direction,
                      _edge_traction(reference, thickness), boundary,
                      first_yield, biaxial, hardening, plastic0, alpha0, layers, curve)


@pytest.mark.parametrize("side,limit", ((-1., 1e-6), (1., 1e-5)),
                         ids=("elastic-left", "plastic-right"))
def test_full_pullback_one_sided_yield_derivative(yield_case, side, limit, tmp_path, request):
    case = yield_case
    before = canonical_json_bytes(case.origin)
    slope = (case.biaxial_modulus if side < 0. else
             case.biaxial_modulus*case.hardening /
             (case.hardening+case.biaxial_modulus/2.))
    expected_derivative = slope*case.unit_stress_force
    scale = np.linalg.norm(expected_derivative)
    assert scale > 0.
    native_tolerance = inspect.signature(plane_stress_return_map).parameters["tolerance"].default
    assert native_tolerance == 1e-10
    steps = case.step_scale*np.array((2e-2, 1e-2))
    record = {
        "case": request.node.nodeid, "analytic_boundary": case.boundary,
        "alpha0": case.alpha0, "plastic0": case.plastic0, "side": side,
        "layers": case.layers,
        "biaxial_modulus": case.biaxial_modulus, "hardening": case.hardening,
        "analytic_stress_slope": slope, "prescribed_h": steps.tolist(),
        "derivative_limit": limit, "oracle_limit": 1e-8,
        "native_return_map_tolerance": native_tolerance,
        "predecessor_before_sha256": hashlib.sha256(before).hexdigest(),
        "samples": [], "stencil_noise_bounds": [], "derivatives": [], "complete": False,
    }
    record_path = tmp_path/"yield-limit.json"
    errors = []
    try:
        # Adjacent resolved levels are chosen prospectively, not from measured
        # errors. Both the base and every stencil/midpoint remain on one side.
        for step in steps:
            # The preserved v1 smaller offsets were unresolved under this native
            # constitutive noise bound. From |(tau^2-flow^2)/3|<=t*flow^2,
            # r=tau/flow lies between sqrt(1-3t) and sqrt(1+3t). Since
            # dflow/dtau=-2H/B, the following exact first-segment bound tends
            # to 1.5*t*flow/(1+2H/B). The outward stencil amplifies it by 4/h.
            # This resolution calculation uses the unchanged native tolerance,
            # analytic stress and prescribed h, rather than observed errors.
            stress_noise = 0.
            if side > 0.:
                r_min = np.sqrt(1.-3.*native_tolerance)
                relative_stress_noise = ((3.*native_tolerance/(1.+r_min)) /
                                        (1.+(2.*case.hardening/case.biaxial_modulus)*r_min))
                max_exact_stress = (case.curve.sigma_prop+case.hardening*case.alpha0+
                                    3.*step*slope)
                stress_noise = max_exact_stress*relative_stress_noise
            stencil_noise = 4.*stress_noise/(step*slope)
            record["stencil_noise_bounds"].append({
                "h": float(step), "constitutive_stress_absolute_bound": float(stress_noise),
                "one_sided_relative_bound": float(stencil_noise),
            })
            assert stencil_noise <= limit
            samples = {}
            for distance in (1., 1.5, 2., 2.5, 3.):
                strain = case.boundary+side*distance*step
                flow0 = case.curve.sigma_prop+case.hardening*case.alpha0
                trial_stress = case.biaxial_modulus*(strain-case.plastic0)
                trial_yield = (trial_stress**2-flow0**2)/(3.*flow0**2)
                sample = {"h": float(step), "distance": distance,
                          "strain": float(strain),
                          "normalized_trial_yield": float(trial_yield)}
                record["samples"].append(sample)
                # Independently selected side, resolved beyond the numerical guard.
                assert side*trial_yield > 1e-8
                trial = strain*case.direction
                force, tangent, state = case.adapter.response(
                    case.model, case.element, trial, case.origin, case.layers)
                active = np.asarray(state["alpha"]) > np.asarray(case.origin["alpha"])
                sample.update(active_count=int(np.count_nonzero(active)),
                              station_count=int(active.size),
                              finite=bool(np.isfinite(force).all() and
                                          np.isfinite(tangent).all() and
                                          all(np.isfinite(state[name]).all() for name in
                                              ("layer_strain", "layer_stress", "plastic_strain", "alpha"))))
                assert sample["finite"]
                np.testing.assert_allclose(
                    case.adapter.local_coordinates(case.model, case.element, trial),
                    trial, rtol=0., atol=2e-14)
                np.testing.assert_array_equal(active, np.full(active.shape, side > 0.))
                if side < 0.:
                    np.testing.assert_array_equal(state["alpha"], case.origin["alpha"])
                    increment = 0.
                else:
                    increment = (case.biaxial_modulus*distance*step /
                                 (case.hardening+case.biaxial_modulus/2.))
                expected_alpha = case.alpha0+increment
                assert expected_alpha < case.curve.eps_p_y1
                expected_strain = np.zeros_like(state["layer_strain"])
                expected_strain[:, :2] = strain
                stress = case.biaxial_modulus*(strain-case.plastic0-increment/2.)
                expected_stress = np.zeros_like(state["layer_stress"])
                expected_stress[:, :2] = stress
                expected_force = stress*case.unit_stress_force
                sample.update(
                    strain_oracle_relative=float(np.linalg.norm(state["layer_strain"]-expected_strain) /
                                                 np.linalg.norm(expected_strain)),
                    stress_oracle_relative=float(np.linalg.norm(state["layer_stress"]-expected_stress) /
                                                 np.linalg.norm(expected_stress)),
                    force_oracle_relative=float(np.linalg.norm(force-expected_force) /
                                                np.linalg.norm(expected_force)),
                    alpha_oracle_max_absolute=float(np.max(np.abs(state["alpha"]-expected_alpha))),
                )
                assert sample["strain_oracle_relative"] <= 1e-8
                assert sample["stress_oracle_relative"] <= 1e-8
                np.testing.assert_allclose(state["alpha"], expected_alpha,
                                           rtol=1e-8, atol=1e-12)
                expected_plastic = np.zeros_like(state["plastic_strain"])
                expected_plastic[:, :2] = case.plastic0+increment/2.
                np.testing.assert_allclose(state["plastic_strain"], expected_plastic,
                                           rtol=1e-8, atol=1e-12)
                assert sample["force_oracle_relative"] <= 1e-8
                assert canonical_json_bytes(case.origin) == before
                samples[distance] = force, tangent
            # Second-order outward one-sided derivative at epsilon_b+side*step.
            # It approaches the boundary while all its samples stay in one branch.
            numeric = side*(-3.*samples[1.][0]+4.*samples[2.][0]-samples[3.][0])/(2.*step)
            predicted = samples[1.][1]@case.direction
            error = np.linalg.norm(numeric-predicted)/scale
            predicted_error = np.linalg.norm(predicted-expected_derivative)/scale
            numeric_error = np.linalg.norm(numeric-expected_derivative)/scale
            record["derivatives"].append({
                "h": float(step), "one_sided_vs_tangent_relative": float(error),
                "predicted_vs_analytic_relative": float(predicted_error),
                "one_sided_vs_analytic_relative": float(numeric_error),
                "constitutive_noise_relative_bound": float(stencil_noise),
            })
            errors.append(error)
            assert predicted_error <= limit
            assert numeric_error <= limit
        assert max(errors) <= limit, errors
        assert canonical_json_bytes(case.origin) == before
        record["complete"] = True
    finally:
        try:
            after = canonical_json_bytes(case.origin)
            record["predecessor_after_sha256"] = hashlib.sha256(after).hexdigest()
            record["predecessor_unchanged"] = after == before
        except Exception as error:
            record["predecessor_unchanged"] = False
            record["predecessor_serialization_error"] = repr(error)
        for field in ("force_oracle_relative", "stress_oracle_relative"):
            values = [sample[field] for sample in record["samples"] if field in sample]
            record["max_"+field] = max(values) if values else None
        record_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        print(f"yield-limit predicates: {record_path}")
