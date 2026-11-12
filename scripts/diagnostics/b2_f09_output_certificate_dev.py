"""Bounded F09 output-certificate discriminator, not production admission.

Exact basic energies plus bounded physical rigid/cross work. This does not
certify an assembled graph or all-amplitude relative energy in its rigid kernel.
Run as pytest; the integration note binds this batch before execution.
"""
from fractions import Fraction as F
import json

import numpy as np
import pytest

from anyfesolver._b2_physical_flexibility_linear import LinearOperator, LinearSection
from test_b2_f09_fixed_aspect_wrench import _section, WIDTH_DEPTH_M, FIRST_M

ENERGY_LIMIT = F(1, 10**8)
RIGID_LIMIT = F(1, 10**9)


def _transpose(a):
    return list(map(list, zip(*a)))


def _product(a, b):
    return [[sum((x*y for x, y in zip(row, column)), F(0))
             for column in zip(*b)] for row in a]


def _inverse(a):
    n = len(a)
    rows = [list(map(F, row))+[F(i == j) for j in range(n)]
            for i, row in enumerate(a)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [v/scale for v in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [x-factor*y for x, y in zip(rows[i], rows[j])]
    return [row[n:] for row in rows]


def _psd(a):
    """Exact symmetric Schur test, including singular positive semidefiniteness."""
    a = [list(row) for row in a]
    assert a == _transpose(a)
    for j in range(len(a)):
        pivot = a[j][j]
        if pivot < 0:
            return False
        if not pivot:
            if any(a[i][j] for i in range(j+1, len(a))):
                return False
            continue
        for i in range(j+1, len(a)):
            for k in range(j+1, len(a)):
                a[i][k] -= a[i][j]*a[j][k]/pivot
    return True


def certificate(operator, tangent=None):
    length = operator.length
    B = [list(row) for row in operator._B]
    A = _transpose([operator._basic_action([F(i == j) for i in range(6)])
                    for j in range(6)])
    R = [[F(0)]*6 for _ in range(6)] + _inverse([row[6:] for row in B])
    assert _product(B, R) == [[F(i == j) for j in range(6)] for i in range(6)]
    Z = [[F(0)]*6 for _ in range(12)]
    for end, x in ((0, -length/2), (6, length/2)):
        for axis in range(3):
            Z[end+axis][axis] = length
            Z[end+3+axis][3+axis] = 1
        Z[end+2][4] = -x
        Z[end+1][5] = x
    assert not any(v for row in _product(B, Z) for v in row)
    if tangent is None:
        tangent = operator.evaluate(np.zeros(12)).tangent
    K = [[F(float(v)) for v in row] for row in tangent]
    assert K == _transpose(K), "certificate never repairs nonsymmetry"
    pulled = _product(_product(_transpose(R), K), R)
    error = [[pulled[i][j]-A[i][j] for j in range(6)] for i in range(6)]
    energy_ok = all(_psd([[ENERGY_LIMIT*A[i][j]+sign*error[i][j]
                           for j in range(6)] for i in range(6)])
                    for sign in (-1, 1))
    # Physical reference scales, not candidate eigenvalues or raw mixed-unit norms.
    S = operator._stiffness
    force = max([S[i][i] for i in range(3)] +
                [S[i][i]/length**2 for i in range(3, 6)])
    energy = force*length
    wrench_scales = [force]*3+[force*length]*3
    KZ = _product(K, Z)
    rigid_wrench = max(sum(abs(v) for v in row)/wrench_scales[i % 6]
                       for i, row in enumerate(KZ))
    rigid_work = sum(abs(v) for row in _product(_transpose(Z), KZ)
                     for v in row)/energy
    # Scale the axial basic displacement by L; the other basic channels rotate.
    R_scaled = [[v*(length if j == 0 else 1) for j, v in enumerate(row)]
                for row in R]
    cross_work = sum(abs(v) for row in _product(
        _product(_transpose(Z), K), R_scaled) for v in row)/energy
    rigid_ok = rigid_wrench <= RIGID_LIMIT and rigid_work <= ENERGY_LIMIT
    cross_ok = cross_work <= ENERGY_LIMIT
    return {"basic_energy_ok": energy_ok, "rigid_ok": rigid_ok,
            "cross_ok": cross_ok, "all_ok": energy_ok and rigid_ok and cross_ok,
            "rigid_wrench_bound": float(rigid_wrench),
            "rigid_work_bound": float(rigid_work),
            "cross_work_bound": float(cross_work)}


def _cases():
    scalar = LinearSection.isotropic(E=210e9, nu=.3, area=.0032,
        Iy=.08*.04**3/12, Iz=.04*.08**3/12, J=.000001,
        ky=.8, kz=.7)
    yield "physical-scalar", LinearOperator(2., scalar), True
    for i, dims in enumerate((FIRST_M, *WIDTH_DEPTH_M)):
        section, _ = _section(*dims)
        yield f"physical-generalized-{i}", LinearOperator(1.5, section), True
    S = np.diag((22., 13., 17., 9., 7., 11.))
    S[0, 4] = S[4, 0] = 1.5
    yield "metre-generalized", LinearOperator(2.5, LinearSection.generalized(S)), True
    scale = np.diag((1., 1., 1., 1000., 1000., 1000.))
    yield "millimetre-generalized", LinearOperator(
        2500., LinearSection.generalized(scale@S@scale)), True
    stress = LinearSection.isotropic(E=1., nu=.3, area=2.**-80,
        Iy=1., Iz=1., J=1., ky=1., kz=1.)
    # Historical nonphysical stress family: observe, do not infer physical admission.
    yield "nonphysical-shear-soft-stress", LinearOperator(2., stress), None


def test_bounded_physical_output_certificate(tmp_path):
    results = {}
    for name, operator, expected in _cases():
        row = certificate(operator)
        row["expected_physical_pass"] = expected
        results[name] = row
    (tmp_path / "certificate.json").write_text(json.dumps(results, indent=2))
    assert all(row["all_ok"] for row in results.values()
               if row["expected_physical_pass"] is True), results


@pytest.mark.parametrize("defect", ("rigid", "cross", "basic"))
def test_certificate_detects_hidden_work_defects(defect):
    _, operator, _ = next(_cases())
    tangent = operator.evaluate(np.zeros(12)).tangent.copy()
    size = float(operator._stiffness[0][0]/operator.length)*1e-4
    if defect == "rigid":
        tangent[0, 0] += size  # R's first node is fixed, so basic block cannot see it.
    elif defect == "cross":
        tangent[0, 6] += size
        tangent[6, 0] += size
    else:
        tangent[6, 6] += size
    row = certificate(operator, tangent)
    assert not row["all_ok"]
    if defect in ("rigid", "cross"):
        assert row["basic_energy_ok"], "control must evade the basic-only check"
    else:
        assert not row["basic_energy_ok"]
