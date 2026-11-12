"""Enforced bounded output work; independent physical oracles remain separate."""
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest

import anyfesolver._b2_physical_flexibility_linear as local
from anyfesolver import stiffened_shell


def _section():
    return local.LinearSection.isotropic(
        E=210e9, nu=.3, area=.0032, Iy=.08*.04**3/12,
        Iz=.04*.08**3/12, J=.000001, ky=.8, kz=.7)


def test_certification_and_tangent_preparation_are_once_and_immutable(monkeypatch):
    calls = {"tangent": 0, "certificate": 0}
    original_tangent = local._nodal_tangent
    original_certificate = local._certify_output_tangent

    def tangent(operator):
        calls["tangent"] += 1
        return original_tangent(operator)

    def certificate(operator, emitted):
        calls["certificate"] += 1
        return original_certificate(operator, emitted)

    monkeypatch.setattr(local, "_nodal_tangent", tangent)
    monkeypatch.setattr(local, "_certify_output_tangent", certificate)
    operator = local.LinearOperator(2., _section())
    assert operator.output_certificate.passed
    assert operator.output_certificate.identity == local.OUTPUT_REPRESENTATION_ID
    with pytest.raises(FrozenInstanceError):
        operator.output_certificate.basic_energy_preserved = False
    with pytest.raises(AttributeError, match="immutable"):
        operator._output_certificate = None
    with pytest.raises(TypeError):
        operator._tangent[0][0] = 0.
    first = operator.evaluate(np.zeros(12))
    expected = first.tangent.copy()
    first.tangent[:] = 0.
    for i in range(10):
        trial = operator.evaluate(np.linspace(0., .001*(i+1), 12))
        np.testing.assert_array_equal(trial.tangent, expected)
        assert not np.shares_memory(first.tangent, trial.tangent)
    assert calls == {"tangent": 1, "certificate": 1}


@pytest.mark.parametrize("defect", ("basic", "rigid", "cross"))
def test_injected_emission_defects_are_rejected_before_an_operator_exists(monkeypatch, defect):
    original = local._nodal_tangent

    def broken(operator):
        tangent = np.array(original(operator))
        size = float(operator._stiffness[0][0]/operator.length)*1e-4
        if defect == "basic":
            tangent[6, 6] += size
        elif defect == "rigid":
            tangent[0, 0] += size
        else:
            tangent[0, 6] += size
            tangent[6, 0] += size
        return tangent

    monkeypatch.setattr(local, "_nodal_tangent", broken)
    with pytest.raises(local.OutputRepresentationError, match="output representation") as caught:
        local.LinearOperator(2., _section())
    certificate = caught.value.certificate
    assert not certificate.passed
    if defect in ("rigid", "cross"):
        assert certificate.basic_energy_preserved
    else:
        assert not certificate.basic_energy_preserved


def test_nonfinite_or_asymmetric_emission_is_rejected_without_symmetrization(monkeypatch):
    original = local._nodal_tangent

    def broken(operator):
        tangent = np.array(original(operator))
        tangent[0, 1] = 1.
        return tangent

    monkeypatch.setattr(local, "_nodal_tangent", broken)
    with pytest.raises(local.RepresentabilityError, match="finite symmetric"):
        local.LinearOperator(2., _section())


def test_exact_semidefinite_work_bounds_handle_singular_and_indefinite_cases():
    assert local._positive_semidefinite([[F(1), F(1)], [F(1), F(1)]])
    assert local._positive_semidefinite([[F(0), F(0)], [F(0), F(1)]])
    assert not local._positive_semidefinite([[F(0), F(1)], [F(1), F(1)]])
    assert not local._positive_semidefinite([[F(1), F(2)], [F(2), F(1)]])


def test_public_compiled_member_cannot_substitute_an_uncertified_operator():
    member = stiffened_shell.PhysicalB2Member(
        "member", "source", (0, 1), ((0., 0., 0.), (2., 0., 0.)),
        (0., 1., 0.), "section", _section())
    compiled = member.compile_local()
    assert isinstance(compiled.operator.output_certificate,
                      stiffened_shell.OutputRepresentationCertificate)
    assert (compiled.operator.output_certificate.identity ==
            stiffened_shell.B2_BOUNDED_OUTPUT_REPRESENTATION_V1)
    assert stiffened_shell.OutputRepresentationError is local.OutputRepresentationError
    unguarded = SimpleNamespace(section=member.section, length=F(2))
    with pytest.raises(ValueError, match="operator/definition mismatch"):
        stiffened_shell.CompiledPhysicalB2Member(
            member, unguarded, compiled._frame_bytes, compiled._transform_bytes)
