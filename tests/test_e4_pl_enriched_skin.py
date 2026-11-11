"""Development discriminators; no full pressure-stability/D1 acceptance claim."""
import numpy as np
import pytest

from examples.general_contact_enriched_patch import build
from anyfesolver.e4_pl_enriched import bubble_basis, compile_cell, EnrichedProfileError


@pytest.fixture(scope="module")
def compiled():
    return build(target_z=-.01)


def test_four_bubbles_have_zero_boundary_trace_and_independent_values():
    for t in (-1., -.3, 0., .4, 1.):
        for r, s in ((-1., t), (1., t), (t, -1.), (t, 1.)):
            assert np.array_equal(bubble_basis(r, s)[0], np.zeros(4))
    sample = np.array([bubble_basis(r, s)[0] for r, s in
                       ((-.4, -.3), (.4, -.3), (.4, .3), (-.4, .3))])
    assert np.linalg.matrix_rank(sample) == 4


def test_complete_stationary_functional_nodal_identity_and_all_bubble_energy(compiled):
    model, system = compiled
    from anyfesolver.e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
    assert type(model.mesh.elements[2]) is NativeParityE4PLS3V2DShellElement
    target_identity = next(row for row in system.definition["element_formulations"] if row["element_id"] == 2)
    assert target_identity["formulation_id"] == model.mesh.elements[2].formulation_id
    assert target_identity["class_module"] == NativeParityE4PLS3V2DShellElement.__module__
    cell = system.cells[0]
    element = model.mesh.elements[1]
    original = element.compute_stiffness_components(model.mesh, model.materials["elastic"])
    assert np.allclose(cell.stiffness[:24, :24], original["total"], rtol=1e-10, atol=1e-10)
    assert cell.stationary.shape == (43, 43) and cell.stiffness.shape == (28, 28)
    assert np.array_equal(cell.stationary, cell.stationary.T)
    # Independent compatible shear energy, using analytic gradients and the
    # physical isotropic section coefficient 5/6*G*t, not stationary recovery.
    shear = (5/6)*(1e5/(2*(1+.3)))*.05
    expected = np.zeros((4, 4))
    x, w = np.polynomial.legendre.leggauss(6)
    for r, wr in zip(x, w):
        for s, ws in zip(x, w):
            b = (1-r*r)*(1-s*s)
            br, bs = -2*r*(1-s*s), -2*s*(1-r*r)
            dr = np.array((br, b+r*br, s*br, s*b+r*s*br))*2
            ds = np.array((bs, r*bs, b+s*bs, r*b+r*s*bs))*2
            expected += .25*wr*ws*shear*(np.outer(dr, dr)+np.outer(ds, ds))
    assert np.allclose(cell.stiffness[24:, 24:], expected, rtol=1e-10, atol=1e-10)
    assert np.linalg.eigvalsh(expected)[0] > 0
    # The zero-trace enrichment does no work against a constant shear patch;
    # original nodal energy/patch behavior remains the actual qualified block.
    constant_shear = np.zeros(28)
    constant_shear[2:24:6] = cell.reference[:, 0]*1e-5
    assert np.max(np.abs((cell.stiffness @ constant_shear)[24:])) <= 1e-10
    bubble_only = np.r_[np.zeros(24), (1e-5, -2e-5, 3e-5, -1e-5)]
    recovered_fields = cell.recover_fields(bubble_only, .17, -.23)
    _, br, bs = bubble_basis(.17, -.23)
    assert np.allclose(recovered_fields["generalized_strain"][6:],
                       np.array((br, bs))*2 @ bubble_only[24:], rtol=1e-8, atol=1e-12)
    u = np.linspace(-2e-5, 3e-5, 28)
    recovered = cell.recover(u)
    assert np.linalg.norm(cell.stationary @ recovered+cell.coupling.T @ u) <= 1e-9
    assert np.isclose(cell.energy(u), .5*(cell.coupling.T @ u) @ recovered
                      + .5*u[:24] @ original["numerical"] @ u[:24], rtol=1e-10, atol=1e-16)


def test_rotated_cell_recovery_uses_same_global_coordinates_as_energy(compiled):
    from anyfesolver._ge_beam3_mixed_ad import rotation_exponential
    rotation = rotation_exponential((.21, -.17, .11))
    model, system = build(target_z=-.01, rotation=rotation)
    cell = system.cells[0]
    u = np.linspace(-2e-5, 3e-5, 28)
    a = cell.recover(u)
    assert np.linalg.norm(cell.stationary @ a+cell.coupling.T @ u) <= 1e-9
    flat = compiled[1].cells[0]
    transform = np.eye(28)
    for node in range(4):
        transform[6*node:6*node+3, 6*node:6*node+3] = rotation
        transform[6*node+3:6*node+6, 6*node+3:6*node+6] = rotation
    assert np.isclose(cell.energy(transform @ u), flat.energy(u), rtol=1e-10, atol=1e-14)


def test_rectangular_reference_and_immutable_energy_refuse_mutation(compiled):
    model, system = compiled
    with pytest.raises(ValueError):
        system.cells[0].stiffness[24, 24] = 0.
    with pytest.raises(EnrichedProfileError):
        system.cells[0].energy(np.full(28, np.nan))
    with pytest.raises(EnrichedProfileError, match="every assembled"):
        build(target_yield_stress=1.)
    with pytest.raises(EnrichedProfileError, match="every assembled"):
        build(target_hardening_curve=object())
    for unsupported in ({"director_polarity": -1}, {"reference_surface_offset": .001}):
        with pytest.raises(EnrichedProfileError, match="V2D elastic overlap"):
            build(target_options=unsupported)
    from anyfesolver.shell_sections import GeneralizedShellSection
    section = GeneralizedShellSection(A=np.eye(3), B=np.zeros((3, 3)), D=np.eye(3), As=np.eye(2))
    with pytest.raises(EnrichedProfileError, match="V2D elastic overlap"):
        build(target_options={"shell_section": section, "material_direction": (1., 0., 0.)})
    with pytest.raises(EnrichedProfileError, match="immutable"):
        system.stiffness = np.eye(system.offset.size)
    old = model.mesh.nodes[3].x
    model.mesh.nodes[3].x = old+.1
    try:
        with pytest.raises(EnrichedProfileError, match="affine rectangle"):
            compile_cell(model.mesh.elements[1], model.mesh, model.materials["elastic"])
    finally:
        model.mesh.nodes[3].x = old
