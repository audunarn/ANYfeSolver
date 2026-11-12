"""Geometry/admission only; no E04 equilibrium, limit-load or external pass."""
import numpy as np
import pytest
from docs.reference_cases.stiffened_shell_e04_roof import build


@pytest.mark.parametrize('size',(8,16,32))
def test_roof_reference_geometry_and_measures(size):
    c=build(size,size)
    assert len(c.quads)==size*size and len(c.beam_cells)==size
    np.testing.assert_allclose(c.beam_frames.transpose(0,2,1)@c.beam_frames,np.broadcast_to(np.eye(3),c.beam_frames.shape),atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(c.beam_frames),1.,atol=1e-15)
    for q in c.quads:
        p=c.shell_positions[q];cross=np.cross(p[1]-p[0],p[3]-p[0])
        assert abs(np.dot(cross,p[2]-p[0]))<1e-18
        assert np.dot(cross,c.shell_directors[q].mean(axis=0))>0.
    s,b=c.point_pairs.T
    np.testing.assert_allclose(c.beam_positions[b]-c.shell_positions[s],
        (c.thickness+c.bar_height)/2*c.shell_directors[s],atol=1e-16)
    assert c.beam_length==pytest.approx(2*c.length)
    assert c.shell_area==pytest.approx(c.length*size*2*c.radius*np.sin(c.half_angle/size))
    n=len(c.shell_positions)
    assert c.metric_weights[:n].sum()==pytest.approx(.5)
    assert c.metric_weights[n:].sum()==pytest.approx(.5)
    assert np.all(c.metric_weights>0.)
    assert c.loaded_shell_node not in c.fixed_shell_translations
    np.testing.assert_allclose(c.shell_positions[c.loaded_shell_node,:2],0.,atol=1e-15)
    with pytest.raises(ValueError):c.shell_positions.setflags(write=True)
    # The present owner must keep rejecting a curved reference until the local
    # frame/pullback/attachment extension is implemented and verified.
    from anyfesolver._stiffened_shell_ge_experimental import _shell_reference,ShellPanel
    panel=ShellPanel(node_ids=tuple(range(1,n+1)),coordinates_m=c.shell_positions,
        q4_elements=tuple(tuple(int(i)+1 for i in q) for q in c.quads),
        young_pa=c.young,poisson=c.poisson,thickness_m=c.thickness,density_kg_m3=1.)
    with pytest.raises(ValueError,match='flat'):_shell_reference(panel)


def test_reference_metric_converges_for_affine_physical_translation():
    # Area-lumped shell trapezoids converge quadratically; quadratic beam
    # Simpson integration is exact for x**2. The shared 1/2 domain weighting is
    # fixed across meshes, and introduces no inertia or algebraic-variable norm.
    errors=[]
    for size in (8,16,32):
        c=build(size,size)
        x=np.r_[c.shell_positions[:,0],c.beam_positions[:,0]]
        measured=float(c.metric_weights@(x*x))
        exact=c.length**2/12.
        assert measured==pytest.approx(exact*(1+1/size**2),rel=1e-13)
        errors.append(measured-exact)
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:],4.,rtol=1e-10)


def test_roof_public_definition_preserves_geometry_and_centroidal_section():
    from docs.reference_cases.stiffened_shell_e04_roof import definition
    from anyfesolver import stiffened_shell as ss
    c=build(4,4);d=definition(c,density_kg_m3=1.)
    assert d.shell_reference_policy==ss.CURVED_Q4_POINT_REFERENCE_V1
    assert len(d.points)==len(c.point_pairs)
    assert len(d.fixed)==3*len(c.fixed_shell_translations)
    element,inertia=d.beams[0].instantiate()
    np.testing.assert_array_equal(element.operator.reference.coordinates,c.beam_positions[c.beam_cells[0]])
    np.testing.assert_array_equal(element.operator.reference.nodal_triads,c.beam_frames[c.beam_cells[0]])
    assert inertia[0,0]==pytest.approx(c.bar_width*c.bar_height)
    assert inertia[4,4]==pytest.approx(c.bar_width*c.bar_height**3/12.)
    assert inertia[5,5]==pytest.approx(c.bar_height*c.bar_width**3/12.)
    # Repeated construction owns identical semantics; density is not guessed.
    assert d.beams[0].sha256==definition(c,density_kg_m3=1.).beams[0].sha256
    with pytest.raises(ValueError,match='density'):definition(c,density_kg_m3=0.)
