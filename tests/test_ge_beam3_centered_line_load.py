"""Small work/sensitivity checks before native distributed-load integration."""

import ast
from pathlib import Path

import numpy as np
import pytest

from docs.reference_cases import ge_beam3_centered_line_load_oracle as oracle
from docs.reference_cases import ge_beam3_centered_line_load_probe as probe
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Centered
from anyfesolver._ge_beam3_p5_coordinates.positions import from_total
from anyfesolver._ge_beam3_p5.core import canonical
from anyfesolver._ge_beam3_p5.algebra import rotation
from anyfesolver._ge_beam3_p5.section import SectionHistory, DirectedHardeningSection
from test_ge_beam3_centered_reference import fixture
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law
from test_ge_beam3_curved_p5_native_chart_probe import sample


FORCE = np.array([.03,-.02,.01])


def setup(*, curved=True, plastic=False, shift=0.):
    if curved:
        nodes, frames = fixture(translation=shift, spatial=True)
    else:
        nodes = np.array([[-1.,0.,0.],[0.,0.,0.],[1.,0.,0.]])+shift
        frames = np.tile(np.eye(3),(3,1,1))
    ref = Centered(nodes, frames); total = sample()
    high, low = from_total(nodes, total, nodes+total.reshape(3,6)[:,:3])
    vertex = np.array([rotation(v)@q for v,q in zip(total.reshape(3,6)[:,3:], frames)])
    origins = tuple(SectionHistory(.0001*i,.0002*i) for i in range(16))
    material = law(.02 if plastic else 1000.)
    section = DirectedHardeningSection(material._elastic,material._direction,material._yield,material._hardening)
    return ref, section, high, low, vertex, total, origins


def run(data, parameter=.7):
    ref, section, high, low, vertex, _, origins = data
    return probe.evaluate(ref, section, high, low, vertex, FORCE, parameter, order=8, origins=origins)


def close(actual, expected, tolerance=1e-11):
    assert np.linalg.norm(np.asarray(actual)-expected) <= tolerance*max(1.,np.linalg.norm(expected))


@pytest.fixture(scope='module', params=[(False,False),(False,True),(True,False),(True,True)])
def solved(request):
    curved, plastic = request.param
    data = setup(curved=curved,plastic=plastic); result = run(data)
    assert any(result.branch) == plastic
    return data, result, curved, plastic


def test_full_work_force_and_hessian_match_independent_reconstruction(solved):
    data, result, _, _ = solved
    ref, _, _, _, _, total, _ = data
    expected = oracle.evaluate(ref.coordinates,total.reshape(3,6)[:,:3],result.response.local_rotations,FORCE)
    close(result.unit_work,expected.work)
    close(result.augmented_unit_force,expected.force)
    close(result.augmented_unit_work_hessian,expected.hessian)
    assert np.count_nonzero(expected.force[24:]) == 0  # Multipliers carry no external work.
    assert np.count_nonzero(expected.force[np.array([3,4,5,9,10,11,15,16,17])]) == 0


def test_load_only_derivatives_match_the_frozen_complete_variational_core(solved):
    from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
    data, result, _, _ = solved; ref, section, high, low, vertex, _, origins = data
    full = []
    for value in (0.,1.):
        model = CenteredStationaryBeam(ref,section,position_low=low,order=8,origins=origins,line_force=value*FORCE)
        full.append(model.evaluate(high,vertex,result.response.local_rotations,result.response.moments))
    close(full[0].potential-full[1].potential,result.unit_work)
    close(full[0].residual-full[1].residual,result.augmented_unit_force)
    close(full[0].hessian-full[1].hessian,result.augmented_unit_work_hessian)


def test_load_condensation_has_required_internal_work_contribution(solved):
    _, result, curved, _ = solved
    naive = -result.augmented_unit_force[:18]
    if curved:
        assert np.linalg.norm(result.internal_parameter_derivative[:6]) > 1e-5
        assert np.linalg.norm(result.residual_parameter_derivative-naive) > 1e-5
        assert result.potential_parameter_second_derivative < -1e-7
    else:
        close(result.internal_parameter_derivative,np.zeros(18))
        close(result.residual_parameter_derivative,naive)
        assert abs(result.potential_parameter_second_derivative) <= 1e-11


def test_condensed_load_parameter_derivatives_with_fixed_section_origins(solved):
    data, base, _, _ = solved; before = canonical(data[-1]); step = 1e-5
    plus = run(data,.7+step); minus = run(data,.7-step)
    assert plus.branch == base.branch == minus.branch  # No nonsmooth crossing claim.
    close((plus.response.residual-minus.response.residual)/(2*step),base.residual_parameter_derivative,1e-7)
    close((plus.response.potential-minus.response.potential)/(2*step),base.potential_parameter_derivative,1e-7)
    close((plus.potential_parameter_derivative-minus.potential_parameter_derivative)/(2*step),
          base.potential_parameter_second_derivative,1e-7)
    assert canonical(data[-1]) == before


@pytest.mark.parametrize('plastic',[False,True])
def test_condensed_dead_line_derivatives_are_byte_identical_under_exact_translation(plastic):
    records = []
    for shift in (0.,2.**30,2.**40):
        records.append(canonical(run(setup(plastic=plastic,shift=shift))))
    assert records[0] == records[1] == records[2]


def test_independent_work_rotation_covariance_and_noncommuting_cell_spins():
    data = setup(); ref, _, _, _, _, total, _ = data
    cells = np.array([rotation([.4,-.2,.3]),rotation([-.1,.35,.2])])
    original = oracle.evaluate(ref.coordinates,total.reshape(3,6)[:,:3],cells,FORCE)
    g = rotation([1.2,-.7,.8])
    moved = oracle.evaluate(ref.coordinates@g.T,total.reshape(3,6)[:,:3]@g.T,
        g@cells@g.T,g@FORCE)
    transform = np.eye(36)
    for slot in range(0,24,3): transform[slot:slot+3,slot:slot+3] = g
    close(moved.work,original.work)
    close(moved.force,transform@original.force)
    close(moved.hessian,transform@original.hessian@transform.T)


def test_analytic_load_rotation_variations_and_rigid_force_resultant():
    data = setup(); ref, _, _, _, _, total, _ = data
    cells = np.array([rotation([.4,-.2,.3]),rotation([-.1,.35,.2])])
    d = np.sin(np.arange(36)+.3)/5; step = 1e-5
    def at(amount):
        shift = total.reshape(3,6)[:,:3]+amount*d[:18].reshape(3,6)[:,:3]
        u = np.array([rotation(amount*d[18+3*c:21+3*c])@cells[c] for c in (0,1)])
        return oracle.evaluate(ref.coordinates,shift,u,FORCE)
    base = at(0.); plus = at(step); minus = at(-step)
    close((plus.work-minus.work)/(2*step),base.force@d,1e-7)
    close((plus.work-2*base.work+minus.work)/step**2,d@base.hessian@d,1e-7)
    nodal_resultant = base.force[:18].reshape(3,6)[:,:3].sum(axis=0)
    close(nodal_resultant,base.nodal_measures.sum()*FORCE)


@pytest.mark.parametrize('scale',[0.,1e-20,1.])
def test_load_only_variations_retain_tiny_work_without_material_subtraction(scale):
    data = setup(shift=2.**40); ref, _, high, low, _, total, _ = data
    cells = np.array([rotation([.4,-.2,.3]),rotation([-.1,.35,.2])])
    expected = oracle.evaluate(ref.coordinates,total.reshape(3,6)[:,:3],cells,FORCE*scale)
    actual = probe.load_work(ref,high,low,cells,FORCE*scale)
    for a,b in ((actual.work,expected.work),(actual.force,expected.force),(actual.hessian,expected.hessian)):
        if scale == 0.: assert not np.any(a)
        else: assert np.linalg.norm(np.asarray(a)-b) <= 1e-11*np.linalg.norm(b)


def test_material_potential_subtraction_would_erase_tiny_line_work():
    from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
    data = setup(shift=2.**40); ref, section, high, low, vertex, _, origins = data
    cells = np.array([rotation([.4,-.2,.3]),rotation([-.1,.35,.2])])
    moments = np.zeros((2,2,3)); tiny = FORCE*1e-20
    totals = [CenteredStationaryBeam(ref,section,position_low=low,order=8,origins=origins,
        line_force=f).evaluate(high,vertex,cells,moments).potential for f in (np.zeros(3),tiny)]
    assert totals[0]-totals[1] == 0.
    direct = probe.load_work(ref,high,low,cells,tiny)
    assert direct.work != 0. and np.linalg.norm(direct.force) > 0.


@pytest.mark.parametrize('mutation',['nonfinite','shape','improper_rotation','order'])
def test_oracle_rejects_invalid_load_inputs(mutation):
    nodes, _ = fixture(); u = np.zeros((3,3)); cells = np.tile(np.eye(3),(2,1,1)); force = FORCE.copy(); order = 8
    if mutation == 'nonfinite': force[0] = np.nan
    if mutation == 'shape': u = np.zeros(18)
    if mutation == 'improper_rotation': cells[0,0,0] = -1.
    if mutation == 'order': order = 12
    with pytest.raises(ValueError): oracle.evaluate(nodes,u,cells,force,order=order)


def test_oracle_imports_no_producer_mechanics_or_automatic_differentiation():
    tree = ast.parse(Path(oracle.__file__).read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(n.name.split('.')[0] for n in node.names)
        if isinstance(node,ast.ImportFrom): imports.add(node.module.split('.')[0])
    assert imports <= {'dataclasses','fractions','math','numpy'}
