"""No generic pressure or couple fallback for the private beam candidate."""
import numpy as np
import pytest
from anyfesolver.boundary import LoadCase
from anyfesolver.matrix_assembly import assemble_load_vector,assemble_external_load_tangent
from anyfesolver.nonlinear_static import solve_static_nonlinear
from test_ge_beam3_native_fibre_connected_restart import make
from test_ge_beam3_schur_line_program import save


@pytest.mark.parametrize('route',['direct','assembly','tangent','solver'])
@pytest.mark.parametrize('kind',['pressure','follower','moment','element','gravity','extra-schema'])
def test_unsupported_loads_rejected_before_mechanics(route,kind,monkeypatch,tmp_path):
    m,_=make('curved-2-elastic-clamped');load=LoadCase('unsupported')
    if kind=='pressure':load.add_pressure_load(1,1.)
    elif kind=='follower':load.follower_pressure=True;load.add_pressure_load(1,1.)
    elif kind=='moment':load.add_nodal_load(3,moments=np.array([0.,0.,.1]))
    elif kind=='element':load.element_loads[1]=np.zeros(18)
    elif kind=='gravity':load.gravity=np.array([0.,0.,-9.81])
    else:load.line_forces={1:(0.,0.,1.)}
    def forbidden(*a,**k):raise AssertionError('unsupported load reached element mechanics')
    for e in m.mesh.elements.values():
        monkeypatch.setattr(e,'compute_stiffness_matrix',forbidden)
        monkeypatch.setattr(e,'compute_nonlinear_response',forbidden)
        monkeypatch.setattr(e,'compute_mass_matrix',forbidden)
    with pytest.raises(ValueError,match='private native beam'):
        if route=='direct':load.get_load_vector(m.mesh,m.mesh.dof_manager,m.get_material)
        elif route=='assembly':assemble_load_vector(m,load)
        elif route=='tangent':assemble_external_load_tangent(m,load)
        else:solve_static_nonlinear(m,load,num_steps=1,max_iterations=1,num_layers=1)
    save(tmp_path/'rejected.json',dict(route=route,kind=kind,rejected_before_mechanics=True))


def test_admitted_nodal_force_and_exact_zero_tangent(tmp_path):
    m,_=make('curved-2-elastic-clamped');load=LoadCase('nodal-only');load.add_nodal_load(3,forces=np.array([.1,-.2,.3]))
    direct=load.get_load_vector(m.mesh,m.mesh.dof_manager,m.get_material);assembled,_=assemble_load_vector(m,load);tangent,_=assemble_external_load_tangent(m,load)
    expected=np.zeros(30);expected[12:15]=(.1,-.2,.3)
    np.testing.assert_array_equal(direct,expected);np.testing.assert_array_equal(assembled,expected);assert tangent.nnz==0
    save(tmp_path/'admitted.json',dict(force=direct,tangent_nonzeros=tangent.nnz,production_qualified=False))
