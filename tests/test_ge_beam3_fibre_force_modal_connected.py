"""Connected curved native force owners retain the complete physical pencil."""
import numpy as np
from scipy.linalg import eigh
from anyfesolver._ge_beam3_native_analysis import NativeBeamAnalysis
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._ge_beam3_native_fibre_static_element import NativeFibreStaticElement
from anyfesolver._ge_beam3_p5_seeded.core import canonical
from test_ge_beam3_retained_fibre import make_model
from test_ge_beam3_native_fibre_current_modal import decode


def test_two_connected_force_owned_cells_match_full_stationary_system(tmp_path):
    source=make_model();definitions=tuple(NativeBeamDefinition.capture(
        NativeFibreStaticElement(i,tuple(e.node_ids),e.operator.reference,e.section,order=4),
        np.diag([2.,2.,2.,.07,.09,.11])) for i,e in sorted(source.mesh.elements.items()))
    made=NativeBeamAnalysis(definitions,tuple(source.boundary_conditions))
    assert len(made._elements)==2
    run=made.solve_nodal(((5,.004,-.0002,.0001),),steps=1)
    assert run.status=='completed',run.backend_result.info
    state,external=decode(made,run);before=canonical(state)
    packet,modes=made.current_modes(run.checkpoint,expected_sha256=run.checkpoint_sha256,
        bounds=(-100.,1e6),num_modes=6)
    nodal=made.model.mesh.dof_manager.total_dofs;n=len(made._elements)
    full=np.zeros((nodal+24*n,nodal+24*n));force=np.zeros(len(full))
    kinematics=list(range(nodal));stresses=[]
    for index,e in enumerate(made._elements):
        internal=list(range(nodal+24*index,nodal+24*(index+1)))
        slots=list(e.get_dof_mapping(made.model.mesh))+internal
        response=state['states'][e.element_id]['response']
        full[np.ix_(slots,slots)]+=response.full_hessian
        force[slots]+=response.full_residual
        kinematics.extend(internal[:6]);stresses.extend(internal[6:])
    expected=full[np.ix_(kinematics,kinematics)]-full[np.ix_(kinematics,stresses)]@np.linalg.solve(
        full[np.ix_(stresses,stresses)],full[np.ix_(stresses,kinematics)])
    error=float(np.linalg.norm(expected-packet.base.stiffness)/max(1.,np.linalg.norm(expected)))
    assert error<=1e-11 and packet.base.mass.shape==(42,42)
    free=list(packet.base.free_dofs);trace=list(packet.base.algebraic_dofs);physical=[i for i in free if i not in trace]
    assert len(physical)==24 and len(trace)==12
    # Constrained residuals are reactions; equilibrium is required on free DOFs.
    assert np.linalg.norm((force[kinematics]-np.r_[external,np.zeros(6*n)])[free])<=1e-11
    reactions=(force[:nodal]-external).reshape(-1,6)[:,:3].sum(axis=0)
    assert np.linalg.norm(reactions+external.reshape(-1,6)[:,:3].sum(axis=0))<=1e-11
    stiffness=packet.base.stiffness;mass=packet.base.mass
    reduced=stiffness[np.ix_(physical,physical)]-stiffness[np.ix_(physical,trace)]@np.linalg.solve(
        stiffness[np.ix_(trace,trace)],stiffness[np.ix_(trace,physical)])
    expected_modes=eigh(reduced,mass[np.ix_(physical,physical)],eigvals_only=True)[:6]
    np.testing.assert_allclose(modes.eigenvalues,expected_modes,atol=1e-11,rtol=1e-11)
    assert np.all(modes.eigenvalues>0.) and canonical(state)==before
    (tmp_path/'connected-checkpoint.json').write_bytes(run.checkpoint)
    (tmp_path/'connected-modal.json').write_bytes(canonical(dict(packet=packet,modes=modes,schur_error=error)))
