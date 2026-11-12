"""Exact tilted chart and explicit manufactured constant equilibrium load.

Load values are constructed from production structural force: this is a
self-equilibrated integration fixture, not an independent constitutive oracle.
"""
from fractions import Fraction as Q
import numpy as np
def _exact_float(x):
    value = float(x)
    assert Q(value) == x, "fixture chart is not exactly representable"
    return value

def build_pose():
    from anyfesolver import (BoundaryCondition, ContactDefinition, ContactSurface,
                           ContactInteraction, ContactSolverSettings,
                           create_shell_element)
    from anyfesolver.elements import QuadraticBeamElement
    from anyfesolver.fe_core import FEModel
    m, delta, height, radius = 256, Q(1, 16384), Q(1, 128), Q(1, 128)
    X, Y, rise = (m*m-1)*delta, Q(4), 2*m*delta
    stations = (Q(1, 4), Q(5, 16), Q(3, 8))
    beam_x = [-1+t*X-radius/m for t in stations]
    refs = {1: (beam_x[0], Q(0), Q(1, 32)),
            6: (beam_x[1], Q(0), Q(1, 32)),
            2: (beam_x[2], Q(0), Q(1, 32)),
            3: (Q(-1), Q(-1), Q(0)),
            4: (-1+X, Q(-1), Q(0)),
            5: (Q(-1), Q(3), Q(0))}
    model = FEModel("MANUFACTURED_equilibrated_tilted_touch")
    model.add_material("beam", 1.e3, .3, density=1.)
    model.add_material("skin", 1.e5, .3, density=1.)
    for node, point in refs.items():
        model.add_node(node, *map(_exact_float, point))
    model.add_element(1, QuadraticBeamElement(1, [1, 6, 2], "beam",
        dict(area=1.e-3, Iy=1.e-4, Iz=1.e-4, J=1.e-4)))
    model.add_element(2, create_shell_element(2, (3, 4, 5), "skin",
        thickness=float(2*height), formulation="e4-pl-s3-v2d",
        reference_normal=(0., 0., 1.), reference_surface_offset=0.))
    model.add_boundary_condition(BoundaryCondition("beam_guides", [1, 6, 2],
        dict.fromkeys(("ux", "uy", "rx", "rz"), 0.)))
    model.add_boundary_condition(BoundaryCondition("skin_guides", [3, 4, 5],
        dict.fromkeys(("ux", "uy", "rz"), 0.)))
    dofs = {node: model.mesh.dof_manager.get_node_dofs(node) for node in refs}
    for axis in (3, 4):
        for node in (4, 5):
            model.add_constraint_equation(
                terms=((dofs[node][axis], 1.), (dofs[3][axis], -1.)),
                dependent_dof=dofs[node][axis],
                source_id=f"uniform_skin_rotation_{axis}_{node}")
    definition = ContactDefinition(
        (ContactSurface("tube", (1,), beam_radius=float(radius)),
         ContactSurface("skin", (2,), side="top")), (),
        (ContactInteraction("line", "tube", "skin", None, 1.e-5,
                            line_stiffness=1.e5),))
    settings = ContactSolverSettings(pair_motion="general",
        beam_shell_rim_clearance=.02, line_integration_relative_tolerance=1.e-8,
        beam_shell_geometry="jet2_deformable_s3_dev",
        beam_shell_maximum_swept_intervals=128)
    u = np.zeros(model.mesh.dof_manager.total_dofs)
    u[dofs[4][2]] = float(rise)
    for node, t in zip((1, 6, 2), stations):
        u[dofs[node][2]] = _exact_float(height+radius+t*rise-Q(1, 32))
    return model, definition, settings, u, refs, dofs, height, radius, X, Y, rise


def prepare():
    from anyfesolver import LoadCase
    from anyfesolver.assembly import assemble_mass_matrix,build_constraint_transformation
    from anyfesolver.matrix_assembly import assemble_stiffness_matrix
    from anyfesolver.algebraic_dynamics import build_declared_algebraic_basis
    from anyfesolver.nonlinear_static import _assemble_nonlinear_system
    from anyfesolver.nonlinear_state import discard_active_state_candidate
    from anyfesolver.general_contact.runtime import suspend_contact
    from anyfesolver.general_contact.beam_s3_directional_rates import DirectionalBeamS3Manager,directional_coefficients,solve_directional_velocity,exact_gap_polynomial
    m,d,s,u,refs,dofs,h,rho,X,Y,rise=build_pose();m.apply_boundary_conditions()
    K0,_=assemble_stiffness_matrix(m);M,_=assemble_mass_matrix(m)
    _,_,T,offset,independent,_=build_constraint_transformation(K0,np.zeros(len(u)),m)
    Mr=(T.T@M@T).tocsr();basis=build_declared_algebraic_basis(m,M,Mr,T,independent,dense_size_limit=512)
    manager=DirectionalBeamS3Manager(m,d,s,initial_displacement=u);origin={}
    try:
        with suspend_contact():f,K,_=_assemble_nonlinear_system(m,u,origin,3,tangent=True,kinematics='von_karman')
    finally:discard_active_state_candidate(origin)
    fc,H,summary=manager.evaluate(u)
    load=LoadCase('manufactured_constant_equilibrium')
    for node,ids in dofs.items():load.add_nodal_load(node,forces=tuple((f+fc)[ids[:3]]),moments=tuple((f+fc)[ids[3:]]))
    B,factor=directional_coefficients(manager,u);Z=basis.reduced_basis.toarray()
    physical=np.zeros(len(independent));vfull=np.zeros(len(u))
    for node,value in ((1,-.01),(6,-.012),(2,-.01)):vfull[dofs[node][2]]=value
    physical=np.linalg.lstsq(T.toarray(),vfull,rcond=None)[0]
    physical-=Z@np.linalg.solve(Z.T@Z,Z.T@physical)
    reduced=np.array(B@np.array([[Q(float(x)) for x in row] for row in T.toarray()],object),object)
    velocity=solve_directional_velocity((T.T@K@T).toarray(),Z,physical,reduced,factor)
    full_velocity=np.asarray(T@velocity).ravel()
    initial=dict(exact_touch=bool(exact_gap_polynomial(manager,u)[0]),contact_null_norm=float(np.linalg.norm(np.array(B,float)@T@Z)),mass_nullity=Z.shape[1],stored_prestrain_half_force_work_heuristic_j=float(.5*u@f),positive_impact_kinetic_scale_j=float(.5*full_velocity@(M@full_velocity)),pose=u.tolist(),velocity=full_velocity.tolist(),load_forces=(f+fc).tolist(),equilibrium_load_construction='PRODUCTION_FORCE_MANUFACTURED_NOT_INDEPENDENT_CONSTITUTIVE_ORACLE')
    return m,d,s,u,full_velocity,load,initial



def run(*,dt=1.e-5,end=3.e-5,restart=None):
    from anyfesolver.general_contact.transient import solve_transient_contact,ContactTransientSolverControls
    from anyfesolver.dynamics import TransientConfig
    m,d,s,u,v,load,initial=prepare()
    result=solve_transient_contact(m,d,TransientConfig(dt,end,initial_displacement=u,initial_velocity=v),base_load_case=load,contact_settings=s,
        solver_controls=ContactTransientSolverControls(allow_equilibrated_directional_beam_s3_dev=True,save_constraint_reactions_dev=True),
        restart_checkpoint=restart,emit_restart_checkpoint=True)
    return m,result,initial



if __name__=='__main__':
    _,result,initial=run()
    print(dict(profile=result.profile,version=result.restart_checkpoint['version'],states=len(result.times),contact_null_coupling=initial['contact_null_norm'],minimum_gap=min(c.minimum_gap for c in result.contact)))
