"""Actual classical quadratic beam/deforming S3 coupled development control."""
import json
import runpy
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
from anyfesolver import BoundaryCondition, LoadCase, solve_static_nonlinear
from anyfesolver.elements import QuadraticBeamElement
from anyfesolver.matrix_assembly import assemble_stiffness_matrix
from anyfesolver.general_contact.deformable_beam_s3_static import DeformableBeamS3Manager
from s3_von_karman_oracle import model_triangle_von_karman


def independent_gradient(x):
    tip, middle, w = x
    area = np.sqrt(9 + w*w)
    # Actual node order start/mid/end; fixed in-plane translations.
    def shape(t):
        return t*(2*t-1), 4*t*(1-t)
    def gap(t):
        nt, nm = shape(t)
        return (3*(.01005 + nt*tip + nm*middle)-w)/area - .01
    values = np.array([gap(t) for t in (0., .5, 1.)])
    c2 = 2*(values[0] - 2*values[1] + values[2])
    c1 = values[2] - values[0] - c2
    coeff = np.trim_zeros(np.array((values[0], c1, c2)), "b")
    roots = np.polynomial.polynomial.polyroots(coeff) if len(coeff)>1 else []
    splits = sorted(float(z.real) for z in roots if abs(z.imag)<1e-10 and 0<z.real<1)
    def component(t, i):
        nt, nm = shape(t)
        n = 3*(.01005 + nt*tip + nm*middle)-w
        g = n/area-.01
        grad = (3*nt/area, 3*nm/area, -1/area-n*w/area**3)
        return 1e5*.5*g*grad[i] if g<0 else 0.
    return np.array([quad(lambda t:component(t,i),0.,1.,points=splits,epsabs=1e-12,epsrel=1e-12)[0] for i in range(3)])


def test_actual_quadratic_beam_coupling_and_restart():
    example = runpy.run_path("examples/general_contact_deformable_round_beam_s3.py")
    m, definition, settings = example["fixture"]()
    m.add_node(6,.25,0.,.02005)
    section = dict(m.mesh.elements[1].cross_section)
    m.mesh.elements[1] = QuadraticBeamElement(1,[1,6,2],"beam",section)
    example["constraints"](m)
    m.add_boundary_condition(BoundaryCondition("midpoint_guides",[6],dict.fromkeys(("ux","uy","rx","ry","rz"),0.)))
    load = LoadCase("press");load.add_nodal_load(2,forces=(0.,0.,-.001))
    controls = dict(contact=definition,contact_settings=settings,num_steps=4,max_iterations=20,
                    tolerance=1e-10,kinematics="von_karman",emit_restart_checkpoint=True)
    full = solve_static_nonlinear(m,load,**controls)
    assert full.status == "completed", full.failure_reason
    indices = np.array([m.mesh.dof_manager.get_node_dofs(n)[2] for n in (2,6,5)])
    x = full.displacements[indices]
    # A genuinely quadratic field, not the B2 interpolant under another label.
    assert abs(x[1]-.5*x[0])/max(abs(x[0]),abs(x[1])) > 1e-4
    assert x[2] < -1e-7
    manager = DeformableBeamS3Manager(m,definition,settings)
    r, _, _ = manager.evaluate(full.displacements)
    np.testing.assert_allclose(r[indices],independent_gradient(x),rtol=1e-9,atol=1e-12)
    stiffness,_ = assemble_stiffness_matrix(m)
    k = stiffness[indices][:,indices].toarray()
    force = np.array((-.001,0.,0.))
    # Retarget to the adopted S3 VK law (precedent 706bf011c): add the
    # independent membrane VK force of the skin (nodes 3,4 fixed, node 5 only
    # uz free; fixture constants E=1e5, nu=.3, t=.02) to the linear structure.
    def membrane(y):
        return np.array((0.,0.,model_triangle_von_karman(m,(3,4,5),(0.,0.,1.),(0.,0.,y[2]),1.e5,.3,.02)[1][2]))
    reference = root(lambda y:k@y+membrane(y)+independent_gradient(y)-force,x,tol=1e-11)
    assert reference.success
    relative = np.linalg.norm(reference.x-x)/np.linalg.norm(reference.x)
    assert relative < 1e-3  # Retained bounded structural response gate (linear + VK membrane).
    assert full.info["general_contact"]["gap_tolerance_met"]
    half = solve_static_nonlinear(m,load,max_load_factor=.5,**controls)
    assert half.status == "completed",half.failure_reason
    saved = json.loads(json.dumps(half.to_restart_checkpoint()))
    resumed = solve_static_nonlinear(m,load,restart_checkpoint=saved,**controls)
    assert resumed.status == "completed",resumed.failure_reason
    np.testing.assert_allclose(resumed.displacements,full.displacements,rtol=1e-7,atol=1e-10)
    print("B3_DEFORMABLE_S3_RESPONSE "+json.dumps(dict(coordinates=x.tolist(),
        von_karman_reference=reference.x.tolist(),relative_response=relative,contact=full.info["general_contact"]),sort_keys=True))
