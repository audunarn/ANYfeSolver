"""Plan S3-R1(e) regression guards; fixture copied unchanged from evidence/plate.py.
Run separately with s3-base-run for protected hashes. These tests do not qualify release.
"""
import hashlib
import pytest
import sys,json,time,warnings,numpy as np
# Locally label these expensive tests; this frozen tree has no registered slow marker.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', pytest.PytestUnknownMarkWarning)
    slow = pytest.mark.slow
import anyfesolver as fs
from anyfesolver import _compensated_native as native_contractions
from anyfesolver import _compensated_contractions as contractions
from anyfesolver.elements import create_shell_element
from anyfesolver.boundary import FixedSupport,LoadCase
from anyfesolver.material_curves import DNVC208MaterialCurve
def plate(c):
    m=fs.FEModel('p');n=c['n'];t=c['t'];form=c['form']
    curve=DNVC208MaterialCurve(sigma_prop=3e7,sigma_yield=3.2e7,sigma_yield_2=3.4e7,eps_p_y1=.005,eps_p_y2=.01,K=4e8,n=.2)
    m.add_material('mat',210e9,.3,density=7850.,yield_stress=3e7 if c.get('pl',1) else 0.,hardening_curve=curve if c.get('pl',1) else None)
    for j in range(n+1):
        for i in range(n+1):m.add_node(1+j*(n+1)+i,i/n,j/n,0.)
    e=0
    for j in range(n):
        for i in range(n):
            v=1+j*(n+1)+i;a,b,cc,d=v,v+1,v+n+2,v+n+1
            tris=[[a,b,cc],[a,cc,d]] if (i+j)%2==0 else [[a,b,d],[b,cc,d]]
            for tr in tris:
                e+=1
                kw={'formulation':'legacy-s3'} if form=='legacy' else {'reference_normal':[0.,0.,1.]}
                m.add_element(e,create_shell_element(e,tr,'mat',thickness=t,**kw))
    edge=[1+j*(n+1)+i for j in range(n+1) for i in range(n+1) if i in(0,n) or j in(0,n)]
    m.add_boundary_condition(FixedSupport('clamp',edge))
    L=LoadCase('p')
    for el in m.mesh.elements:L.add_pressure_load(el,c['p'])
    m.apply_boundary_conditions();return m,L

def _solve_plate(n, pressure, plastic, kinematics="von_karman"):
    model, load = plate(dict(n=n, t=.003, form="qualified", p=pressure, pl=int(plastic)))
    if kinematics == "linear":
        displacements, _ = fs.solve_linear(model, load)
    else:
        result = fs.solve_static_nonlinear(model, load, num_steps=20 if plastic else 4,
            max_iterations=25, kinematics=kinematics,
            corotational_tangent="work_conjugate" if kinematics == "corotational" else "auto")
        assert result.converged
        assert str(result.status) == "completed"
        assert result.load_factor == 1.0
        displacements = result.displacements
        if plastic:
            assert max(float(np.max(s["alpha"])) for s in result.element_states.values()) > 0
    u = np.asarray(displacements)
    assert np.isfinite(u).all()
    centre = 1 + (n // 2) * (n + 1) + n // 2
    return float(u[model.mesh.dof_manager.get_node_dofs(centre)[2]]) * 1000, u


@slow
def test_c1_plastic_n8_twenty_steps_physical_screening_band():
    # Coarse-mesh screening only (OD-S3-4); scientific acceptance needs n>=16.
    uz_mm, _ = _solve_plate(8, 50000, True)
    assert 27.3 <= uz_mm <= 44.4


@slow
def test_c4_plastic_n8_twenty_steps_s8r_band():
    uz_mm, _ = _solve_plate(8, 10000, True)
    assert 7.79 <= uz_mm <= 8.77


@slow
def test_protected_linear_and_corotational_displacement_hashes():
    # Stored pre-R1 constants, never regenerated from the candidate. Execute
    # with run_job --env-profile s3-base-run to preserve baseline runtime.
    expected = {
        ("linear", 8): "0be6e35ec37efac23e1c79a7f6e8b681b158d0d54994d2910242c79664b17c45",
        ("corotational", 4): "9620e80d96933d835146adc585e2d71e6c35d0e3b1099a6d1f6ff9535f18ce1d",
        ("corotational", 8): "23fe778428522ca78b4cb75ffa40854510405625d2c608f1ecb1af88a64c9355",
    }
    for (kinematics, n), stored_hash in expected.items():
        _, u = _solve_plate(n, 50000, False, kinematics)
        assert hashlib.sha256(u.tobytes()).hexdigest() == stored_hash, (kinematics, n)