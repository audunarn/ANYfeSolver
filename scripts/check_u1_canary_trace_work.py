"""Bounded algebraic check of the saved U1 canary's beam trace, not a solve.

Integrate the polynomial derivatives in pinned CalculiX 2.23
resultsmech_u1.f:443-468, anchoring values at the left node. Check their
strain energy against e_c3d_u1.f:479-529 before comparing trace work.
No shell interpolation equivalence or native B3 qualification is implied.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tarfile

import numpy as np
from numpy.polynomial import Polynomial as P


ARCHIVE = Path("C:/Github/ANYfeSolver/.qualification/b3-ge-e4-pl/calculix-223/ccx_2.23.src.tar.bz2")
EXPECTED = "9c88385c10fb04f5dc6c4e98027a51bebdd8aee3920e05190d6c1dd08357d6e7"


def plane(length, ei, shear):
    """Return [v1,theta1,v2,theta2] interpolation and source stiffness."""
    a = 12 * ei / (shear * length**2)
    b = 1 / (1 - a)
    v = [P([1, b*a, -3*b, 2*b]),
         length*b*P([0, 1-a/2, (a-4)/2, 1]),
         b*P([0, -a, 3, -2]),
         length*b*P([0, a/2, -(2+a)/2, 1])]
    theta = [6*b/length*P([0, -1, 1]),
             P([1, b*(a-4), 3*b]),
             6*b/length*P([0, 1, -1]),
             b*P([0, -(a+2), 3])]
    y1 = shear*ei*(12*ei + shear*length**2)
    y2 = (12*ei - shear*length**2)**2
    y3 = 4*ei*(shear**2*length**4 + 3*shear*length**2*ei + 36*ei**2)
    k = 12*y1/(length*y2)
    h = 6*y1/y2
    r = y3/(length*y2)
    t = -2*ei*(-shear**2*length**4 - 30*shear*length**2*ei + 72*ei**2)/(length*y2)
    source = np.array([[k,h,-k,h], [h,r,-h,t], [-k,-h,k,-h], [h,t,-h,r]])
    energy = np.empty((4,4))
    for i in range(4):
        for j in range(4):
            f = ei*theta[i].deriv()*theta[j].deriv()/length
            f += shear*length*(v[i].deriv()/length-theta[i])*(v[j].deriv()/length-theta[j])
            energy[i,j] = f.integ()(1)-f.integ()(0)
    error = np.linalg.norm(energy-source)/np.linalg.norm(source)
    assert error < 1e-12, error
    assert np.allclose([p(0) for p in v], [1,0,0,0], atol=1e-14)
    assert np.allclose([p(1) for p in v], [0,0,1,0], atol=1e-14)
    assert np.allclose([p(0) for p in theta], [0,1,0,0], atol=1e-14)
    assert np.allclose([p(1) for p in theta], [0,0,0,1], atol=1e-14)
    for x in (0,.17,.5,.83,1):
        # Unit rigid translation and unit rotation about the left end.
        assert abs(sum(v[i](x)*q for i,q in enumerate([1,0,1,0]))-1) < 1e-14
        assert abs(sum(v[i](x)*q for i,q in enumerate([0,1,length,1]))-length*x) < 1e-14
        assert abs(sum(theta[i](x)*q for i,q in enumerate([0,1,length,1]))-1) < 1e-14
    return v, theta, error, source


def main():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == EXPECTED
    deck = Path(os.environ['TEMP'])/'r8-bl5-u1-us3-canary-v1/canary.inp'
    deck_hash = hashlib.sha256(deck.read_bytes()).hexdigest()
    assert deck_hash == 'c30eeabe1bb3c1086687afb83f193dab0881176713cabedeca06b1dc0f28c1dc'
    with tarfile.open(ARCHIVE) as archive:
        identities = {name: hashlib.sha256(archive.extractfile(
            './CalculiX/ccx_2.23/src/'+name).read()).hexdigest()
            for name in ('resultsmech_u1.f','e_c3d_u1.f','us3_sub.f')}
    length = .5
    e, nu, area, inertia, kappa = 210e9, .3, .01, 8.333333333333333e-6, .8
    shear = kappa*e/(2*(1+nu))*area
    v, theta, error, stiffness = plane(length, e*inertia, shear)
    # Independent cantilever force-method compliance for the stated physical
    # section. This is an accuracy discriminator, not a source-transcription test.
    tip = np.linalg.solve(stiffness[2:,2:], np.eye(2))
    physical = np.array([[length**3/(3*e*inertia)+length/shear,
                          length**2/(2*e*inertia)],
                         [length**2/(2*e*inertia),length/(e*inertia)]])
    relative = (tip-physical)/physical
    assert abs(relative[1,1]) < 1e-12
    assert abs(relative[0,0]) > .1  # retained counterexample at this fixed fixture
    # Integral over each half-metre element; assemble all 18 beam columns.
    def integral(polys):
        return np.array([length*(p.integ()(1)-p.integ()(0)) for p in polys])
    local = np.zeros((6,12))
    for component in (0,3):
        local[component,component] = local[component,component+6] = length/2
    local[1,[1,5,7,11]] = integral(v)
    local[5,[1,5,7,11]] = integral(theta)
    # w' + psi is the source's other shear convention: psi = -theta.
    local[2,[2,4,8,10]] = integral(v)*[1,-1,1,-1]
    local[4,[2,4,8,10]] = integral(theta)*[-1,1,-1,1]
    mean = np.zeros((6,18))
    mean[:,:12] += local
    mean[:,6:] += local
    p1 = np.hstack([.25*np.eye(6), .5*np.eye(6), .25*np.eye(6)])
    # Lower beam surface arm (0,0,-.05), directors about global X/Y.
    port = np.zeros((5,6))
    port[:3,:3] = np.eye(3)
    port[0,4], port[1,3] = -.05, .05
    port[3,3], port[4,4] = 1,1
    change = port @ (mean-p1)
    # Actual canary supports: root DOFs 1..5 fixed, root yaw free.
    supported = change[:,5:]
    assert np.max(np.abs(supported)) > .1
    # Explicit supported field in the OLD constraint kernel; shell is zero.
    # epsilon=1 merely normalizes a linear virtual field, not a finite motion.
    witness = np.zeros(18)
    witness[8], witness[14] = -.5, 1
    assert np.linalg.norm(port @ p1 @ witness) == 0
    witness_gap = port @ mean @ witness
    assert abs(witness_gap[4]) > 1
    # US3 physical five-field P1 trace: us3_sub.f:1396-1423 (mass),
    # with P1 director gradients at :739-777. Its CS-DSG/ANDES strains
    # are separate assumed-strain maps, not gradients of this whole trace.
    shell_mean = np.hstack([.5*np.eye(6), .5*np.eye(6), np.zeros((6,6))])
    shell_port = port.copy()
    shell_port[0,4], shell_port[1,3] = .0175, -.0175
    constraint = np.hstack([-shell_port @ shell_mean, port @ mean])
    coordinates = np.array([[0,0,0],[1,0,0],[0,.2,0],
                            [0,0,.0675],[.5,0,.0675],[1,0,.0675]])
    rigid_errors = []
    for j in range(6):
        t, omega = np.zeros(3), np.zeros(3)
        (t if j < 3 else omega)[j % 3] = 1
        q = np.zeros((6,6))
        q[:,:3] = t + np.cross(omega, coordinates)
        q[:,3:] = omega
        rigid_errors.append(float(np.max(np.abs(constraint @ q.ravel()))))
    assert max(rigid_errors) < 1e-14
    # Transpose forces: check physical balance about the actual common origin,
    # independently of the gap calculation and including both eccentric arms.
    balances = []
    for j in range(5):
        forces = constraint[j].reshape(6,6)
        force = forces[:,:3].sum(axis=0)
        moment = (np.cross(coordinates,forces[:,:3])+forces[:,3:]).sum(axis=0)
        balances.append(float(max(np.max(np.abs(force)), np.max(np.abs(moment)))))
    assert max(balances) < 1e-14
    # Do not lock director rotation to geometric normal: a transverse slope
    # with zero rotations has displacement trace, but zero director channels.
    shear_field = np.zeros((3,6))
    shear_field[:,2] = coordinates[:3,0]
    shear_trace = shell_port @ shell_mean @ shear_field.ravel()
    assert np.array_equal(shear_trace[3:], [0,0]) and shear_trace[2] == .5
    # Normal spin is excluded from the attachment; not a zero-energy assertion.
    for node in range(3):
        assert np.array_equal(constraint[:,node*6+5], np.zeros(5))
    report = dict(source_archive_sha256=EXPECTED, source_members=identities,
                  canary_deck_sha256=deck_hash,
                  old_kernel_witness_new_gap=witness_gap.tolist(),
                  full_port_rigid_errors=rigid_errors,
                  full_port_wrench_balance_errors=balances,
                  shell_transverse_shear_trace=shear_trace.tolist(),
                  source_energy_relative_error=error,
                  u1_cantilever_tip_compliance=tip.tolist(),
                  force_method_tip_compliance=physical.tolist(),
                  cantilever_compliance_relative_difference=relative.tolist(),
                  beam_port_change_max_by_row=np.max(np.abs(change),axis=1).tolist(),
                  supported_beam_port_change_max_by_row=np.max(np.abs(supported),axis=1).tolist(),
                  mean_theta_y_columns=mean[4].tolist(),
                  status='NOT_EQUIVALENT_BEAM_TRACE',
                  limits='Declared US3 P1 physical trace with source-consistent U1 polynomial extension; no full strain-gradient identity, external execution, native B3 equivalence or qualification.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
