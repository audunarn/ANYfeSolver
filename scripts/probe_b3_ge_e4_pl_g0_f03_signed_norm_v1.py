"""Signed retained KKT norm diagnostic; finite-mesh evidence, not F03 proof.

Uses the exact p/r factorization and absolute local Krr block. Reports both
positive constraint inf-sup and the fully normalized signed saddle floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if not os.environ.get('NUMBA_CACHE_DIR'):
    raise RuntimeError('set an explicit writable NUMBA_CACHE_DIR before import')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import anyfesolver
import numba
import numpy as np
import scipy
from scipy import linalg
from threadpoolctl import threadpool_limits

from _b3_e4_f03_native_fixture import build
from _b3_e4_f03_shell_trace import make_shell
from probe_b3_ge_e4_pl_g0_f03_k1_rank import candidate_spaces
from probe_b3_ge_e4_pl_g0_f03_kkt import (
    assemble, scales, supported_stiffness,
)
from probe_b3_ge_e4_pl_g0_f03_local_dual_v1 import integrated


F0 = 1e6  # N/m: line-load and multiplier-density scale
L0 = 1.   # m: physical attachment interval
U = F0*L0/(210e9*.07)
W0 = F0*L0*U
CASES = tuple((topology, n, n) for topology in ('Q4','S3-V2D')
              for n in (4,8,16)) + tuple((topology,8,n)
              for topology in ('Q4','S3-V2D') for n in (1,64))


def couplings(beam, shell):
    spaces = {s.name:s for s in candidate_spaces(beam,shell)}
    p1 = assemble(beam,shell,spaces['p1-beam'])
    p0 = assemble(beam,shell,spaces['p0-beam'])
    n = beam.n_half_cells
    x = [3*i for i in range(1,n+1)]
    y = [3*i+1 for i in range(1,n+1)]
    yaw = [3*i+2 for i in range(n)]
    k4c = np.vstack((p1.matrix[x].toarray(),p1.matrix[y].toarray(),
                     p0.matrix[yaw].toarray()))
    k4m = linalg.block_diag(p1.mass[np.ix_(x,x)],
                            p1.mass[np.ix_(y,y)],p0.mass[np.ix_(yaw,yaw)])
    dualc,dualm,_ = integrated(beam,shell,p1.integration_knots,4)
    return {'k4':(k4c,k4m),'local-dual':(dualc,dualm)}


def reciprocal_sqrt(values, vectors, label):
    maximum = max(float(np.max(np.abs(values))),np.finfo(float).tiny)
    if np.min(values) <= 1e-12*maximum:
        raise ValueError(f'{label} lacks verified positive definiteness')
    return vectors/np.sqrt(values)[None,:]


def measure(beam,shell,c,m):
    n_p = shell.free_count+len(beam.physical)
    k = supported_stiffness(beam,shell).toarray()
    if len(k)+len(c) > 1200:
        raise ValueError('bounded dense diagnostic exceeds 1200 KKT rows')
    sq,_ = scales(beam,shell,c.shape[0])
    k_scaled = (sq[:,None]*k)*sq[None,:]/W0
    skew = float(np.linalg.norm(k_scaled-k_scaled.T)/
                 max(np.linalg.norm(k_scaled),np.finfo(float).tiny))
    if skew > 1e-10:
        raise ValueError(f'signed operator nonsymmetric: {skew}')
    kpp,kpr,krp,rr_scaled = (k_scaled[:n_p,:n_p],k_scaled[:n_p,n_p:],
                             k_scaled[n_p:,:n_p],k_scaled[n_p:,n_p:])
    # Work-scaled factorization avoids raw mixed displacement/resultant units.
    rp = linalg.solve(rr_scaled,krp,assume_a='sym')
    ke_scaled = kpp-kpr@rp
    ke_skew = float(np.linalg.norm(ke_scaled-ke_scaled.T)/np.linalg.norm(ke_scaled))
    rr_skew = float(np.linalg.norm(rr_scaled-rr_scaled.T)/np.linalg.norm(rr_scaled))
    mass_skew = float(np.linalg.norm(m-m.T)/np.linalg.norm(m))
    if max(ke_skew,rr_skew,mass_skew) > 1e-10:
        raise ValueError('physical, retained or multiplier norm block is nonsymmetric')
    ep,up = linalg.eigh(ke_scaled,check_finite=True)
    er,ur = linalg.eigh(rr_scaled,check_finite=True)
    maximum = max(float(np.max(np.abs(er))),np.finfo(float).tiny)
    if np.min(np.abs(er)) <= 1e-12*maximum:
        raise ValueError('native Krr has a near-zero signed mode')
    pp = reciprocal_sqrt(ep,up,'physical Ke')
    rr = ur/np.sqrt(np.abs(er))[None,:]
    c_scaled = c*sq[None,:]/(U*L0)
    cp,cr = c_scaled[:,:n_p],c_scaled[:,n_p:]
    cl = cp-cr@rp
    b = np.hstack((cl@pp,cr@rr))
    em,um = linalg.eigh(m,check_finite=True)
    # Whitening by M^(-1/2) implements an L2 multiplier coefficient norm.
    multiplier_inv_sqrt = reciprocal_sqrt(em,um,'multiplier mass')
    bw = multiplier_inv_sqrt.T@b
    singular = linalg.svdvals(bw)
    signed = np.r_[np.ones(n_p),np.sign(er)]
    signed_schur = (bw*signed[None,:])@bw.T
    schur_eig = linalg.eigvalsh(signed_schur)
    saddle = np.zeros((len(signed)+len(c),)*2)
    saddle[np.diag_indices(len(signed))] = signed
    saddle[:len(signed),len(signed):] = bw.T
    saddle[len(signed):,:len(signed)] = bw
    saddle_singular = linalg.svdvals(saddle)
    # Native Krr is exactly local by element for this retained formulation.
    n_e = beam.n_half_cells//2
    if len(rr_scaled)%n_e:
        raise ValueError('native retained block count is not element-local')
    block = len(rr_scaled)//n_e
    off = rr_scaled.copy()
    for e in range(n_e):
        j = slice(e*block,(e+1)*block)
        off[j,j] = 0.
    off_ratio = float(np.linalg.norm(off)/np.linalg.norm(rr_scaled))
    if off_ratio > 1e-12:
        raise ValueError(f'Krr is not element-local: {off_ratio}')
    return {'primal_count':len(k),'retained_count':len(rr_scaled),'rows':len(c),
            'skew':skew,'ke_skew':ke_skew,'krr_skew':rr_skew,
            'mass_skew':mass_skew,'ke_min_eigen':float(ep[0]),
            'krr_min_abs_eigen':float(np.min(np.abs(er))),
            'krr_negative_modes':int(np.count_nonzero(er < 0)),
            'krr_off_element_ratio':off_ratio,
            'constraint_infsup_floor':float(singular[-1]),
            'signed_schur_min_abs_eigen':float(np.min(np.abs(schur_eig))),
            'signed_schur_negative_modes':int(np.count_nonzero(schur_eig < 0)),
            'signed_kkt_min_singular':float(saddle_singular[-1]),
            'signed_kkt_max_singular':float(saddle_singular[0]),
            'status':'FINITE_MESH_OPERATOR_ADAPTED_ONLY'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=False)
    sources = (Path(__file__).name, '_b3_e4_f03_native_fixture.py',
               '_b3_e4_f03_shell_trace.py', 'probe_b3_ge_e4_pl_g0_f03_kkt.py',
               'probe_b3_ge_e4_pl_g0_f03_k1_rank.py',
               'probe_b3_ge_e4_pl_g0_f03_local_dual_v1.py')
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in sources}
    manifest={'purpose':'EXPLORATORY_SIGNED_RETAINED_NORM_NOT_F03_ACCEPTANCE',
              'cases':CASES,'spaces':('k4','local-dual'),'source_sha256':hashes,
              'anyfesolver_import':str(Path(anyfesolver.__file__).resolve()),
              'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'numba':numba.__version__,'cache':os.environ['NUMBA_CACHE_DIR'],
              'blas_threads':1,'max_dense_kkt_rows':1200,
              'scale':{'force_density_n_per_m':F0,'attachment_length_m':L0,
                       'displacement_m':U,'work_j':W0,
                       'yaw_conversion_length_m':.46}}
    (args.root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    failed = False
    with threadpool_limits(limits=1), (args.root/'results.jsonl').open('x',encoding='utf-8') as out:
        for topology,n_beam,n_shell in CASES:
            row={'topology':topology,'n_beam':n_beam,'n_shell':n_shell,'spaces':{}}
            try:
                beam,shell=build(n_beam),make_shell(n_shell,topology)
                for name,(c,m) in couplings(beam,shell).items():
                    row['spaces'][name]=measure(beam,shell,c,m)
            except Exception as exc:
                row['failure_type']=type(exc).__name__
                row['failure']=str(exc)
                failed=True
            out.write(json.dumps(row,sort_keys=True)+'\n')
            out.flush()
            print(json.dumps({'case':(topology,n_beam,n_shell),
                              'failure':row.get('failure'),
                              'summary':{name:{'beta':v['constraint_infsup_floor'],
                                               'signed_sigma':v['signed_kkt_min_singular']}
                                         for name,v in row['spaces'].items()}},sort_keys=True),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
