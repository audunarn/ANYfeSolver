"""Opt-in principal-Log shell supports; local blocks, spatial dual reactions.

Original psi=Log(Exp(theta) A), with fixed components of psi zero. Supported
nodes use psi as their Newton parameter; other nodes use theta. No support
equation transport is implicit: the new policy explicitly retains the original
psi law through local chart changes. Element responses are unchanged; there is
no global dense transformation.
"""
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from ._ge_beam3_pose_joint import _exp_terms

POLICY = 'NATIVE_SHELL_COORDINATE_SUPPORT_V1'
LEGACY = 'historical_fixed_coordinates'
BRANCH_MARGIN = 1e-7


class CoordinateSupportChartDomainError(ValueError):
    """A trial leaves the prospectively admitted local/original chart branch."""


def principal(frame):
    value = Rotation.from_matrix(np.array(frame, dtype=float, copy=True)).as_rotvec()
    if not np.isfinite(value).all() or np.linalg.norm(value) >= np.pi-BRANCH_MARGIN:
        raise CoordinateSupportChartDomainError('coordinate support requires bounded principal Log chart')
    return value


def blocks(q, anchors, shell_count, fixed):
    """(slots, psi, J_parameter, dJ_parameter, dtheta/dparameter, mask)."""
    fixed = set(map(int, fixed)); result=[]
    for i,row in enumerate(np.asarray(q[:shell_count]).reshape(-1,6)):
        if not np.isfinite(row).all() or np.linalg.norm(row[3:]) >= np.pi-BRANCH_MARGIN:
            raise CoordinateSupportChartDomainError('coordinate support requires bounded local principal chart')
        slots=np.arange(6*i+3,6*i+6); mask=np.array([int(s) in fixed for s in slots])
        R,Jtheta,dJtheta=_exp_terms(row[3:])
        A=np.eye(3) if anchors is None else anchors[i]
        psi=principal(R@A); _,Jpsi,dJpsi=_exp_terms(psi)
        if min(np.linalg.svd(Jtheta,compute_uv=False)) <= 1e-11:
            raise ValueError('singular local shell coordinate chart')
        if np.any(mask):
            result.append((slots,psi,Jpsi,dJpsi,np.linalg.solve(Jtheta,Jpsi),mask))
        else:
            result.append((slots,psi,Jtheta,dJtheta,np.eye(3),mask))
    return result


def _map(size, records, which):
    rows=list(range(size));cols=list(range(size));data=[1.]*size
    for record in records:
        slots=record[0]; block=record[which]-np.eye(3)
        for j in range(3):
            for k in range(3):
                if block[j,k] != 0.:
                    rows.append(int(slots[j]));cols.append(int(slots[k]));data.append(float(block[j,k]))
    return sparse.coo_matrix((data,(rows,cols)),shape=(size,size)).tocsc()


def pullback(q, anchors, shell_count, fixed, residual, tangent, applied):
    """Complete spatial residual pullback, including reaction/load curvature."""
    records=blocks(q,anchors,shell_count,fixed);size=len(residual)
    M=_map(size,records,2);P=_map(size,records,4)
    rows=[];cols=[];data=[]
    for slots,psi,J,dJ,_,mask in records:
        for k in range(3):
            values=dJ[:,:,k].T@residual[slots]
            rows.extend(map(int,slots));cols.extend([int(slots[k])]*3);data.extend(values)
    curvature=sparse.coo_matrix((data,(rows,cols)),shape=(size,size)).tocsc()
    r=np.asarray(M.T@residual);K=(M.T@tangent@P+curvature).tocsc()
    f=np.asarray(M.T@np.r_[applied,np.zeros(size-len(applied))])[:len(applied)]
    support_gap=max((float(np.max(np.abs(psi[mask]),initial=0.)) for _,psi,_,_,_,mask in records),default=0.)
    physical_residual=np.array(residual,copy=True);physical_applied=np.array(applied,copy=True)
    for slots,_,J,_,_,mask in records:
        if not np.any(mask):continue
        free=np.flatnonzero(~mask)
        if not len(free):continue
        Q,_=np.linalg.qr(J[:,free],mode='reduced')
        physical_residual[slots[free]]=Q.T@residual[slots]
        physical_applied[slots[free]]=Q.T@applied[slots]
    if not np.isfinite(r).all() or not np.isfinite(K.data).all() or not np.isfinite(f).all():
        raise ValueError('nonfinite coordinate support pullback')
    return r,K,f,support_gap,physical_residual,physical_applied


def advance(q, made, anchors, shell_count, fixed, delta):
    """Advance supported original psi, reconstruct the actual local theta."""
    for slots,psi,_,_,_,mask in blocks(q,anchors,shell_count,fixed):
        if not np.any(mask):continue
        if np.any(delta[slots][mask] != 0.):
            raise ValueError('fixed original shell coordinate increment must be zero')
        moved=psi+delta[slots];moved[mask]=0.
        if np.linalg.norm(moved) >= np.pi-BRANCH_MARGIN:
            raise CoordinateSupportChartDomainError('coordinate support advance leaves bounded principal Log chart')
        A=np.eye(3) if anchors is None else anchors[int(slots[0])//6]
        made[slots]=principal(_exp_terms(moved)[0]@A.T)
    blocks(made,anchors,shell_count,fixed)


def reactions(q, anchors, shell_count, fixed, covector, count):
    value=np.zeros(count);value[fixed]=covector[fixed]
    for slots,_,J,_,_,mask in blocks(q,anchors,shell_count,fixed):
        if np.any(mask):
            selected=np.zeros(3);selected[mask]=covector[slots][mask]
            value[slots]=np.linalg.solve(J.T,selected)
    return value


def support_gap(q, anchors, shell_count, fixed):
    return max((float(np.max(np.abs(psi[mask]),initial=0.)) for _,psi,_,_,_,mask in blocks(q,anchors,shell_count,fixed)),default=0.)
