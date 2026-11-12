"""G0 F03-P: actual physical Q4/S3 shell response to integrated two-rail line work."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEMesh, Material
from anyfesolver._ge_beam3_mixed_ad import so3_log, transpose
from anyfesolver._ge_beam3_variational_shell import rotation_jets


def line_samples():
    reference = np.array(((-.2,-.25,0.),(.2,-.25,0.),(.2,.25,0.),(-.2,.25,0.)))
    rotation = rotation_jets(reference, reference)
    # c_R=Log(R_shell.T) for a fixed beam; lambda_R=(0,0,1).
    potential = so3_log(transpose(rotation))[2]
    gradient = potential.gradient.reshape(4,6)[:,:3]
    assert np.linalg.norm(potential.gradient.reshape(4,6)[:,3:]) == 0.
    assert np.linalg.norm(gradient.sum(axis=0)) < 1e-12
    assert abs(np.cross(reference,gradient).sum(axis=0)[2]+1.) < 1e-12
    return gradient


def mesh_stiffness(n, topology):
    xs = np.linspace(-1.,1.,n+1)
    ys = np.linspace(-.5,.5,n+1)
    mesh = FEMesh()
    positions = []
    for y in ys:
        for x in xs:
            positions.append((float(x),float(y),0.))
            mesh.add_node(len(positions),float(x),float(y),0.)
    law = Material("steel",210e9,.3,density=7850.)
    row,col,data = [],[],[]
    elements = 0
    for j in range(n):
        for i in range(n):
            a = 1+j*(n+1)+i
            b = a+1
            d = a+n+1
            c = d+1
            faces = ((a,b,c,d),) if topology == "Q4" else ((a,b,c),(a,c,d))
            for nodes in faces:
                elements += 1
                shell = create_shell_element(elements,list(nodes),"steel",
                    formulation="e4-pl" if topology == "Q4" else "e4-pl-s3-v2d",
                    thickness=.07,reference_normal=(0.,0.,1.))
                physical = np.asarray(shell.compute_stiffness_components(mesh,law)["physical"])
                selection = [6*k+axis for k in range(len(nodes)) for axis in (0,1)]
                part = physical[np.ix_(selection,selection)]
                ids = [2*(node-1)+axis for node in nodes for axis in (0,1)]
                for r,global_r in enumerate(ids):
                    for c_index,global_c in enumerate(ids):
                        row.append(global_r);col.append(global_c);data.append(part[r,c_index])
    count = 2*len(positions)
    k = sparse.coo_matrix((data,(row,col)),shape=(count,count)).tocsr()
    return xs,ys,np.array(positions),k,elements


def host_weights(x,y,xs,ys,topology):
    n = len(xs)-1
    i = min(max(int(np.searchsorted(xs,x,side="right")-1),0),n-1)
    j = min(max(int(np.searchsorted(ys,y,side="right")-1),0),n-1)
    xi = float((x-xs[i])/(xs[i+1]-xs[i]))
    eta = float((y-ys[j])/(ys[j+1]-ys[j]))
    a = j*(n+1)+i
    b = a+1
    d = a+n+1
    c = d+1
    if topology == "Q4":
        return ((a,(1-xi)*(1-eta)),(b,xi*(1-eta)),
                (c,xi*eta),(d,(1-xi)*eta))
    if eta <= xi:
        return ((a,1-xi),(b,xi-eta),(c,eta))
    return ((a,1-eta),(c,xi),(d,eta-xi))


def specimen(n,topology,gradient):
    xs,ys,positions,k,elements = mesh_stiffness(n,topology)
    force = np.zeros(len(positions)*2)
    roots,weights = np.polynomial.legendre.leggauss(4)
    # Host changes where either rail window edge s+-0.2 crosses a shell edge.
    breaks = sorted({-.4,.4} | {float(np.round(x-shift,14))
        for x in xs for shift in (-.2,.2) if -.4 < x-shift < .4})
    for start,end in zip(breaks[:-1],breaks[1:]):
        for root,weight in zip(roots,weights):
            s = float((start+end)/2.+(end-start)*root/2.)
            measure = float((end-start)*weight/2.)
            for sample,(dx,y) in enumerate(((-.2,-.25),(.2,-.25),
                                            (.2,.25),(-.2,.25))):
                for node,shape in host_weights(s+dx,y,xs,ys,topology):
                    force[2*node:2*node+2] += measure*shape*gradient[sample,:2]
    total = force.reshape(-1,2).sum(axis=0)
    moment = float(np.sum(positions[:,0]*force[1::2]-positions[:,1]*force[0::2]))
    fixed = np.array([2*(j*(n+1)+i)+axis for j in range(n+1) for i in range(n+1)
                      if i in (0,n) or j in (0,n) for axis in (0,1)],dtype=int)
    free = np.array(sorted(set(range(len(force)))-set(fixed)),dtype=int)
    displacement = np.zeros_like(force)
    displacement[free] = spsolve(k[free,:][:,free].tocsc(),force[free])
    reaction = k@displacement-force
    work = float(force@displacement)
    energy = float(displacement@(k@displacement)/2.)
    free_equilibrium = float(np.linalg.norm(reaction[free])/max(1.,np.linalg.norm(force)))
    work_error = abs(work-2*energy)/max(1.,abs(work),abs(2*energy))
    passed = (np.isfinite(displacement).all() and work > 0.
              and np.linalg.norm(total)<=1e-8 and abs(moment+.8)<=1e-8
              and free_equilibrium<=1e-8 and work_error<=1e-8)
    return dict(n=n,topology=topology,shell_elements=elements,nnz=int(k.nnz),
                line_force_norm=float(np.linalg.norm(total)),line_moment=moment,
                compliance=work,twice_energy=2*energy,work_error=work_error,
                free_equilibrium=free_equilibrium,
                status="diagnostic_pass" if passed else "failed")


if __name__ == "__main__":
    gradient = line_samples()
    rows = [specimen(n,topology,gradient) for topology in ("Q4","S3-V2D")
            for n in (4,8,16)]
    passed = all(row["status"]=="diagnostic_pass" for row in rows)
    print(json.dumps(dict(profile="G0-F03-P-v1",status="diagnostic_pass" if passed else "failed",
                          results=rows),sort_keys=True))
    raise SystemExit(0 if passed else 1)
