"""G0 F04: physical linear shell response under rigid/distributing footprints."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from anyfesolver.elements import create_shell_element
from anyfesolver.fe_core import FEMesh, Material


def case(n: int, topology: str) -> dict:
    grid = np.linspace(-1., 1., n + 1)
    mesh = FEMesh()
    for j, y in enumerate(grid):
        for i, x in enumerate(grid):
            mesh.add_node(1 + j * (n + 1) + i, float(x), float(y), 0.)
    law = Material("steel", 210e9, .3, density=7850.)
    rows, columns, values = [], [], []
    element_id = 0
    for j in range(n):
        for i in range(n):
            a = 1 + j*(n+1) + i
            b, d = a+1, a+n+1
            c = d+1
            cells = ((a,b,c,d),) if topology == "Q4" else ((a,b,c),(a,c,d))
            for nodes in cells:
                element_id += 1
                shell = create_shell_element(element_id, list(nodes), "steel",
                    formulation="e4-pl" if topology == "Q4" else "e4-pl-s3-v2d",
                    thickness=.07, reference_normal=(0.,0.,1.))
                local = np.asarray(shell.compute_stiffness_components(mesh, law)["physical"])
                local_dofs = [6*k+axis for k in range(len(nodes)) for axis in (0,1)]
                global_dofs = [2*(node-1)+axis for node in nodes for axis in (0,1)]
                local = local[np.ix_(local_dofs, local_dofs)]
                for row, global_row in enumerate(global_dofs):
                    for col, global_col in enumerate(global_dofs):
                        rows.append(global_row)
                        columns.append(global_col)
                        values.append(local[row, col])
    count = 2*(n+1)**2
    stiffness = sparse.coo_matrix((values,(rows,columns)), shape=(count,count)).tocsr()
    outer, footprint = [], []
    xy = {}
    for j, y in enumerate(grid):
        for i, x in enumerate(grid):
            node = j*(n+1)+i
            xy[node] = (float(x),float(y))
            if i in (0,n) or j in (0,n):
                outer.extend((2*node,2*node+1))
            if abs(x)<=.25+1e-12 and abs(y)<=.25+1e-12:
                footprint.append(node)
    outer = np.array(sorted(outer))
    footprint_dofs = np.array([2*node+axis for node in footprint for axis in (0,1)])
    rigid = np.zeros(count)
    for node in footprint:
        x,y=xy[node]
        rigid[2*node:2*node+2] = (-y,x)
    fixed = np.array(sorted(set(outer)|set(footprint_dofs)))
    rigid_free = np.array(sorted(set(range(count))-set(fixed)))
    rigid[rigid_free] = spsolve(stiffness[rigid_free,:][:,rigid_free].tocsc(),
                               -stiffness[rigid_free,:][:,fixed]@rigid[fixed])
    rigid_reaction = stiffness@rigid
    rigid_moment = sum(-xy[node][1]*rigid_reaction[2*node]
                       +xy[node][0]*rigid_reaction[2*node+1] for node in footprint)
    rigid_work = float(rigid@rigid_reaction)

    free = np.array(sorted(set(range(count))-set(outer)))
    inertia = sum(xy[node][0]**2+xy[node][1]**2 for node in footprint)
    constraint = sparse.lil_matrix((3,count))
    for node in footprint:
        x,y=xy[node]
        constraint[0,2*node]=1./len(footprint)
        constraint[1,2*node+1]=1./len(footprint)
        constraint[2,2*node]=-y/inertia
        constraint[2,2*node+1]=x/inertia
    c = constraint.tocsr()[:,free]
    k=stiffness[free,:][:,free].tocsc()
    saddle=sparse.bmat([[k,c.T],[c,None]],format="csc")
    rhs=np.r_[np.zeros(len(free)),np.array([0.,0.,1.])]
    solved=spsolve(saddle,rhs)
    distributed=np.zeros(count)
    distributed[free]=solved[:len(free)]
    lam=solved[len(free):]
    distributed_moment=float(-lam[2])
    distributed_work=float(distributed@(stiffness@distributed))
    free_error=float(np.linalg.norm(stiffness[free,:]@distributed+c.T@lam)
                     /max(1.,np.linalg.norm(stiffness[free,:]@distributed)))
    constraint_error=float(np.linalg.norm(c@distributed[free]-np.array([0.,0.,1.])))
    rigid_error=abs(rigid_moment-rigid_work)/max(1.,abs(rigid_moment))
    distributed_error=abs(distributed_moment-distributed_work)/max(1.,abs(distributed_moment))
    return {"topology":topology,"n":n,"footprint_nodes":len(footprint),
            "rigid_moment":float(rigid_moment),"distributing_moment":distributed_moment,
            "rigid_work_error":float(rigid_error),"distributing_work_error":float(distributed_error),
            "constraint_error":constraint_error,"free_error":free_error}


def main() -> int:
    results=[case(n,topology) for topology in ("Q4","S3-V2D") for n in (8,16)]
    passed=all(0.<r["distributing_moment"]<r["rigid_moment"]
               and max(r[key] for key in ("rigid_work_error","distributing_work_error",
                                        "constraint_error","free_error"))<=1e-10 for r in results)
    print(json.dumps({"case":"G0_F04_PHYSICAL_FOOTPRINT_DISTINCTION_V1",
                      "status":"diagnostic_pass" if passed else "failed","results":results},sort_keys=True))
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
