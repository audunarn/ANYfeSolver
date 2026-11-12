"""First touch on an explicit affine FE path with constant target normal.

This is an increment-history contract, not a conservative general collision
kernel or an assertion about the exact intervening equilibrium trajectory.
Quadratic extrema check the complete represented planar path, without sampling.
"""
import numpy as np

from .algebraic_path import _extrema


def _cross_coefficients(a,da,b,db):
    return np.array((np.cross(a,b),np.cross(a,db)+np.cross(da,b),np.cross(da,db)))


class S3LinearTouchPath:
    def __init__(self,owner,u,corners,*,require_full_coverage=True):
        self.owner=owner
        self.u=np.asarray(u)
        self.first,self.normal,_frame,self.origin=owner._patch(owner.committed_displacement)
        self.last=corners
        self.delta={e:corners[e]-c for e,c in self.first.items()}
        settings=owner.friction_settings
        tol=settings.geometry_relative_tolerance
        n=self.normal
        scale=max(np.linalg.norm(c-self.origin,axis=1).max() for c in (*self.first.values(),*corners.values()))
        self.roundoff=256*np.finfo(float).eps*scale
        vertices,edges={},{}
        for e,adapter in owner.target_adapters.items():
            c,dc=self.first[e],self.delta[e]
            raw=_cross_coefficients(c[1]-c[0],dc[1]-dc[0],c[2]-c[0],dc[2]-dc[0])
            magnitude=max(np.linalg.norm(raw,axis=1).max(),np.finfo(float).tiny)
            minimum=float(_extrema(raw@n,0.,1.)[0])
            if minimum<=1e-12*magnitude:
                raise ValueError("first-touch affine path has an unresolved or inverted target facet")
            # Sum bounds transverse coefficients everywhere; normalize by the
            # *minimum* represented area, not a large endpoint/change coefficient.
            if np.linalg.norm(raw-(raw@n)[:,None]*n,axis=1).sum()>tol*minimum:
                raise ValueError("first-touch affine path requires a constant target normal")
            for i,node in enumerate(adapter.node_ids):vertices[node]=(c[i],dc[i])
            for i in range(3):
                edge=(adapter.node_ids[i],adapter.node_ids[(i+1)%3])
                if tuple(sorted(edge)) in owner.boundary_edges:edges[edge[0]]=edge[1]
        # Convexity and source corner containment are quadratic along this
        # affine, constant-normal path. Fixed conforming topology comes from owner.
        source=[]
        for adapter in owner.source_adapters:
            c=adapter.current_positions(owner.committed_displacement)
            dc=adapter.current_positions(self.u)-c
            source.extend(zip(c,dc))
        for a,b in edges.items():
            p,dp=vertices[a];q,dq=vertices[b]
            r,dr=vertices[edges[b]]
            if float(_extrema(_cross_coefficients(q-p,dq-dp,r-q,dr-dq)@n,0.,1.)[0]) < -tol*scale**2:
                raise ValueError("first-touch affine target path is not convex")
            if require_full_coverage:
                for x,dx in source:
                    if float(_extrema(_cross_coefficients(q-p,dq-dp,x-p,dx-dp)@n,0.,1.)[0]) < -tol*scale**2:
                        raise ValueError("first-touch affine path loses complete source coverage")
        self.local_start=owner.committed_displacement[owner.work_columns]
        self.local_end=self.u[owner.work_columns]

    def anchor(self,index,*,probe=None):
        owner=self.owner
        x0=owner.point_reference[index]+owner.point_maps[index]@self.local_start
        x1=owner.point_reference[index]+owner.point_maps[index]@self.local_end
        element0=next(iter(self.first))
        o0=self.first[element0][0]
        o1=self.last[element0][0]
        g0=float(self.normal@(x0-o0));g1=float(self.normal@(x1-o1))
        if g0 < -self.roundoff:
            raise ValueError("first-touch history origin has unresolved negative gap")
        if g1>=0:return None
        if g0<=self.roundoff:
            alpha=0.
        else:
            if g0-g1<=self.roundoff:
                raise ValueError("first-touch affine gap slope is unresolved")
            alpha=g0/(g0-g1)
        stage={e:c+alpha*self.delta[e] for e,c in self.first.items()}
        c=stage[element0]
        v=c[1]-c[0]
        frame=np.column_stack((v/np.linalg.norm(v),np.cross(self.normal,v/np.linalg.norm(v)),self.normal))
        x=x0+alpha*(x1-x0)
        if probe is None:
            element,bary=owner._locate(x,stage,self.normal,frame,c[0])
        else:
            element,base_bary,base_alpha=probe
            if element not in stage or abs(alpha-base_alpha)>owner.friction_settings.maximum_probe_touch_fraction_change:
                raise ValueError("first-touch probe exceeds its fraction resolution bound")
            bary=tuple(owner._facet_coordinates(x,stage[element],frame))
            if np.max(np.abs(np.asarray(bary)-base_bary))>owner.friction_settings.maximum_probe_barycentric_change:
                raise ValueError("first-touch probe exceeds its material-facet resolution bound")
        return element,bary,float(alpha)
