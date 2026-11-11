"""Partial S3 friction with a replay-defined material history function.

Development owner: moving common refinement queries immutable committed poses
and pressures. No internal-state interpolation, sample-index history or history
truncation. Exactness refers to this discrete endpoint return law, not continuous
work or engineering qualification. Material-region boundaries partition queries.
"""
from dataclasses import asdict,dataclass
from copy import copy
from collections import OrderedDict
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .overlap import _clip_convex,_signed_area
from .s3_sliding_friction import S3SlidingFrictionManager,S3SlidingFrictionSettings,_reference_quadrature
from .s3_friction_touch import S3LinearTouchPath

PROFILE="S3_PARTIAL_MATERIAL_REPLAY_STATIC_DEV_V1"
_SOURCE_UV=np.array(((0.,0.),(1.,0.),(0.,1.)))


class _ReplayTangentUnresolved(ValueError):
    def __init__(self,message,force,summary):
        super().__init__(message)
        self.force=force
        self.summary=summary


class ReplayContactSubdivisionRequired(ValueError):
    """A material path needs event-resolved constitutive integration.

    Fractions refer to the affine pose segment being replayed, which may be
    historical. They are diagnostics, never accepted material history. Cutting
    back the current equilibrium step cannot repair an earlier missed pulse.
    """
    def __init__(self,intervals):
        self.intervals=tuple((int(point),float(start),float(end))
                             for point,start,end in intervals)
        if not self.intervals or any(not 0<=start<end<=1
                                    for _,start,end in self.intervals):
            raise ValueError("invalid replay contact subdivision intervals")
        # An interior sample witnesses the earliest missed interval. It is not
        # permission to approximate the entire interval by one return update.
        _,start,end=min(self.intervals,key=lambda item:(item[1],item[2],item[0]))
        self.suggested_fraction=start+(end-start)/2
        super().__init__("replay material path needs subdivision for a hidden contact interval")


@dataclass(frozen=True)
class S3ReplayFrictionSettings:
    material: S3SlidingFrictionSettings
    integration_refinement: int
    force_integration_tolerance_n: float
    moment_integration_tolerance_nm: float
    work_integration_tolerance_j: float
    maximum_history_frames: int=64
    maximum_integration_points: int=32768
    maximum_partition_cells: int=4096
    maximum_path_samples: int=2097152
    event_subdivisions: int=0
    maximum_event_stages: int=4096
    prepared_pressure_paths: bool=False
    maximum_prepared_stage_batch: int=256
    prepared_partial_pressure_paths: bool=False
    maximum_prepared_partial_pressure_samples: int=32768

    def __post_init__(self):
        if type(self.material) is not S3SlidingFrictionSettings or self.material.activation_history!="linear_constant_normal_touch_dev":
            raise ValueError("replay friction requires the explicit first-touch material law")
        if type(self.integration_refinement) is not int or not 0<=self.integration_refinement<=4:
            raise ValueError("replay integration refinement must be from0 to4")
        for value in (self.force_integration_tolerance_n,self.moment_integration_tolerance_nm,self.work_integration_tolerance_j):
            if not np.isfinite(value) or value<=0:raise ValueError("replay needs positive dimensioned integration controls")
        if type(self.event_subdivisions) is not int or not 0<=self.event_subdivisions<=256:
            raise ValueError("replay event subdivisions must be from0 to256")
        if type(self.prepared_pressure_paths) is not bool or (self.prepared_pressure_paths and not self.event_subdivisions):
            raise ValueError("prepared pressure paths require explicit event subdivisions")
        if type(self.prepared_partial_pressure_paths) is not bool or (self.prepared_partial_pressure_paths and not self.prepared_pressure_paths):
            raise ValueError("prepared partial pressure paths require prepared pressure paths")
        for name in ("maximum_history_frames","maximum_integration_points","maximum_partition_cells","maximum_path_samples","maximum_event_stages","maximum_prepared_stage_batch","maximum_prepared_partial_pressure_samples"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError("replay resource caps must be positive integers")


class _PartialTemplate(S3SlidingFrictionManager):
    def _patch(self,u,*,validate=True,require_full_coverage=False):
        return super()._patch(u,validate=validate,require_full_coverage=require_full_coverage)


def _rational_polygon(polygon):
    return np.array([[x if isinstance(x,Fraction) else Fraction(float(x)) for x in point]
                     for point in polygon],dtype=object)


def _twice_area(polygon):
    edges=polygon[1:]-polygon[0]
    return sum((a[0]*b[1]-a[1]*b[0] for a,b in zip(edges[:-1],edges[1:])),Fraction(0))


def _area_centroid(polygon):
    """Exact area centroid; invariant to redundant boundary vertices."""
    numerator=np.array((Fraction(0),Fraction(0)));denominator=Fraction(0)
    for i in range(1,len(polygon)-1):
        triangle=np.array((polygon[0],polygon[i],polygon[i+1]),dtype=object)
        weight=_twice_area(triangle)
        numerator+=weight*np.sum(triangle,axis=0);denominator+=weight
    if denominator<=0:raise ValueError("replay reference polygon has no positive area")
    return numerator/(3*denominator)


def _split(polygon,start,end):
    """Split a convex material polygon by a fixed historical boundary line."""
    # Exact arithmetic only for the reference-cell partition. Retaining
    # rational intersections until integration avoids fictitious inverted
    # slivers from re-clipping rounded intersections. FE maps remain floats.
    polygon=_rational_polygon(polygon)
    start,end=_rational_polygon((start,end))
    edge=end-start;relative=polygon-start
    distance=edge[0]*relative[:,1]-edge[1]*relative[:,0]
    if np.min(distance)>=0 or np.max(distance)<=0:return (polygon,)
    result=[]
    for sign in (1,-1):
        output=[];previous=polygon[-1];old=sign*distance[-1]
        for current,value in zip(polygon,sign*distance):
            if (old>=0)!=(value>=0):output.append(previous+old/(old-value)*(current-previous))
            if value>=0:output.append(current)
            previous,old=current,value
        # A vertex on the splitting line can be emitted as both an
        # intersection and an endpoint. Remove only identical neighbours;
        # a tolerance here would erase thin, positive-area material regions.
        distinct=[]
        for point in output:
            if not distinct or not np.array_equal(point,distinct[-1]):
                distinct.append(point)
        if len(distinct)>1 and np.array_equal(distinct[0],distinct[-1]):
            distinct.pop()
        item=np.asarray(distinct)
        if len(item)>=3:
            signed=_twice_area(item)
            if signed<0:raise ValueError("replay material partition inverted")
            if signed>0:result.append(item)
    return tuple(result)


def _clip_negative_gap(polygon,heights):
    # Gap is affine in material coordinates; include its moving boundary.
    values=heights[0]+polygon@(heights[1:]-heights[0])
    if np.max(values)<0:return polygon
    if np.min(values)>=0:return np.empty((0,2))
    output=[];previous=polygon[-1];old=values[-1]
    for current,value in zip(polygon,values):
        if (old<0)!=(value<0):output.append(previous+old/(old-value)*(current-previous))
        if value<0:output.append(current)
        previous,old=current,value
    return np.asarray(output).reshape(-1,2)


class S3ReplayFrictionManager:
    profile=PROFILE
    def __init__(self,model,normal,settings):
        if type(settings) not in {S3ReplayFrictionSettings,S3RimReplayFrictionSettings}:raise TypeError("replay friction settings required")
        self.normal,self.friction_settings=normal,settings
        self.template=_PartialTemplate(model,normal,settings.material)
        self.n_dof=self.template.n_dof
        self.work_columns=self.template.work_columns
        self.columns=self.template.columns
        self.reduced_directions=np.eye(len(self.work_columns))
        self.frames=()
        self._material_history_cache=OrderedDict()
        self._material_history_frames=self.frames
        self._material_history_cache_enabled=True
        self._separated_candidate_work_enabled=True
        self._separated_partition_enabled=True
        self._canonical_partition_frames=None
        self._partition_retry_count=0
        self.committed_displacement=np.zeros(self.n_dof)
        self.cumulative_dissipation=0.
        self.committed_storage=0.
        self.last_committed_summary=None
        self.translations=[];self.rotations=[]
        for node in sorted({n for a in (*self.template.source_adapters,*self.template.target_adapters.values()) for n in a.node_ids}):
            dofs=model.mesh.dof_manager.get_node_dofs(node)
            self.translations.extend(dofs[:3]);self.rotations.extend(dofs[3:])
        self.evaluation_count=0
        self.event_stage_evaluations=0
        self.event_frame_cache_hits=0
        self._event_frame_cache=OrderedDict()
        self._prepared_path_cache=OrderedDict()
        self.prepared_pressure_samples=0
        self.prepared_partial_pressure_samples=0
        self.prepared_uniform_pressure_samples=0
        self.initial_frame=self._frame(np.zeros(self.n_dof),check_path=False)
        if any(self.initial_frame["domains"]):
            raise ValueError("replay friction rejects unresolved initial penetration")

    def __getattr__(self,name):return getattr(object.__getattribute__(self,"normal"),name)

    def restrict_tangent_to_reduced_variations(self,transform):
        if not np.all(np.isfinite(transform.data)):raise ValueError("nonfinite replay reduced variations")
        self.template.restrict_tangent_to_reduced_variations(transform)
        self.columns=self.template.columns
        local=transform.tocsr()[self.work_columns].tocsc()
        nonzero=np.flatnonzero(np.diff(local.indptr))
        self.reduced_directions=local[:,nonzero].toarray()

    def _full_coverage_pressure(self,source,target,corners,n,origin):
        """Exact P1 reference moments, or None for the ordinary mortar route.

        Eligibility is strict full coverage in the *source-normal* projection
        and an exactly planar target. This changes neither the pressure space
        nor the normal law. Shared-node moments are summed before normalization.
        """
        law=self.normal.law;geometry=law.geometry
        if law.pressure_basis is not None or np.any(self.normal.normal_multipliers):return None
        if np.any((target-origin)@n!=0.):return None
        support=np.zeros(len(source));weighted=np.zeros(len(source))
        for nodes in geometry.source_face_indices:
            if len(nodes)!=3:return None
            positions=source[nodes];reference=geometry.reference_source[nodes]
            cross=np.cross(positions[1]-positions[0],positions[2]-positions[0])
            size=np.linalg.norm(cross)
            reference_cross=np.cross(reference[1]-reference[0],reference[2]-reference[0])
            if not np.isfinite(size) or size<=0:return None
            normal=cross/size;cosine=normal@n
            if cosine<=0 or normal@reference_cross<=0:return None
            gap=((positions-origin)@n)/cosine
            projected=positions-gap[:,None]*normal
            for e,adapter in self.template.target_adapters.items():
                vertices=corners[e]
                for i in range(3):
                    edge=(adapter.node_ids[i],adapter.node_ids[(i+1)%3])
                    if tuple(sorted(edge)) not in self.template.boundary_edges:continue
                    vector=vertices[(i+1)%3]-vertices[i]
                    relative=projected-vertices[i]
                    margin=128*np.finfo(float).eps*np.linalg.norm(vector)*np.maximum(
                        np.linalg.norm(relative,axis=1),np.finfo(float).tiny)
                    if np.any(np.cross(vector,relative)@n<=margin):return None
            area=np.linalg.norm(reference_cross)/2
            support[nodes]+=area/3
            weighted[nodes]+=area/12*(np.sum(gap)+gap)
        # Keep the ordinary geometry admission checks, even on this exact
        # integral specialization. Ineligible geometry still takes its owner.
        geometry._check_same_side_overlap(source,geometry.source_face_indices)
        geometry._check_same_side_overlap(target,geometry.target_face_indices)
        pressure=np.zeros_like(support);active=support>0
        pressure[active]=law.stiffness*np.maximum(-weighted[active]/support[active],0.)
        return pressure

    def _partial_planar_pressure(self,source,target,corners,n,origin):
        """Exact P1 moments using unchanged per-face overlap ownership.

        Stage-only, one S3 source and exactly planar target. Ordinary overlap
        clipping, feature merging and zero-area rules remain authoritative.
        Affine gaps allow direct moments without pressure Gram/recovery work.
        """
        from .overlap import PlanarFace,planar_projected_overlap,_to_frame
        law=self.normal.law;geometry=law.geometry
        if law.pressure_basis is not None or np.any(self.normal.normal_multipliers):return None
        if np.any((target-origin)@n!=0.):return None
        if len(geometry.source_face_indices)!=1:return None
        nodes=geometry.source_face_indices[0]
        if len(nodes)!=3:return None
        positions=source[nodes];reference=geometry.reference_source[nodes]
        source_face=PlanarFace((1,1),positions)
        edge1,edge2=positions[1:]-positions[0]
        cross=np.cross(edge1,edge2);size=np.linalg.norm(cross)
        reference_cross=np.cross(reference[1]-reference[0],reference[2]-reference[0])
        normal=cross/size;cosine=normal@n
        if cosine<=0 or normal@reference_cross<=0:return None
        geometry._check_same_side_overlap(target,geometry.target_face_indices)
        support=np.zeros(3);weighted=np.zeros(3)
        ratio=np.linalg.norm(reference_cross)/size
        for element,positions_target in corners.items():
            target_face=PlanarFace((2,int(element)),positions_target)
            overlap=planar_projected_overlap(source_face,target_face)
            if overlap is None:continue
            frame=(overlap.origin,overlap.tangent,overlap.binormal)
            xy=_to_frame(overlap.polygon,*frame)
            source_xy=_to_frame(positions,*frame);target_xy=_to_frame(positions_target,*frame)
            # Keep the ordinary shape-domain guard, including projections
            # enlarged by its geometric clipping tolerance. No gradients are
            # needed for a stage pressure query.
            values=np.array([overlap._shape_from_xy(source_xy,p)[0] for p in xy])
            target_values=np.array([overlap._shape_from_xy(target_xy,p)[0] for p in xy])
            gaps=(overlap.polygon-target_values@positions_target)@normal
            for i in range(1,len(xy)-1):
                indices=[0,i,i+1]
                triangle=overlap.polygon[indices]
                area=ratio*np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))/2
                shape=values[indices];gap=gaps[indices];total=shape.sum(axis=0)
                support+=area*total/3
                weighted+=area*(total*gap.sum()+shape.T@gap)/12
        if not np.any(support):return np.zeros(len(source))
        if np.any(support<=0) or not np.all(np.isfinite(weighted)):return None
        pressure=np.zeros(len(source))
        pressure[nodes]=law.stiffness*np.maximum(-weighted/support,0.)
        return pressure

    def _frame(self,u,pressure=None,*,check_path=True,start_pose=None):
        u=np.asarray(u,dtype=float)
        if u.shape!=(self.n_dof,) or not np.all(np.isfinite(u)):raise ValueError("replay pose needs finite authoritative coordinates")
        corners,n,frame,origin=self.template._patch(u)
        if check_path:
            path_owner=SimpleNamespace(_patch=self.template._patch,
                committed_displacement=self.committed_displacement if start_pose is None else start_pose,
                friction_settings=self.friction_settings.material,
                target_adapters=self.template.target_adapters,source_adapters=self.template.source_adapters,
                boundary_edges=self.template.boundary_edges,work_columns=self.work_columns)
            S3LinearTouchPath(path_owner,u,corners,require_full_coverage=False)
        if pressure is None:
            source,target=self.normal._positions(u)
            if not check_path:
                pressure=self._full_coverage_pressure(source,target,corners,n,origin)
                if pressure is None:
                    pressure=self._partial_planar_pressure(source,target,corners,n,origin)
            if pressure is None:
                pressure=self.normal.law.evaluate(source,target,self.normal.normal_multipliers).pressure
        pressure=np.asarray(pressure,dtype=float)
        if pressure.shape!=(len(self.normal.law.source_node_ids),) or not np.all(np.isfinite(pressure)) or np.min(pressure)<0:
            raise ValueError("replay frame pressure is invalid")
        local=u[self.work_columns].copy();local.setflags(write=False)
        p=pressure.copy();p.setflags(write=False)
        source=tuple(a.current_positions(u) for a in self.template.source_adapters)
        domains=[]
        for i,c in enumerate(source):
            xy=(c-origin)@frame[:,:2]
            inverse=np.linalg.inv((xy[1:]-xy[0]).T)
            heights=(c-origin)@n
            pieces=[]
            for target in corners.values():
                projected=((target-origin)@frame[:,:2]-xy[0])@inverse.T
                if _signed_area(projected)<0:projected=projected[::-1]
                polygon=_clip_convex(_SOURCE_UV,projected,0.,0.)
                if len(polygon)>=3:
                    polygon=_clip_negative_gap(polygon,heights)
                    if len(polygon)>=3 and _signed_area(polygon)>0:pieces.append(polygon)
            domains.append(tuple(pieces))
        for array in (*corners.values(),n,frame,origin,*source,*(p for group in domains for p in group)):
            array.setflags(write=False)
        return dict(local=local,pressure=p,target=corners,normal=n,basis=frame,origin=origin,
                    source=source,domains=tuple(domains))

    def _full(self,local):
        u=np.zeros(self.n_dof);u[self.work_columns]=local
        return u

    def _empty(self,count):
        return dict(element=np.full(count,-1,dtype=int),bary=np.zeros((count,3)),
                    slip=np.zeros(count),dissipation=np.zeros(count),storage=np.zeros(count),
                    released_storage=np.zeros(count))

    def _locate(self,x,frame):
        element=np.full(len(x),-1,dtype=int);bary=np.zeros((len(x),3))
        for e,c in frame["target"].items():
            xy=(c-c[0])@frame["basis"][:,:2]
            uv=((x-c[0])@frame["basis"][:,:2])@np.linalg.inv(xy[1:].T).T
            b=np.c_[1-uv.sum(axis=1),uv]
            active=(element<0)&(np.min(b,axis=1)>=-128*np.finfo(float).eps)
            element[active]=e;bary[active]=np.maximum(b[active],0)
            bary[active]/=bary[active].sum(axis=1)[:,None]
        return element,bary

    def _entry(self,x0,x1,first,last,*,intervals=False):
        """Last endpoint-connected active interval; hidden pulses reject."""
        n=first["normal"]
        if np.linalg.norm(n-last["normal"])>self.friction_settings.material.geometry_relative_tolerance:
            raise ValueError("replay activation requires constant-normal frames")
        origin0,origin1=first["origin"],last["origin"]
        constraint_count=len(self.template.boundary_edges)+1
        if constraint_count*len(x0)*(2*constraint_count+1)>self.friction_settings.maximum_path_samples:
            raise MemoryError("replay material-path sample cap exceeded")
        g0=(x0-origin0)@n;g1=(x1-origin1)@n
        alpha=np.zeros(len(x0))
        closing=(g0>0)&(g1<0)
        alpha[closing]=g0[closing]/(g0[closing]-g1[closing])
        vertices={};edges={};edge_area={}
        constraints=[np.array((-g0,g0-g1,np.zeros(len(g0))))]
        for e,a in self.template.target_adapters.items():
            c0,c1=first["target"][e],last["target"][e]
            for i,node in enumerate(a.node_ids):vertices[node]=(c0[i],c1[i]-c0[i])
            for i in range(3):
                edge=(a.node_ids[i],a.node_ids[(i+1)%3])
                if tuple(sorted(edge)) in self.template.boundary_edges:
                    edges[edge[0]]=edge[1]
                    edge_area[edge[0]]=max(np.linalg.norm(np.cross(c0[1]-c0[0],c0[2]-c0[0])),
                        np.linalg.norm(np.cross(c1[1]-c1[0],c1[2]-c1[0])))
        for a,b in edges.items():
            p,dp=vertices[a];q,dq=vertices[b]
            edge,de=q-p,dq-dp
            r,dr=x0-p,x1-x0-dp
            coefficients=np.array((np.cross(edge,r)@n,(np.cross(edge,dr)+np.cross(de,r))@n,np.cross(de,dr)@n))
            h0=coefficients[0];h1=coefficients.sum(axis=0)
            distance=np.maximum(np.linalg.norm(r,axis=1),np.linalg.norm(r+dr,axis=1))
            roundoff=128*np.finfo(float).eps*np.maximum(edge_area[a],
                max(np.linalg.norm(edge),np.linalg.norm(edge+de))*np.maximum(distance,np.finfo(float).tiny))
            on_start=abs(h0)<=roundoff;on_end=abs(h1)<=roundoff
            # Match the locator's roundoff ownership at an actual boundary
            # endpoint. An accepted endpoint must not be paired with an
            # impossible earlier event outside that same boundary.
            h0=np.where(on_start,0.,h0);h1=np.where(on_end,0.,h1)
            bounded=coefficients.copy()
            bounded[0]=h0;bounded[1]+=h1-bounded.sum(axis=0)
            constraints.append(bounded)
            entering=((h0<0)|((h0==0)&(coefficients[1]<0)))&(h1>=0)
            if np.any(entering):
                c,beta,a2=coefficients[:,entering]
                roots=np.zeros(len(c))
                endpoint=on_end[entering]
                roots[endpoint]=1.
                linear=(a2==0)&~endpoint
                roots[linear]=-c[linear]/beta[linear]
                nonlinear=(a2!=0)&~endpoint
                if np.any(nonlinear):
                    discriminant=beta[nonlinear]**2-4*a2[nonlinear]*c[nonlinear]
                    if np.min(discriminant)<0:raise ValueError("replay entry root is unresolved")
                    stable=-.5*(beta[nonlinear]+np.copysign(np.sqrt(discriminant),beta[nonlinear]))
                    r0=stable/a2[nonlinear];r1=c[nonlinear]/stable
                    valid0=(r0>=0)&(r0<=1)&(beta[nonlinear]+2*a2[nonlinear]*r0>0)
                    chosen=np.where(valid0,r0,r1)
                    if np.any((chosen<0)|(chosen>1)):raise ValueError("replay entry root lacks a closing bracket")
                    roots[nonlinear]=chosen
                alpha[entering]=np.maximum(alpha[entering],roots)
        regions=self._check_contact_intervals(np.asarray(constraints),return_intervals=intervals)
        return (alpha,regions) if intervals else alpha

    def _check_contact_intervals(self,coefficients,*,return_intervals=False):
        """Intersect every convex-patch half-plane and the closing gap."""
        edges,_,points=coefficients.shape
        if edges*points*(2*edges+1)>self.friction_settings.maximum_path_samples:
            raise MemoryError("replay material-path sample cap exceeded")
        if not np.all(np.isfinite(coefficients)):raise ValueError("nonfinite replay path coefficients")
        c,b,a=coefficients[:,0],coefficients[:,1],coefficients[:,2]
        roots=np.ones((2*edges+2,points));roots[0]=0.
        for i in range(edges):
            linear=(a[i]==0)&(b[i]!=0)
            first=np.ones(points);second=np.ones(points)
            first[linear]=-c[i,linear]/b[i,linear]
            quadratic=a[i]!=0
            discriminant=b[i]**2-4*a[i]*c[i]
            valid=quadratic&(discriminant>=0)
            stable=-.5*(b[i,valid]+np.copysign(np.sqrt(discriminant[valid]),b[i,valid]))
            first[valid]=stable/a[i,valid]
            nonzero=stable!=0
            indices=np.flatnonzero(valid)
            second[indices[nonzero]]=c[i,indices[nonzero]]/stable[nonzero]
            roots[2*i+1]=np.where((first>0)&(first<1),first,1.)
            roots[2*i+2]=np.where((second>0)&(second<1),second,1.)
        roots.sort(axis=0)
        left,right=roots[:-1],roots[1:]
        middle=(left+right)/2
        value=c[:,None,:]+middle[None,:,:]*(b[:,None,:]+middle[None,:,:]*a[:,None,:])
        active=np.all(value>=0,axis=0)&(value[0]>0)&(right>left)
        previous=np.zeros(points,dtype=bool);regions=np.zeros(points,dtype=int)
        for row,width in zip(active,right-left):
            regions+=row&~previous
            previous=np.where(width>0,row,previous)
        start=np.all(c>=0,axis=0)&(c[0]>0)
        finish=np.all(coefficients.sum(axis=1)>=0,axis=0)&(coefficients[0].sum(axis=0)>0)
        unresolved=(regions>1)|((regions>0)&~start&~finish)
        if return_intervals or np.any(unresolved):
            intervals=[]
            selected=np.arange(points) if return_intervals else np.flatnonzero(unresolved)
            for point in selected:
                begin=None;end=None
                for lo,hi,on in zip(left[:,point],right[:,point],active[:,point]):
                    if hi<=lo:continue
                    if on:
                        if begin is None:begin=float(lo)
                        end=float(hi)
                    elif begin is not None:
                        intervals.append((point,begin,end));begin=end=None
                if begin is not None:intervals.append((point,begin,end))
            if return_intervals:return tuple(intervals)
            raise ReplayContactSubdivisionRequired(intervals)

    def _step(self,source_index,bary,old,first,last):
        if self.friction_settings.event_subdivisions:
            return self._step_events(source_index,bary,old,first,last)
        return self._step_endpoint(source_index,bary,old,first,last)

    def _step_endpoint(self,source_index,bary,old,first,last,*,path_partitioned=False):
        x0=bary@first["source"][source_index];x=bary@last["source"][source_index]
        n=last["normal"]
        gap=(x-last["origin"])@n
        current,current_bary=self._locate(x,last)
        active=(gap<0)&(current>=0)
        fresh=active&(old["element"]<0)
        # The event driver has already checked the complete affine path and
        # samples every active interval and intervening open interval. Its
        # substeps need roots again only to initialize a fresh material anchor.
        # Ordinary endpoint calls still check even open/open paths for pulses.
        if not path_partitioned or np.any(fresh):
            try:
                entry=self._entry(x0,x,first,last)
            except ReplayContactSubdivisionRequired as event:
                event.source_index=int(source_index)
                event.start_local=tuple(map(float,first["local"]))
                event.end_local=tuple(map(float,last["local"]))
                raise
        state={k:v.copy() for k,v in old.items()}
        state["released_storage"]+=np.where(~active,old["storage"],0.)
        state["element"][~active]=-1;state["bary"][~active]=0;state["storage"][~active]=0
        anchor=np.zeros_like(x)
        for e,c in last["target"].items():
            mask=active&(old["element"]==e)
            anchor[mask]=old["bary"][mask]@c
        if np.any(fresh):
            alpha=entry[fresh]
            events=np.flatnonzero(fresh)
            # At most four edges and two target facets; per-event solve is
            # bounded. Replay/force work is otherwise batched over all queries.
            for j,t in zip(events,alpha):
                stage=dict(last,target={e:c+t*(last["target"][e]-c) for e,c in first["target"].items()})
                event=x0[j]+t*(x[j]-x0[j])
                element,coordinates=self._locate(event[None,:],stage)
                if element[0]<0:
                    raise ValueError(f"replay first-touch stage cannot locate its material anchor at fraction={t:.17g}")
                anchor[j]=coordinates[0]@last["target"][element[0]]
        trial=x-anchor;trial-=np.outer(trial@n,n)
        length=np.linalg.norm(trial,axis=1)
        adapter=self.template.source_adapters[source_index]
        indices=[self.normal.law.source_node_ids.index(node) for node in adapter.node_ids]
        pressure=np.maximum(0,bary@last["pressure"][indices])
        with np.errstate(over="ignore",invalid="ignore"):
            cap=self.friction_settings.material.friction_coefficient*pressure
        if not np.all(np.isfinite(cap)):raise ValueError("nonfinite replay Coulomb cap")
        kt=self.friction_settings.material.tangential_stiffness_Pa_per_m
        slipping=active&(kt*length>cap)
        elastic=trial.copy();elastic[~active]=0
        elastic[slipping]*=(cap[slipping]/(kt*length[slipping]))[:,None]
        returned=x-np.outer(gap,n)-elastic
        element,coordinates=self._locate(returned,last)
        if np.any(active&(element<0)):raise ValueError("replay return left its declared target patch")
        moved=returned-anchor;moved-=np.outer(moved@n,n)
        plastic=np.where(slipping,np.linalg.norm(moved,axis=1),0.)
        state["element"][active]=element[active];state["bary"][active]=coordinates[active]
        state["slip"]+=plastic;increment=cap*plastic;state["dissipation"]+=increment
        state["storage"]=.5*kt*np.einsum('ij,ij->i',elastic,elastic)
        traction=-kt*elastic
        if any(not np.all(np.isfinite(v)) for v in (*state.values(),traction,increment,returned)):
            raise ValueError("nonfinite replay material response")
        return state,traction,increment,gap,returned,np.where(active,pressure,0.)

    def _stage_frames(self,first,last,fractions,segment_key,prepared):
        batch_size=self.friction_settings.maximum_prepared_stage_batch
        for offset in range(0,len(fractions),batch_size):
            chunk=fractions[offset:offset+batch_size]
            route="prepared" if prepared is not None else "ordinary"
            keys={t:(route,*segment_key,float(t)) for t in chunk if t not in (0.,1.)}
            needed=tuple(t for t,key in keys.items() if key not in self._event_frame_cache)
            # Retain this bounded chunk's existing entries even if later LRU
            # insertions evict them before their turn in the stage sequence.
            built={t:self._event_frame_cache[key] for t,key in keys.items()
                   if key in self._event_frame_cache}
            if prepared is not None and needed:
                if self.prepared_pressure_samples+len(needed)>self.friction_settings.maximum_path_samples:
                    raise MemoryError("replay prepared-pressure sample cap exceeded")
                self.prepared_pressure_samples+=len(needed)
                if not prepared.full_coverage:
                    if self.prepared_partial_pressure_samples+len(needed)>self.friction_settings.maximum_prepared_partial_pressure_samples:
                        raise MemoryError("replay prepared partial-pressure sample cap exceeded")
                    self.prepared_partial_pressure_samples+=len(needed)
                built.update(zip(needed,prepared.frames(needed)))
            for fraction in chunk:
                if fraction in (0.,1.):
                    yield first if fraction==0. else last
                    continue
                key=keys[fraction]
                if key in self._event_frame_cache:
                    frame=self._event_frame_cache[key]
                    self._event_frame_cache.move_to_end(key);self.event_frame_cache_hits+=1
                else:
                    if fraction in built and built[fraction] is not None:
                        frame=built[fraction]
                        if fraction not in needed:self.event_frame_cache_hits+=1
                    else:
                        if self.event_stage_evaluations>=self.friction_settings.maximum_event_stages:
                            raise MemoryError("replay actual-pressure event stage cap exceeded")
                        self.event_stage_evaluations+=1
                        local=first["local"]+fraction*(last["local"]-first["local"])
                        frame=self._frame(self._full(local),check_path=False)
                    self._event_frame_cache[key]=frame
                    if len(self._event_frame_cache)>self.friction_settings.maximum_event_stages:
                        self._event_frame_cache.popitem(last=False)
                yield frame

    def _step_events(self,source_index,bary,old,first,last):
        """Pointwise event partition with actual stage mortar pressure.

        The partition depends only on a material point and its immutable pose
        segment, never on the other queries in a quadrature batch. Midpoint
        sampling of each active interval defines a refinable return sequence;
        it is not an exact continuous-time dissipation claim.
        """
        count=len(bary);subdivisions=self.friction_settings.event_subdivisions
        x0=bary@first["source"][source_index];x1=bary@last["source"][source_index]
        _,intervals=self._entry(x0,x1,first,last,intervals=True)
        by_point=[[] for _ in range(count)]
        for point,start,end in intervals:by_point[point].append((start,end))
        stages=0
        segment_key=(first["local"].tobytes(),last["local"].tobytes())
        prepared=None
        if self.friction_settings.prepared_pressure_paths:
            if segment_key not in self._prepared_path_cache:
                from .s3_friction_prepared_path import prepare_s3_plane_pressure_path
                self._prepared_path_cache[segment_key]=prepare_s3_plane_pressure_path(self,first,last,
                    allow_partial=self.friction_settings.prepared_partial_pressure_paths)
                if len(self._prepared_path_cache)>self.friction_settings.maximum_history_frames:
                    self._prepared_path_cache.popitem(last=False)
            self._prepared_path_cache.move_to_end(segment_key)
            prepared=self._prepared_path_cache[segment_key]
        state={k:v.copy() for k,v in old.items()}
        traction=np.zeros((count,3));gap=np.zeros(count)
        returned=np.zeros((count,3));pressure=np.zeros(count)
        groups={}
        for point,regions in enumerate(by_point):
            fractions=[];previous=0.
            for start,end in regions:
                if start>previous:fractions.append(previous+(start-previous)/2)
                fractions.extend(start+(end-start)*(j+.5)/subdivisions
                                 for j in range(subdivisions))
                previous=end
            if previous<1.:fractions.append(previous+(1.-previous)/2)
            fractions.append(1.)
            if len(fractions)+stages>self.friction_settings.maximum_path_samples:
                raise MemoryError("replay event return-map stage cap exceeded")
            stages+=len(fractions)
            groups.setdefault(tuple(fractions),[]).append(point)
        for fractions,indices in groups.items():
            indices=np.asarray(indices,dtype=int)
            current={k:v[indices].copy() for k,v in old.items()}
            initial=first
            for final in self._stage_frames(first,last,fractions,segment_key,prepared):
                result=self._step_endpoint(source_index,bary[indices],current,initial,final,path_partitioned=True)
                current=result[0];initial=final
            for key in state:state[key][indices]=current[key]
            traction[indices]=result[1];gap[indices]=result[3]
            returned[indices]=result[4];pressure[indices]=result[5]
        increment=state["dissipation"]-old["dissipation"]
        return state,traction,increment,gap,returned,pressure

    def query_material(self,source_index,bary,*,candidate=None):
        # A public query owns a fresh disposable budget. Internal integration
        # queries share the enclosing evaluation's budget across all faces.
        self.event_stage_evaluations=0
        self.event_frame_cache_hits=0
        self.prepared_pressure_samples=0
        self.prepared_partial_pressure_samples=0
        self.prepared_uniform_pressure_samples=0
        return self._query_material(source_index,bary,candidate=candidate)

    def _query_material(self,source_index,bary,*,candidate=None):
        b=np.asarray(bary,dtype=float)
        if b.ndim!=2 or b.shape[1]!=3 or np.min(b)<0 or not np.all(np.isfinite(b)) or np.max(abs(b.sum(axis=1)-1))>1e-12:
            raise ValueError("replay queries need positive source material coordinates")
        if candidate is not None or not self.frames or not self._material_history_cache_enabled:
            return self._replay_material(source_index,b,candidate=candidate)
        # Frames are immutable physical commits. Keep the tuple itself so that
        # replacement/restore cannot alias a recycled object identity.
        if self._material_history_frames is not self.frames:
            self._material_history_cache.clear()
            self._material_history_frames=self.frames
        state=self._empty(len(b));missing=OrderedDict()
        for index,point in enumerate(b):
            key=(source_index,point.tobytes())
            cached=self._material_history_cache.get(key)
            if cached is None:
                missing.setdefault(key,[]).append(index)
            else:
                self._material_history_cache.move_to_end(key)
                for name in state:state[name][index]=cached[name]
        if missing:
            indices=[positions[0] for positions in missing.values()]
            replay=self._replay_material(source_index,b[indices])
            for row,(key,positions) in enumerate(missing.items()):
                cached={name:np.array(value[row],copy=True) for name,value in replay.items()}
                for name,value in cached.items():
                    value.setflags(write=False)
                    state[name][positions]=value
                self._material_history_cache[key]=cached
                if len(self._material_history_cache)>self.friction_settings.maximum_integration_points:
                    self._material_history_cache.popitem(last=False)
        return state

    def _replay_material(self,source_index,b,*,candidate=None):
        """Uncached replay retained as the verification oracle."""
        state=self._empty(len(b));first=self.initial_frame
        result=None
        for frame in (*self.frames, *((candidate,) if candidate is not None else ())):
            result=self._step(source_index,b,state,first,frame)
            state=result[0];first=frame
        return state if candidate is None else result

    def _source_strictly_separated(self,source_index):
        if not self.frames:return False
        frame=self.frames[-1]
        vertices=frame['source'][source_index]
        normal=frame['normal'];origin=frame['origin']
        gaps=(vertices-origin)@normal
        # Affine S3 geometry: all material gaps are convex combinations of
        # vertex gaps. Decline cancellation-sensitive or touching geometries.
        scale=np.max((abs(vertices)+abs(origin))@abs(normal))
        margin=128*np.finfo(float).eps*max(scale,np.finfo(float).tiny)
        return bool(np.all(np.isfinite(gaps)) and np.all(gaps>margin))

    def _integration_partition_frames(self,frame,full,partition_history):
        return (*self.frames,frame) if full else (self.frames if partition_history else ())

    def _quadrature(self,source_index,frame,level,*,full=False,partition_history=True):
        cells=[_rational_polygon(p) for p in ([_SOURCE_UV] if full else frame["domains"][source_index])]
        # Historical contact boundaries are fixed material lines, not moving
        # point indices; partition discontinuous release/activation history.
        history=self._integration_partition_frames(frame,full,partition_history)
        for past in history:
            for region in past["domains"][source_index]:
                for a,b in zip(region,np.roll(region,-1,axis=0)):
                    cells=[item for cell in cells for item in _split(cell,a,b)]
                    if len(cells)>self.friction_settings.maximum_partition_cells:raise MemoryError("replay history partition cap exceeded")
        points=[];weights=[]
        rule=_reference_quadrature(level)
        adapter=self.template.source_adapters[source_index]
        ref=adapter.reference_positions
        area=np.linalg.norm(np.cross(ref[1]-ref[0],ref[2]-ref[0]))/2
        for cell in cells:
            # Area centroid is continuous when clipping adds a duplicate
            # vertex; a vertex-average centre would jump and change the
            # quadrature of a finite-area cell under an infinitesimal probe.
            centre=_area_centroid(cell)
            for a,b in zip(cell,np.roll(cell,-1,axis=0)):
                uv=np.array((centre,a,b))
                signed=_twice_area(uv)
                # Collinear boundary vertices have exactly zero measure.
                # Retain arbitrarily small positive triangles; reject inversion.
                if signed==0:continue
                if signed<0:raise ValueError("replay reference integration triangle is invalid")
                measure=float(signed)*area
                if measure<=0:raise ValueError("replay positive reference measure underflowed")
                vertices=np.asarray(np.c_[1-uv.sum(axis=1),uv],dtype=float)
                if len(points)+len(rule)>self.friction_settings.maximum_integration_points:
                    raise MemoryError("replay integration-point cap exceeded")
                for bary,fraction in rule:
                    material=np.asarray(bary)@vertices
                    if np.min(material)<-128*np.finfo(float).eps:
                        raise ValueError("replay quadrature left its reference material triangle")
                    # Only the conversion of exact reference coordinates to
                    # FE doubles is rounded. Keep the entire cell measure.
                    material=np.maximum(material,0);material/=material.sum()
                    points.append(material);weights.append(measure*fraction)
        if len(points)>self.friction_settings.maximum_integration_points:raise MemoryError("replay integration-point cap exceeded")
        return np.asarray(points).reshape(-1,3),np.asarray(weights)

    def _integrate(self,u,*,level=None,full=False):
        self.evaluation_count+=1
        self.event_stage_evaluations=0
        self.event_frame_cache_hits=0
        self.prepared_pressure_samples=0
        self.prepared_partial_pressure_samples=0
        self.prepared_uniform_pressure_samples=0
        frame=self._frame(u)
        local=np.zeros(len(self.work_columns));increment=storage=release=0.;count=0
        level=self.friction_settings.integration_refinement if level is None else level
        corners,n=frame["target"],frame["normal"]
        Dn=self.template._normal_variation(corners,n,local=True)
        for i,adapter in enumerate(self.template.source_adapters):
            # Separation resets anchors/storage. Additive historical offsets
            # do not enter force; preserve canonical ledger rounding for commit.
            separated=(not full and self._separated_candidate_work_enabled
                       and self._source_strictly_separated(i))
            bary,weight=self._quadrature(i,frame,level,full=full,
                partition_history=not (separated and self._separated_partition_enabled
                                       and self._canonical_partition_frames is not self.frames));count+=len(weight)
            if count>self.friction_settings.maximum_integration_points:
                raise MemoryError("replay total integration-point cap exceeded")
            if not len(weight):continue
            old=self._empty(len(bary)) if separated else self._query_material(i,bary)
            state,t,diss,gap,returned,_pressure=self._step(i,bary,old,
                self.frames[-1] if self.frames else self.initial_frame,frame)
            projected=t-np.outer(t@n,n)
            gradient=np.einsum('q,qi,qj->ij',weight,bary,-projected)
            local+=np.asarray(adapter.work_map[:,self.work_columns].T@gradient.ravel()).ravel()
            for e,target in self.template.target_adapters.items():
                mask=state["element"]==e
                physical=np.einsum('q,qi,qj->ij',weight[mask],state["bary"][mask],projected[mask])
                local+=np.asarray(target.work_map[:,self.work_columns].T@physical.ravel()).ravel()
            x=bary@frame["source"][i];r=x-returned
            correction=np.einsum('q,qj->j',weight,r*(t@n)[:,None]+gap[:,None]*t)
            local+=Dn.T@correction
            increment+=float(weight@diss);storage+=float(weight@state["storage"])
            release+=float(weight@(state["released_storage"]-old["released_storage"]))
        force=np.zeros(self.n_dof);force[self.work_columns]=local
        return force,frame,dict(dissipation_increment_j=increment,storage_j=storage,released_storage_j=release,
                              integration_points=count,event_pressure_evaluations=self.event_stage_evaluations,
                              event_frame_cache_hits=self.event_frame_cache_hits,
                              prepared_pressure_samples=self.prepared_pressure_samples,
                              prepared_partial_pressure_samples=self.prepared_partial_pressure_samples,
                              prepared_uniform_pressure_samples=self.prepared_uniform_pressure_samples)

    def _check_integration(self,force,summary,control,control_summary):
        if any(not np.all(np.isfinite(v)) for v in (force,control,*summary.values(),*control_summary.values())):
            raise ValueError("nonfinite replay integrated response")
        difference=control-force
        if (np.linalg.norm(difference[self.translations])>self.friction_settings.force_integration_tolerance_n or
                np.linalg.norm(difference[self.rotations])>self.friction_settings.moment_integration_tolerance_nm):
            raise ValueError("replay friction integration controls are unresolved")
        for field in ("dissipation_increment_j","storage_j","released_storage_j"):
            if abs(summary[field]-control_summary[field])>self.friction_settings.work_integration_tolerance_j:
                raise ValueError("replay friction work integration controls are unresolved")

    def evaluate(self,u,*,tangent=True):
        needed=4*self.reduced_directions.shape[1]+2 if tangent else 2
        cap=self.friction_settings.material.maximum_candidate_evaluations
        if needed>cap:raise MemoryError("replay tangent evaluation cap exceeded")
        try:
            return self._evaluate_once(u,tangent=tangent)
        except _ReplayTangentUnresolved as error:
            optimized=(self._separated_candidate_work_enabled and self._separated_partition_enabled
                       and self._canonical_partition_frames is not self.frames
                       and any(self._source_strictly_separated(i)
                               for i in range(len(self.template.source_adapters))))
            if not optimized:raise
            if 2*needed>cap:
                raise MemoryError("replay canonical partition retry exceeds candidate evaluation cap") from error
            previous=self._canonical_partition_frames
            self._canonical_partition_frames=self.frames
            self._partition_retry_count+=1
            succeeded=False
            try:
                # Hold the canonical partition policy for this physical origin,
                # including subsequent residual-only line-search evaluations.
                result=self._evaluate_once(u,tangent=tangent,comparison=(error.force,error.summary))
                succeeded=True
                return result
            finally:
                # Cancellation also invalidates an unverified retry policy.
                if not succeeded:self._canonical_partition_frames=previous

    def _tangent_probe_force(self,u):
        return self._integrate(u)[0]

    def _evaluate_once(self,u,*,tangent=True,comparison=None):
        directions=self.reduced_directions
        if tangent and 4*directions.shape[1]+2>self.friction_settings.material.maximum_candidate_evaluations:
            raise MemoryError("replay tangent evaluation cap exceeded")
        normal,K,certificate=self.normal.evaluate(u,tangent=tangent)
        force,_frame,summary=self._integrate(u)
        control,_control_frame,control_summary=self._integrate(u,level=self.friction_settings.integration_refinement+1)
        self._check_integration(force,summary,control,control_summary)
        if comparison is not None:self._check_integration(*comparison,force,summary)
        if tangent:
            derivative=np.zeros_like(directions)
            coarse=np.zeros_like(derivative)
            limits=np.asarray(self.friction_settings.material.steps_by_dof)[self.work_columns]
            for j,direction in enumerate(directions.T):
                nonzero=direction!=0
                step=np.min(limits[nonzero]/abs(direction[nonzero]))
                delta=np.zeros(self.n_dof);delta[self.work_columns]=step*direction
                plus=self._tangent_probe_force(np.asarray(u)+delta);minus=self._tangent_probe_force(np.asarray(u)-delta)
                coarse[:,j]=((plus-minus)/(2*step))[self.work_columns]
                plus=self._tangent_probe_force(np.asarray(u)+delta/2);minus=self._tangent_probe_force(np.asarray(u)-delta/2)
                derivative[:,j]=((plus-minus)/step)[self.work_columns]
            if np.linalg.norm(derivative-coarse)>1e-5*max(np.linalg.norm(derivative),np.finfo(float).tiny):
                raise _ReplayTangentUnresolved(f"replay friction tangent probes are unresolved: relative={np.linalg.norm(derivative-coarse)/max(np.linalg.norm(derivative),np.finfo(float).tiny):.8g}, norm={np.linalg.norm(derivative):.8g}, committed_frames={len(self.frames)}",force,summary)
            if not np.all(np.isfinite(derivative)):raise ValueError("nonfinite replay tangent")
            # A minimum-norm extension supplies a full-coordinate assembly
            # block, whose action on every admitted reduced direction is the
            # measured Jacobian. No off-manifold coordinate probes are used.
            extension=derivative@np.linalg.pinv(directions)
            if np.linalg.norm(extension@directions-derivative)>1e-5*max(np.linalg.norm(derivative),np.finfo(float).tiny):
                raise ValueError("replay reduced tangent extension is unresolved")
            rows,columns=np.nonzero(extension)
            K=K+sparse.coo_matrix((extension[rows,columns],(self.work_columns[rows],self.work_columns[columns])),shape=K.shape).tocsr()
        return normal+force,K,certificate

    def prepare_commit(self,u):
        if len(self.frames)>=self.friction_settings.maximum_history_frames:raise MemoryError("replay committed history-frame cap exceeded")
        force,frame,summary=self._integrate(u,full=True)
        control,_control_frame,control_summary=self._integrate(u,full=True,level=self.friction_settings.integration_refinement+1)
        self._check_integration(force,summary,control,control_summary)
        # Disposable cache work varies with restart/cache warmth. Keep it on
        # the evaluation diagnostics, outside accepted physical step history.
        summary.pop("event_pressure_evaluations")
        summary.pop("event_frame_cache_hits")
        summary.pop("prepared_pressure_samples")
        summary.pop("prepared_partial_pressure_samples")
        summary.pop("prepared_uniform_pressure_samples")
        summary.update(profile=self.profile,history_frames=len(self.frames)+1,
                       traction_measure="force_per_source_reference_area",
                       dissipation_measure="endpoint_pressure_current_material_chord",
                       storage_change_j=summary["storage_j"]-self.committed_storage,
                       cumulative_dissipation_j=self.cumulative_dissipation+summary["dissipation_increment_j"])
        if not np.all(np.isfinite((summary["cumulative_dissipation_j"],summary["storage_change_j"]))):
            raise ValueError("nonfinite replay cumulative work")
        return np.asarray(u).copy(),frame,summary

    def commit(self,prepared):
        u,frame,summary=prepared
        self.frames=(*self.frames,frame);self.committed_displacement=u
        self._canonical_partition_frames=None
        self.cumulative_dissipation=summary["cumulative_dissipation_j"]
        self.committed_storage=summary["storage_j"];self.last_committed_summary=summary

    def restart_contract(self):
        return dict(normal=self.normal.restart_contract(),friction_profile=self.profile,settings=asdict(self.friction_settings),
                    history=("material_event_partition_actual_pressure_midpoint_return_v2"
                             if self.friction_settings.event_subdivisions else
                             "replay_defined_material_coordinate_endpoint_return_v1"),
                    tangent="reduced_direction_minimum_norm_extension_with_resolution_v1",
                    integration="moving_material_exact_partition_area_centroid_v2")

    def checkpoint_state(self):
        return dict(displacement=self.committed_displacement.tolist(),
                    frames=[dict(local=f["local"].tolist(),pressure=f["pressure"].tolist()) for f in self.frames],
                    cumulative_dissipation_j=self.cumulative_dissipation,storage_j=self.committed_storage)

    def restore(self,state,displacement):
        if type(state) is not dict or set(state)!={"displacement","frames","cumulative_dissipation_j","storage_j"}:
            raise ValueError("invalid replay checkpoint")
        if type(state["frames"]) is not list:raise ValueError("replay checkpoint frames must be a list")
        u=np.asarray(state["displacement"],dtype=float)
        if u.shape!=(self.n_dof,) or not np.array_equal(u,displacement) or not np.all(np.isfinite(u)):
            raise ValueError("replay checkpoint pose mismatch")
        if len(state["frames"])>self.friction_settings.maximum_history_frames:raise MemoryError("replay restart frame cap exceeded")
        if not state["frames"] and np.any(u[self.work_columns]):
            raise ValueError("replay empty history cannot carry a contact pose")
        # Recompute the ledger on a disposable owner. Neither invalid frames
        # nor invalid work may modify the live physical state.
        scratch=copy(self)
        scratch._event_frame_cache=OrderedDict()
        scratch._material_history_cache=OrderedDict()
        scratch._material_history_frames=None
        scratch._canonical_partition_frames=None
        scratch._prepared_path_cache=OrderedDict()
        scratch.frames=();scratch.committed_displacement=np.zeros(self.n_dof)
        scratch.cumulative_dissipation=0.;scratch.committed_storage=0.
        previous=np.zeros(self.n_dof)
        for item in state["frames"]:
            if set(item)!={"local","pressure"} or np.asarray(item["local"]).shape!=(len(self.work_columns),):raise ValueError("invalid replay frame")
            pose=self._full(np.asarray(item["local"]))
            actual=self._frame(pose,start_pose=previous)
            supplied=np.asarray(item["pressure"])
            if supplied.shape!=actual["pressure"].shape or not np.allclose(supplied,actual["pressure"],rtol=1e-10,atol=1e-12):
                raise ValueError("replay checkpoint pressure disagrees with its physical pose")
            scratch.commit(scratch.prepare_commit(pose));previous=pose
        if scratch.frames and not np.array_equal(scratch.frames[-1]["local"],u[self.work_columns]):raise ValueError("replay final frame pose mismatch")
        values=np.asarray((state["cumulative_dissipation_j"],state["storage_j"]),dtype=float)
        if not np.all(np.isfinite(values)) or np.min(values)<0:raise ValueError("replay ledger is invalid")
        if not np.allclose(values,(scratch.cumulative_dissipation,scratch.committed_storage),rtol=1e-10,atol=0.):
            raise ValueError("replay checkpoint work disagrees with its material history")
        self.frames=scratch.frames;self.committed_displacement=u.copy()
        self._canonical_partition_frames=None
        self.cumulative_dissipation,self.committed_storage=values
        self.last_committed_summary=None


@dataclass(frozen=True)
class S3RimReplayFrictionSettings(S3ReplayFrictionSettings):
    """Opt-in partial physical-rim replay with one full-reference partition.

    Residual, tangent, recovery and accepted work use identical integration
    domains. History is the immutable accepted-stage material-coordinate return
    function; no state interpolation or candidate quadrature-index identity.
    This first row requires parallel axis-aligned source/target planes, a fixed
    target perpendicular coordinate, and a fixed second tangential coordinate
    (slide axis is the lowest Cartesian index other than the normal axis).
    It does not
    establish generic event-free history or continuous energy integration.
    """


class S3RimReplayFrictionManager(S3ReplayFrictionManager):
    profile='S3_PARTIAL_RIM_COMMON_PARTITION_REPLAY_STATIC_DEV_V0'

    def __init__(self,model,normal,settings):
        super().__init__(model,normal,settings)
        if len(self.template.source_adapters)!=1 or len(self.template.target_adapters)>2:raise ValueError('rim replay first row requires one S3 source and at most two conforming target faces')
        from .s3_friction_mesh_monolithic import _geometry_identity
        self._rim_geometry=_geometry_identity((normal.settings,normal.profile,normal.reference,normal.map,normal.law.geometry,
            self.template.source_adapters,self.template.target_adapters,self.friction_settings))
        self._rim_path_cache=OrderedDict()
        self.require_friction_path(np.zeros(self.n_dof),np.zeros(self.n_dof))

    @property
    def certified_material_path(self):return True

    def require_friction_path(self,first,last):
        from .s3_friction_rim_path import require_fixed_plane_path
        from .s3_friction_mesh_monolithic import _geometry_identity
        identity=_geometry_identity((self.normal.settings,self.normal.profile,self.normal.reference,self.normal.map,self.normal.law.geometry,
            self.template.source_adapters,self.template.target_adapters,self.friction_settings))
        if identity!=self._rim_geometry:raise ValueError('rim prepared geometry/settings mutated')
        return require_fixed_plane_path(self.template,first,last)

    def _frame(self,u,**kwargs):
        from .s3_friction_rim_path import require_fixed_plane_path
        if kwargs.get('check_path',True):
            from .s3_friction_mesh_monolithic import _geometry_identity
            identity=_geometry_identity((self.normal.settings,self.normal.profile,self.normal.reference,self.normal.map,self.normal.law.geometry,
                self.template.source_adapters,self.template.target_adapters,self.friction_settings))
            if '_rim_geometry' in self.__dict__ and identity!=self._rim_geometry:raise ValueError('rim prepared geometry/settings mutated')
            first=kwargs.get('start_pose',self.committed_displacement)
            key=(np.asarray(first,float).tobytes(),np.asarray(u,float).tobytes())
            cache=self.__dict__.get('_rim_path_cache',{})
            if key not in cache:
                require_fixed_plane_path(self.template,first,u)
                if '_rim_path_cache' in self.__dict__:
                    cache[key]=True
                    if len(cache)>16:cache.popitem(last=False)
        return super()._frame(u,**kwargs)

    def _integrate(self,u,*,level=None,full=False):
        # Include zero-coverage regions for release/storage and use this same
        # partition for physical force, its derivative and accepted work.
        return super()._integrate(u,level=level,full=True)

    def _integration_partition_frames(self,frame,full,partition_history):
        history=super()._integration_partition_frames(frame,full,partition_history)
        additional=[];first=self.initial_frame
        for last in (*self.frames,frame):
            # Uniform parallel-plane normal gap has at most one affine root.
            # Coverage at that root separates material born by normal closure
            # from material entering later through the rim. It is an integration
            # partition only, never an intermediate physical/history commit.
            g0=float((first['source'][0][0]-first['origin'])@first['normal'])
            g1=float((last['source'][0][0]-last['origin'])@last['normal'])
            if (g0<0<g1) or (g1<0<g0):
                fraction=float(-Fraction(g0)/(Fraction(g1)-Fraction(g0)))
                source=first['source'][0]+fraction*(last['source'][0]-first['source'][0])
                target={e:c+fraction*(last['target'][e]-c) for e,c in first['target'].items()}
                origin=first['origin']+fraction*(last['origin']-first['origin']);basis=first['basis'][:,:2]
                xy=(source-origin)@basis;inverse=np.linalg.inv((xy[1:]-xy[0]).T);pieces=[]
                for corners in target.values():
                    projected=((corners-origin)@basis-xy[0])@inverse.T
                    if _signed_area(projected)<0:projected=projected[::-1]
                    polygon=_clip_convex(_SOURCE_UV,projected,0.,0.)
                    if len(polygon)>=3 and _signed_area(polygon)>0:pieces.append(polygon)
                additional.append(dict(domains=(tuple(pieces),)))
                if len(additional)>self.friction_settings.maximum_history_frames+1:raise MemoryError('rim normal-birth partition cap exceeded')
            first=last
        return (*history,*additional)

    def _evaluate_once(self,u,**kwargs):
        previous=self.__dict__.get('_rim_empty_probe_origin',False)
        self._rim_empty_probe_origin=not any(self._frame(u)['domains'])
        try:return super()._evaluate_once(u,**kwargs)
        finally:self._rim_empty_probe_origin=previous

    def _tangent_probe_force(self,u):
        if self.__dict__.get('_rim_empty_probe_origin',False):
            frame=self._frame(u)
            if not any(frame['domains']):
                count=0;inactive=True
                for i in range(len(self.template.source_adapters)):
                    bary,weight=self._quadrature(i,frame,self.friction_settings.integration_refinement,full=True)
                    count+=len(weight)
                    if count>self.friction_settings.maximum_integration_points:raise MemoryError('replay total integration-point cap exceeded')
                    x=bary@frame['source'][i]
                    element,_=self._locate(x,frame)
                    gap=(x-frame['origin'])@frame['normal']
                    if np.any((gap<0)&(element>=0)):inactive=False;break
                # This is the exact discrete force predicate used by _step:
                # every current point inactive => traction and complete work
                # gradient are zero, independent of old slip/storage. The
                # full base/control/commit ledger remains evaluated normally.
                if inactive:return np.zeros(self.n_dof)
        previous=self.__dict__.get('_rim_force_only_open_origin',False)
        self._rim_force_only_open_origin=True
        try:return self._integrate(u)[0]
        finally:self._rim_force_only_open_origin=previous

    def _query_material(self,source_index,bary,*,candidate=None):
        if (candidate is None and self.__dict__.get('_rim_force_only_open_origin',False)
                and self._source_strictly_separated(source_index)):
            # Strictly open last committed face releases every anchor/storage.
            # Past additive plastic-loss/slip counters do not enter traction.
            # Only force-only probes use this origin; all work/commit/recovery
            # queries still replay and retain their complete physical ledger.
            return self._empty(len(bary))
        return super()._query_material(source_index,bary,candidate=candidate)

    def prepare_commit(self,u):
        pose,frame,summary=super().prepare_commit(u)
        summary.update(integration_domain='full_reference_candidate_commit_normal_birth_partition_v2',
                       history_field='immutable_accepted_stage_replay_no_interpolation',
                       event_scope='bounded_discrete_replay_not_generic_event_free_or_path_energy')
        return pose,frame,summary

    def restart_contract(self):
        contract=super().restart_contract()
        contract.update(integration='full_reference_candidate_commit_normal_birth_partition_v2',
                        geometry_path='rounded_parallel_axis_planes_oriented_convex_target_v1',
                        structural_state_mode='validated_owned_mapping_invocation_publication_v1')
        return contract
