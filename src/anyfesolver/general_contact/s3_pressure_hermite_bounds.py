"""Reference P1 mortar pressure-phase bounds on supplied physical Hermite skins.

Original source functions and reference measure are retained on spatial children.
Exact rational Bernstein polynomials bound target barycentric coordinates, source
normal projection and weighted gaps. Boundary cells are never treated as outside
because different vertices lie outside different target edges. Aggregate WÂ±tau*S
is tested before normalization, including arbitrarily small/zero support.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from heapq import heappop,heappush
from math import inf
import numpy as np
from .shell_pair_hermite_sweep import (
    _coordinates, _limit, _nodes, _normal, _dot, _cross, _sub, _split, _lower_float)
from .plane_newmark_events import _sqrt_interval


@dataclass(frozen=True)
class S3PressureHermiteBounds:
    success: bool
    status: str
    # These are reference-weighted force-law gap quantities (units m^3).
    open_joint_lower: tuple[float|None,...]
    closed_joint_upper: tuple[float|None,...]
    support_lower: tuple[float|None,...]
    support_upper: tuple[float|None,...]
    visited_time_panels: int
    visited_spatial_cells: int
    detail: str = ''


def _add(a,b):
    if len(a)!=len(b):
        raise ValueError('Bernstein degrees differ')
    return tuple(x+y for x,y in zip(a,b))


def _scaled(a,factor):
    return tuple(factor*x for x in a)


def _blend(polynomials,weights):
    return tuple(sum(w*p[index] for w,p in zip(weights,polynomials))
                 for index in range(len(polynomials[0])))


def _multiply(a,b):
    values=tuple(x*y for x in a for y in b)
    return min(values),max(values)


def _plus(a,b):
    return a[0]+b[0],a[1]+b[1]


def _upper_float(value):
    result=-_lower_float(-value)
    if not np.isfinite(result):
        raise ValueError('upper bound is not finite')
    return result


@dataclass(frozen=True)
class _Face:
    ids: tuple[int,int,int]
    area: tuple[Fraction,Fraction]
    reference_squared: Fraction
    # source/ref, target/ref, D, L, H[3], target barycentric R[3][3]
    polynomials: tuple[tuple[Fraction,...],...]


_ROOT=((Fraction(1),Fraction(0),Fraction(0)),
       (Fraction(0),Fraction(1),Fraction(0)),
       (Fraction(0),Fraction(0),Fraction(1)))


def _children(vertices):
    a,b,c=vertices
    ab=tuple((x+y)/2 for x,y in zip(a,b))
    bc=tuple((x+y)/2 for x,y in zip(b,c))
    ca=tuple((x+y)/2 for x,y in zip(c,a))
    return (a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)


def _fraction(vertices):
    a,b,c=vertices
    return abs((b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1]))


def _factor(face):
    source_reference,target_reference,D,L=face.polynomials[:4]
    if any(p[0]<=0 or p[-1]<=0 for p in (source_reference,target_reference,D)):
        return None,'orientation_invalid'
    if any(min(p)<=0 for p in (source_reference,target_reference,D)):
        return None,'orientation_unresolved'
    # Cauchy bound remains positive even if L's Bernstein controls are loose.
    lower=max(min(L),min(source_reference)**2/face.reference_squared)
    upper=max(L)
    if lower<=0 or upper<lower:
        return None,'normal_unresolved'
    return (_sqrt_interval(lower)[0]/max(D),
            _sqrt_interval(upper)[1]/min(D)),None


def _cell(face,vertices,tolerance,factor):
    R=face.polynomials[7:]
    # R is grouped by target barycentric edge, then original source vertex.
    edge_values=[tuple(_blend(R[3*edge:3*edge+3],v) for v in vertices)
                 for edge in range(3)]
    outside=any(max(max(p) for p in edge)<0 for edge in edge_values)
    inside=all(min(min(p) for p in edge)>=0 for edge in edge_values)
    zero=(Fraction(0),Fraction(0))
    if outside:
        return (zero,)*3,(zero,)*3,(zero,)*3,False
    H=tuple(_blend(face.polynomials[4:7],v) for v in vertices)
    area=_multiply(face.area,(_fraction(vertices),)*2)
    supports=[];opens=[];closed=[]
    gap=_multiply((min(min(p) for p in H),max(max(p) for p in H)),factor)
    for index in range(3):
        shapes=tuple(v[index] for v in vertices)
        support=_multiply(area,(sum(shapes)/3,)*2)
        if inside:
            # Integral N_i*H = A/12[(sum N_i)(sum H)+sum N_i H].
            polynomial=_add(_scaled(_blend(H,(Fraction(1),)*3),sum(shapes)),
                            _blend(H,shapes))
            weighted=_multiply(_multiply(area,(Fraction(1,12),)*2),
                               _multiply((min(polynomial),max(polynomial)),factor))
            plus=_plus(weighted,_multiply(support,(tolerance,)*2))
            minus=_plus(weighted,_multiply(support,(-tolerance,)*2))
        else:
            # Joint expressions preserve W/support coupling at boundary cells.
            plus=(support[1]*min(Fraction(0),gap[0]+tolerance),
                  support[1]*max(Fraction(0),gap[1]+tolerance))
            minus=(support[1]*min(Fraction(0),gap[0]-tolerance),
                   support[1]*max(Fraction(0),gap[1]-tolerance))
            support=(Fraction(0),support[1])
        supports.append(support);opens.append(plus);closed.append(minus)
    return tuple(supports),tuple(opens),tuple(closed),not inside


def _aggregate(cells,count):
    supports=[[Fraction(0),Fraction(0)] for _ in range(count)]
    opens=[[Fraction(0),Fraction(0)] for _ in range(count)]
    closed=[[Fraction(0),Fraction(0)] for _ in range(count)]
    for face,vertices,depth,data in cells:
        for local,index in enumerate(face.ids):
            for total,part in zip((supports,opens,closed),data[:3]):
                total[index][0]+=part[local][0];total[index][1]+=part[local][1]
    return supports,opens,closed


def _admissible(opens,closed,phase):
    return all(closed[i][1]<=0 if active else opens[i][0]>=0
               for i,active in enumerate(phase))


class _SpatialPartition:
    """Exact incremental joint bounds with the original refinement ordering.

    Rational addition/subtraction changes neither the bounds nor their rounding.
    Monotone insertion IDs reproduce the live list's stable tie order. Only
    refinable cells need retained geometry; every cell remains in the totals.
    """
    def __init__(self,cells,count,maximum_depth):
        self.totals=_aggregate(cells,count)
        self.maximum_depth=maximum_depth
        self.heap=[];self.serial=0
        for cell in cells:self._offer(cell)

    def _offer(self,cell):
        if cell[3][3] and cell[2]<self.maximum_depth:
            heappush(self.heap,(-_fraction(cell[1]),self.serial,cell))
        self.serial+=1

    def _change(self,cell,sign):
        for local,index in enumerate(cell[0].ids):
            for total,part in zip(self.totals,cell[3][:3]):
                total[index][0]+=sign*part[local][0]
                total[index][1]+=sign*part[local][1]

    def pop(self,violations=None):
        # Refining an unrelated clipped cell cannot tighten the failed nodal
        # inequality. Keep it in both the heap and aggregate, and yield to
        # temporal subdivision when no spatial cell can improve that proof.
        deferred=[]
        while self.heap:
            entry=heappop(self.heap);cell=entry[2]
            relevant=violations is None or any(
                index in violations and
                cell[3][2 if violations[index] else 1][local][0]
                    <cell[3][2 if violations[index] else 1][local][1]
                for local,index in enumerate(cell[0].ids))
            if relevant:
                for other in deferred:heappush(self.heap,other)
                self._change(cell,-1)
                return cell
            deferred.append(entry)
        for other in deferred:heappush(self.heap,other)
        return None

    def add(self,cell):
        self._change(cell,1);self._offer(cell)


def _prepare_overlap(reference_source,reference_target,source_faces,
        a0,a1,b0,b1,av0,av1,bv0,bv1,duration,count):
    """Original-face Cramer/normal polynomials shared by phase and strong bounds."""
    data=tuple(_coordinates(value,name,n) for value,name,n in (
        (reference_source,'reference_source',count),(reference_target,'reference_target',3),
        (a0,'source start',count),(a1,'source end',count),(b0,'target start',3),(b1,'target end',3),
        (av0,'source start rate',count),(av1,'source end rate',count),
        (bv0,'target start rate',3),(bv1,'target end rate',3)))
    rs,rt,a0,a1,b0,b1,av0,av1,bv0,bv1=data
    a,b=_nodes(a0,a1,av0,av1,duration),_nodes(b0,b1,bv0,bv1,duration)
    B=_normal(b,0)
    Bref=_normal(tuple(tuple((v,) for v in point) for point in rt),0)
    target_ref=_dot(B,Bref)
    edges=(tuple(_sub(x,y) for x,y in zip(b[1],b[0])),
           tuple(_sub(x,y) for x,y in zip(b[2],b[0])))
    prepared=[]
    represented=set()
    for supplied in source_faces:
        if (len(supplied)!=3 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer))
             or not 0<=int(i)<count for i in supplied) or len(set(supplied))!=3):
            raise ValueError('source_faces require original valid triangle indices')
        ids=tuple(int(i) for i in supplied);represented.update(ids)
        nodes=tuple(a[i] for i in ids);A=_normal(nodes,0)
        Aref=_normal(tuple(tuple((v,) for v in rs[i]) for i in ids),0)
        squared=sum(p[0]**2 for p in Aref)
        if squared<=0:
            raise ValueError('reference source triangle is degenerate')
        area=tuple(v/2 for v in _sqrt_interval(squared))
        D,L=_dot(A,B),_dot(A,A)
        relative=tuple(tuple(_sub(x,y) for x,y in zip(node,b[0])) for node in nodes)
        H=tuple(_dot(point,B) for point in relative)
        R1=tuple(_dot(point,_cross(edges[1],A)) for point in relative)
        R2=tuple(_dot(point,_cross(A,edges[0])) for point in relative)
        R0=tuple(_sub(_sub(D,x),y) for x,y in zip(R1,R2))
        prepared.append(_Face(ids,area,squared,(_dot(A,Aref),target_ref,D,L,*H,*R0,*R1,*R2)))
    if represented!=set(range(count)):
        raise ValueError('every pressure node requires reference face support')
    return tuple(prepared)


class _PreparationCache:
    """One disposable exact geometry payload; no proof, phase or physical state.

    Snapshot every supplied coordinate/rate, reference, face identity and duration.
    A matching key can reuse only immutable original-face polynomials. Every proof
    still checks its own controls, orientation, support, phase and cancellation.
    """
    def __init__(self):
        self._key=self._faces=None
        self.preparations=0
        self.hits=0

    def get(self,reference_source,reference_target,source_faces,
            a0,a1,b0,b1,av0,av1,bv0,bv1,duration,count):
        snapshots=[];signature=[]
        for supplied,name,n in ((reference_source,'reference_source',count),
                (reference_target,'reference_target',3),(a0,'source start',count),
                (a1,'source end',count),(b0,'target start',3),(b1,'target end',3),
                (av0,'source start rate',count),(av1,'source end rate',count),
                (bv0,'target start rate',3),(bv1,'target end rate',3)):
            raw=np.asarray(supplied)
            if (raw.shape!=(n,3) or raw.dtype.kind not in 'fiu' or not np.all(np.isfinite(raw))
                    or any(isinstance(x,(bool,np.bool_,complex,np.complexfloating))
                           for x in np.asarray(supplied,dtype=object).flat)):
                raise ValueError(name+' requires finite real original coordinates')
            data=raw.tobytes(order='C')
            signature.append((raw.dtype.str,raw.shape,data))
            snapshots.append(np.frombuffer(data,dtype=raw.dtype).reshape(raw.shape))
        faces=[]
        for face in source_faces:
            if (len(face)!=3 or any(isinstance(i,(bool,np.bool_))
                    or not isinstance(i,(int,np.integer)) or not 0<=int(i)<count for i in face)
                    or len(set(face))!=3):
                raise ValueError('source_faces require original valid triangle indices')
            faces.append(tuple(int(i) for i in face))
        key=(count,duration,tuple(faces),tuple(signature))
        if key==self._key:
            self.hits+=1
            return self._faces
        prepared=_prepare_overlap(snapshots[0],snapshots[1],tuple(faces),
            *snapshots[2:],duration,count)
        self._key,self._faces=key,prepared
        self.preparations+=1
        return prepared


def s3_pressure_hermite_bounds(reference_source,reference_target,source_faces,
        a0,a1,b0,b1,av0,av1,bv0,bv1,dt,*,active,gap_tolerance_m,
        maximum_time_panels=128,maximum_spatial_cells=1024,maximum_depth=14,cancel=None,
        _preparation_cache=None,_polygon_bounds=False):
    """Certify one pressure phase throughout all represented overlap.

    active names every original pressure node; unsupported modes are retained.
    Positive support is never removed. Success proves W+tau*S>=0 for an open
    mode or W-tau*S<=0 for a closed mode. It does not by itself localize a change
    of phase: the caller must reequilibrate any proposed shorter endpoint.
    """
    maximum_time_panels=_limit(maximum_time_panels,'maximum_time_panels',1)
    maximum_spatial_cells=_limit(maximum_spatial_cells,'maximum_spatial_cells',1)
    maximum_depth=_limit(maximum_depth,'maximum_depth',0)
    if type(_polygon_bounds) is not bool:raise ValueError('polygon bounds selection must be boolean')
    cell_builder=_cell
    if _polygon_bounds:
        from .s3_pressure_polygon_bounds import _polygon_cell
        cell_builder=_polygon_cell
    for value,name,positive in ((dt,'dt',True),(gap_tolerance_m,'gap_tolerance_m',False)):
        raw=np.asarray(value)
        if raw.shape!=() or raw.dtype.kind not in 'fiu' or not np.isfinite(raw):
            raise ValueError(name+' must be finite and real')
        if float(value)<0 or (positive and float(value)==0):
            raise ValueError(name+' has an invalid sign')
    duration,tolerance=Fraction(float(dt)),Fraction(float(gap_tolerance_m))
    phase=np.asarray(active)
    if phase.ndim!=1 or not phase.size or phase.dtype.kind!='b':
        raise ValueError('active must name every source pressure node with booleans')
    count=len(phase)
    if _preparation_cache is not None and type(_preparation_cache) is not _PreparationCache:
        raise ValueError('invalid S3 preparation cache')
    prepare=_prepare_overlap if _preparation_cache is None else _preparation_cache.get
    prepared=prepare(reference_source,reference_target,source_faces,
        a0,a1,b0,b1,av0,av1,bv0,bv1,duration,count)
    visited_time=visited_space=0
    minima=[inf]*count;maxima=[-inf]*count
    slow=[inf]*count;shigh=[-inf]*count
    def result(success,status,detail=''):
        finite=lambda values:tuple(value if np.isfinite(value) else None for value in values)
        return S3PressureHermiteBounds(success,status,finite(minima),finite(maxima),finite(slow),finite(shigh),
            visited_time,visited_space,detail)
    pending=[(tuple(prepared),0)]
    while pending:
        if cancel is not None and cancel():return result(False,'cancelled')
        if visited_time>=maximum_time_panels:
            return result(False,'time_budget_exhausted','unvisited pressure path panels remain')
        faces,depth=pending.pop();visited_time+=1
        factors=[]
        for face in faces:
            factor,status=_factor(face)
            if status=='orientation_invalid':return result(False,status)
            if status is not None:break
            factors.append(factor)
        cells=[]
        if len(factors)==len(faces):
            # A certified opposite phase at either panel endpoint cannot be
            # repaired by further temporal subdivision of that same path.
            for endpoint in (0,-1):
                endpoints=[]
                for face in faces:
                    value=replace(face,polynomials=tuple((p[endpoint],) for p in face.polynomials))
                    factor,status=_factor(value)
                    if status is not None:return result(False,status)
                    if visited_space>=maximum_spatial_cells:return result(False,'spatial_budget_exhausted')
                    endpoints.append((value,_ROOT,0,cell_builder(value,_ROOT,tolerance,factor)));visited_space+=1
                _,point_open,point_closed=_aggregate(endpoints,count)
                if any(point_closed[i][0]>0 if active_mode else point_open[i][1]<0
                       for i,active_mode in enumerate(phase)):
                    return result(False,'pressure_phase_violation','opposite phase is certified at a panel endpoint')
            for face,factor in zip(faces,factors):
                if visited_space>=maximum_spatial_cells:return result(False,'spatial_budget_exhausted')
                cells.append((face,_ROOT,0,cell_builder(face,_ROOT,tolerance,factor)));visited_space+=1
            partition=_SpatialPartition(cells,count,maximum_depth)
            while True:
                supports,opens,closed=partition.totals
                if _admissible(opens,closed,phase):
                    for i in range(count):
                        minima[i]=min(minima[i],_lower_float(opens[i][0]))
                        maxima[i]=max(maxima[i],_upper_float(closed[i][1]))
                        slow[i]=min(slow[i],_lower_float(supports[i][0]))
                        shigh[i]=max(shigh[i],_upper_float(supports[i][1]))
                    break
                violations={
                    i:active_mode for i,active_mode in enumerate(phase)
                    if (closed[i][1]>0 if active_mode else opens[i][0]<0)}
                chosen=partition.pop(violations)
                if chosen is None:break
                face,vertices,spatial_depth,_=chosen
                factor=_factor(face)[0]
                for child in _children(vertices):
                    if cancel is not None and cancel():return result(False,'cancelled')
                    if visited_space>=maximum_spatial_cells:return result(False,'spatial_budget_exhausted')
                    partition.add((face,child,spatial_depth+1,_cell(face,child,tolerance,factor)));visited_space+=1
            if _admissible(opens,closed,phase):continue
        if depth>=maximum_depth:
            return result(False,'pressure_phase_unresolved','joint WÂ±tolerance*S does not certify the phase')
        left=[];right=[]
        for face in faces:
            pairs=tuple(_split(p) for p in face.polynomials)
            left.append(replace(face,polynomials=tuple(p[0] for p in pairs)))
            right.append(replace(face,polynomials=tuple(p[1] for p in pairs)))
        pending.append((tuple(right),depth+1));pending.append((tuple(left),depth+1))
    return result(True,'resolved')
