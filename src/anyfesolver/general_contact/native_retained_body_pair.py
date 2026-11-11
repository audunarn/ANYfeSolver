"""Explicit connected native bodies, complete-union certified local windows."""
from dataclasses import dataclass, replace
from fractions import Fraction
from types import SimpleNamespace
import numpy as np

from anyfesolver._ge_beam3_p5_seeded.core import sha
from .native_retained_curve_cell import NativeRetainedCurveCell
from .native_retained_station import _cross_matrix, _readonly
from .native_retained_whole_pair import NativeRetainedWholeCurvePair, NativeRetainedWholePairResult, _exact_controls
from .native_curved_pair_partition import _exact_curve, _sqrt_up, _up, _down, physical_point_map
from .native_curve_union import resolved_physical_gap_lower


def omitted_cell_dominated(source, target, endpoints, left, right, guard, shared=(False,False)):
    """Whole-domain strict KKT/convexity and squared-distance dominance.

    The omitted nearest endpoint need NOT belong to the window. Its distance
    is compared with an actual window endpoint, uniformly over source hull.
    Strict KKT excludes positive-measure endpoint activation plateaus.
    """
    hull=_exact_controls(source,left,right); controls=_exact_controls(target,0.,1.)
    square=lambda row:sum(x*x for x in row)
    _,v=_exact_curve(target,0.);_,last=_exact_curve(target,1.)
    a=tuple(y-x for x,y in zip(v,last));aa=square(a)
    t=max(Fraction(0),min(Fraction(1),-sum(x*y for x,y in zip(v,a))/aa)) if aa else Fraction(0)
    minimum=square(tuple(x+t*y for x,y in zip(v,a)))
    reach=max(square(tuple(x-y for x,y in zip(p,q))) for p in hull for q in controls)
    convexity=_down(minimum-Fraction(_sqrt_up(reach))*Fraction(_sqrt_up(aa))-Fraction(guard))
    if convexity<=0:return False
    for station,sign in ((0.,1),(1.,-1)):
        q,tangent=_exact_curve(target,station)
        phis=[sign*sum((x-y)*z for x,y,z in zip(q,p,tangent)) for p in hull]
        if min(phis)<=Fraction(guard):continue
        for endpoint in endpoints:
            j=tuple(Fraction(float(x)) for x in endpoint)
            differences=[square(tuple(x-y for x,y in zip(p,q)))-square(tuple(x-y for x,y in zip(p,j))) for p in hull]
            if min(differences)>=Fraction(guard):return True
            if shared[int(station)] and q==j:return True
    return False


@dataclass(frozen=True)
class BodyCell:
    element: int
    cell: int
    endpoints: tuple
    geometry: NativeRetainedCurveCell


@dataclass(frozen=True)
class NativeRetainedBodyResult(NativeRetainedWholePairResult):
    # Componentwise enclosures in the native mixed translation/rotation chart;
    # these are NOT a relabeling of the physical-length N and N/m error gates.
    native_residual_component_error: float = 0.0
    native_tangent_component_error: float = 0.0
    native_spatial_tangent_component_error: float = 0.0


def native_component_error_enclosure(chosen, states, context, force_error, hessian_error):
    """Propagate physical30 entry bounds through ALL native work/crossblocks.

    J=B M for fixed physical shared-end trace B. |M.T e|<=L |e|,
    |M.T E M|<=L**2|E|, L=max column1norm(M). Nonaffine lift curvature
    adds at most 2*max1norm(lift)*|e| per rotational entry; its DOFs are
    distinct per retained cell. No cancellation or condensation is used.
    """
    full=np.zeros((36,context.count))
    for i,cell in enumerate(chosen):
        full[np.ix_(np.arange(9*i,9*i+9),context.slots[cell.element])]=states[cell].geometric_jacobian
    b=physical_point_map();m=np.zeros((30,context.count))
    for column in range(30):m[column]=full[np.flatnonzero(b[:,column])[0]]
    if not np.array_equal(b@m,full):raise ValueError('native body shared trace/global work map differs')
    l=max(_up(sum(Fraction(float(x)) for x in np.abs(m[:,j]))) for j in range(context.count))
    lift=max(_up(sum(Fraction(float(x)) for x in np.abs(states[cell].rotated_midpoint_lift))) for cell in chosen)
    return np.array((_up(Fraction(l)*Fraction(float(force_error))),
        _up(Fraction(l)**2*Fraction(float(hessian_error))+2*Fraction(lift)*Fraction(float(force_error)))))


class NativeRetainedBodyPair:
    """Nearest distance law over every physical cell, with finite admission."""
    def __init__(self,context,bodies,*,radius_m=.05,line_stiffness_n_per_m2=4000.,
                 distance_tolerance_m=1e-10,energy_tolerance_j=1e-8,force_tolerance_n=1e-7,
                 tangent_tolerance_n_per_m=1e-6,maximum_search_cells=5000,
                 maximum_search_depth=20,maximum_integration_intervals=1000):
        self.context=context;self.bodies=bodies;self.radius=2*radius_m
        self.stiffness=line_stiffness_n_per_m2;self.distance_tolerance=distance_tolerance_m
        self.maximum_search_cells=maximum_search_cells;self.maximum_search_depth=maximum_search_depth
        self.maximum_integration_intervals=maximum_integration_intervals
        self.tolerances=np.array((energy_tolerance_j,force_tolerance_n,tangent_tolerance_n_per_m))
        ids={eid:i for i,(eid,_) in enumerate(context.elements)}
        if (type(bodies) is not tuple or len(bodies)!=2 or any(type(body) is not tuple or not body for body in bodies)
                or len(set(sum(bodies,())))!=len(sum(bodies,())) or set(sum(bodies,()))!=set(ids)):
            raise ValueError('native bodies require two disjoint complete element chains')
        chains=[];body_nodes=[]
        for body in bodies:
            cells=[];seen=set();previous=None
            for eid in body:
                index=ids[eid];element=context.elements[index][1];nodes=tuple(element.node_ids)
                if previous is not None and (previous!=nodes[0] or seen.intersection(nodes)!={previous}):
                    raise ValueError('native body must be an ordered connected endpoint chain')
                if previous is None and seen.intersection(nodes):raise ValueError('native body repeated nodes')
                seen.update(nodes);previous=nodes[-1]
                for cell in (0,1):cells.append(BodyCell(index,cell,(nodes[cell],nodes[cell+1]),
                    NativeRetainedCurveCell(eid,element.operator.reference,cell)))
            chains.append(tuple(cells));body_nodes.append(seen)
        if body_nodes[0]&body_nodes[1]:raise ValueError('native bodies cannot share material nodes')
        self.chains=tuple(chains);self.cells=sum(self.chains,())
        count=2*len(self.cells)
        self.kernels={}
        for source in self.cells:
            opposite=self.chains[1] if source in self.chains[0] else self.chains[0]
            target=opposite[0];se=context.elements[source.element][1];te=context.elements[target.element][1]
            self.kernels[source]=NativeRetainedWholeCurvePair(context.elements[source.element][0],se.operator.reference,
                tuple(se.node_ids),radius_m,context.elements[target.element][0],te.operator.reference,tuple(te.node_ids),radius_m,
                line_stiffness_n_per_m2=self.stiffness,distance_tolerance_m=self.distance_tolerance,
                feature_tolerance_m=1e-9,search_parameter_tolerance=1e-3,
                maximum_search_depth=maximum_search_depth,maximum_search_cells=maximum_search_cells,
                energy_tolerance_j=energy_tolerance_j/count,force_tolerance_n=force_tolerance_n/count,
                tangent_tolerance_n_per_m=tangent_tolerance_n_per_m/count,
                maximum_integration_intervals=maximum_integration_intervals,
                certify_open_intervals=True,certify_curved_ownership=True)
        self.identity=self._fingerprint()

    def _fingerprint(self):
        return sha(dict(policy='COMPLETE_NATIVE_BODY_WINDOW_PAIR_DEV_V9_CHARGED_JUNCTION_OMISSION',context=self.context.identity,
            bodies=self.bodies,chains=[[self.cells.index(cell) for cell in chain] for chain in self.chains],radius=self.radius,stiffness=self.stiffness,distance=self.distance_tolerance,
            tolerances=self.tolerances,search=(self.maximum_search_cells,self.maximum_search_depth),
            integration=self.maximum_integration_intervals,
            cells=[(cell.element,cell.cell,cell.endpoints,cell.geometry.cell,cell.geometry.element_id,cell.geometry.identity,cell.geometry.reference.canonical_data()) for cell in self.cells],
            kernels=[kernel._fingerprint() for kernel in self.kernels.values()]))

    def _states(self,mechanical):
        self.context.guard()
        if any(cell.geometry.cell!=cell.cell or cell.geometry.element_id!=self.context.elements[cell.element][0]
               for cell in self.cells):raise ValueError('native body physical cell routing changed')
        if self._fingerprint()!=self.identity:raise ValueError('native body contact definition changed')
        return {cell:cell.geometry.evaluate(mechanical.positions[self.context.nodes[cell.element]],
            mechanical.position_low[self.context.nodes[cell.element]],mechanical.cell_rotations[cell.element]) for cell in self.cells}

    def physical_gap_lower(self,mechanical):
        states=self._states(mechanical);searches=[]
        for side in (0,1):
            targets=tuple(states[cell].positions for cell in self.chains[1-side])
            for cell in self.chains[side]:
                source=states[cell].positions
                scale=max(1.,self.radius,*(float(np.max(np.abs(x))) for x in (source,*targets)))
                searches.append(SimpleNamespace(source=source,targets=targets,radius=self.radius,
                    roundoff=256*np.finfo(float).eps*scale,maximum_cells=self.maximum_search_cells,
                    maximum_depth=self.maximum_search_depth))
        return resolved_physical_gap_lower(searches,self.distance_tolerance).lower_gap_m

    def evaluate(self,mechanical,*,_diagnostics=None):
        states=self._states(mechanical);n=len(self.cells);size=9*n
        points=tuple(states[cell].positions for cell in self.cells)
        refs={cell:cell.geometry.evaluate(cell.geometry.reference.coordinates,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1))).positions for cell in self.cells}
        indices={cell:i for i,cell in enumerate(self.cells)}
        gradient=np.zeros(size);hessian=np.zeros((size,size));energy=0.;errors=np.zeros(3);counts=np.zeros(6,dtype=int);native_error=np.zeros(2)
        for side in (0,1):
            target_cells=self.chains[1-side]
            for source in self.chains[side]:
                parents=tuple(cell for cell in self.chains[side] if cell.element==source.element)
                for left,right in ((0.,.5),(.5,1.)):
                    guard=4096*np.finfo(float).eps*max(1.,*(float(np.max(np.abs(p)))**2 for p in points))
                    windows=[]
                    for first in range(len(target_cells)-1):
                        window=target_cells[first:first+2]
                        endpoints=(states[window[0]].positions[0],states[window[1]].positions[-1])
                        outer_ids=(window[0].endpoints[0],window[1].endpoints[-1])
                        if all(omitted_cell_dominated(states[source].positions,states[cell].positions,endpoints,left,right,guard,
                                tuple(node in outer_ids for node in cell.endpoints))
                               for cell in target_cells if cell not in window):windows.append(window)
                    if not windows:raise ValueError('native body complete target window is unresolved')
                    window=windows[0]
                    if window[0].endpoints[1]!=window[1].endpoints[0]:raise ValueError('native body window lacks material junction')
                    if not np.array_equal(states[window[0]].positions[-1],states[window[1]].positions[0]):
                        raise ValueError('native body junction physical traces differ')
                    chosen=parents+window;adapter=[]
                    for index,cell in enumerate(chosen):
                        state=states[cell]
                        if index>=2:
                            local=index-2;jac=np.zeros((9,42));a=6*local;b=6*(local+1)
                            jac[:3,a:a+3]=np.eye(3);jac[3:6,a:a+3]=.5*np.eye(3)
                            jac[3:6,b:b+3]=.5*np.eye(3);jac[6:9,b:b+3]=np.eye(3)
                            jac[3:6,18+3*local:21+3*local]=-_cross_matrix(state.rotated_midpoint_lift)
                            state=replace(state,geometric_jacobian=jac,cell=local)
                        adapter.append(state)
                    call_diagnostics={}
                    if _diagnostics is not None:
                        _diagnostics.setdefault('calls',[]).append(dict(source=(source.element,source.cell),
                            targets=[(cell.element,cell.cell) for cell in window],domain=(left,right),kernel=call_diagnostics))
                    value,g,h,error,counter=self.kernels[source].evaluate(None,None,None,None,None,None,
                        _states=adapter,_references=tuple(refs[cell] for cell in chosen),
                        _target_ids=tuple(cell.endpoints for cell in window),
                        _source_domains={source.cell:(left,right)},_geometry_result=True,_diagnostics=call_diagnostics)
                    increment=native_component_error_enclosure(chosen,states,self.context,error[1],error[2])
                    native_error=np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(native_error,increment)])
                    slots=np.concatenate([np.arange(9*indices[cell],9*indices[cell]+9) for cell in chosen])
                    energy+=value;np.add.at(gradient,slots,g);rows,cols=np.meshgrid(slots,slots,indexing='ij')
                    np.add.at(hessian,(rows,cols),h);errors=np.array([_up(Fraction(float(a))+Fraction(float(b))) for a,b in zip(errors,error)]);counts+=counter
                    if counts[0]>self.maximum_search_cells or counts[3]>self.maximum_integration_intervals:
                        raise MemoryError('native body aggregate finite integration/search budget exhausted')
        jac=np.zeros((size,self.context.count))
        for i,cell in enumerate(self.cells):jac[np.ix_(np.arange(9*i,9*i+9),self.context.slots[cell.element])]=states[cell].geometric_jacobian
        residual=jac.T@gradient;tangent=jac.T@hessian@jac
        for i,cell in enumerate(self.cells):
            _,curvature=states[cell].pullback(gradient[9*i:9*i+9],np.zeros((9,9)))
            slots=self.context.slots[cell.element];tangent[np.ix_(slots,slots)]+=curvature
        physical=-gradient.reshape(n,3,3);forces=[];moments=[]
        for chain in self.chains:
            ids=[indices[cell] for cell in chain]
            forces.append(np.sum(physical[ids],axis=(0,1)))
            moments.append(sum((np.sum(np.cross(points[i],physical[i]),axis=0) for i in ids),np.zeros(3)))
        if np.any(errors>self.tolerances):raise MemoryError('native body global error budget exhausted')
        if _diagnostics is not None:
            _diagnostics.update(aggregate_error=errors.copy(),native_component_error=native_error.copy(),counters=counts.copy())
        gap=self.physical_gap_lower(mechanical)
        return NativeRetainedBodyResult(energy,_readonly(residual),_readonly(tangent),
            *map(_readonly,(forces[0],forces[1],moments[0],moments[1])),gap,gap,
            *map(float,errors),*map(int,counts),*map(float,native_error),_up(Fraction(float(native_error[1]))+Fraction(float(native_error[0]))/2))
