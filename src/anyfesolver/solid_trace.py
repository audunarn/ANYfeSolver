"""Production-derived linear Q4/S3 edge traces for the research connector."""
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy import sparse

from .e4_pl_element import QualifiedE4PLShellElement
from .e4_pl_s3_v2d_element import NativeParityE4PLS3V2DShellElement
from .submodel_transfer import element_interpolation_weights
from ._coupled_certificate import CoupledSolveError, _scaled_norm, _scaled_unit

_CODE = 'COUPLED_TRACE_INVALID'
_REAL_KINDS = 'biuf'


def _trace_error(message):
    return CoupledSolveError(message, code=_CODE)


def _real_array(value, *, what, code=_CODE, shape=None, ndim=None):
    """The one real-data choke point (ported from the PR #67 port): ``np.asarray`` with no dtype, only
    dtype kinds b/i/u/f (object and complex refused, never projected), then the float64 cast, the
    shape/ndim and finite checks; anything else is a coded refusal."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _trace_error(f'{what} must be real numeric values') from exc
    if raw.dtype.kind not in _REAL_KINDS:
        raise _trace_error(f'{what} must be real numeric values (dtype kind {raw.dtype.kind!r} refused)')
    result = np.asarray(raw, dtype=float)
    if shape is not None and result.shape != tuple(shape):
        raise _trace_error(f'{what} must have shape {tuple(shape)}, got {result.shape}')
    if ndim is not None and result.ndim != ndim:
        raise _trace_error(f'{what} must be {ndim}-dimensional, got shape {result.shape}')
    if not np.isfinite(result).all():
        raise _trace_error(f'{what} must be finite')
    return result


def _real_scalar(value, *, what, code=_CODE):
    return float(_real_array(value, what=what, shape=()))


def _exact_index(value, *, what, code=_CODE):
    """An index or identifier must be an exact integer (bool and float refused, never truncated)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise _trace_error(f'{what} must be an exact integer (bool and float refused, never truncated)')
    return int(value)


def _finite_float_array(value, *, what, shape=None, ndim=None):
    """Caller/model numeric input through the one real-data choke point; anything else is SOLID_TRACE_INVALID."""
    return _real_array(value, what=what, code=_CODE, shape=shape, ndim=ndim)


def _finite_float(value, *, what):
    """A real number (bool/int/float kinds only) as a finite float; anything else is SOLID_TRACE_INVALID."""
    return _real_scalar(value, what=what, code=_CODE)


def _points(value, *, what, columns):
    """An exact (N, columns) finite real array; any other shape is SOLID_TRACE_INVALID."""
    result = _finite_float_array(value, what=what, ndim=2)
    if result.shape[1] != columns:
        raise _trace_error(f'{what} must have shape (N, {columns}), got {result.shape}')
    return result


def _element_coordinates(model, element):
    """Node coordinates of a trace element: the stored node values are checked real and finite BEFORE the
    element converts them (a complex coordinate is refused, never projected), and get_node_coordinates must
    return exactly those values with shape (nodes, 3)."""
    try:
        node_ids = tuple(element.node_ids)
        nodes = [model.mesh.nodes[nid] for nid in node_ids]
        raw = [(node.x, node.y, node.z) for node in nodes]
    except (KeyError, TypeError, AttributeError) as exc:
        raise _trace_error('shell trace element nodes are unavailable') from exc
    stored = _finite_float_array(raw, what='shell trace node coordinates', shape=(len(node_ids), 3))
    coordinates = _finite_float_array(element.get_node_coordinates(model.mesh), what='shell trace geometry',
                                      shape=(len(node_ids), 3))
    if not np.array_equal(coordinates, stored):
        raise _trace_error('shell trace geometry differs from the stored node coordinates')
    return coordinates


def _local_errstate(method):
    """Run under np.errstate(all='ignore'): a caller's 'raise' cannot surface an uncoded FloatingPointError.

    Values are unchanged (IEEE results do not depend on errstate); non-finite geometry is refused explicitly.
    """
    def wrapped(*args, **kwargs):
        with np.errstate(all='ignore'):
            return method(*args, **kwargs)
    wrapped.__name__, wrapped.__qualname__, wrapped.__doc__ = method.__name__, method.__qualname__, method.__doc__
    return wrapped


class ShellEdgeTrace:
    """Ordered straight Q4/S3 chain, constant thickness and physical director.

    Offset is the signed reference-surface to solid midsurface distance along
    the physical director. It locates the interface; transport always uses the
    actual target minus the interpolated reference coordinate, exactly once.
    Q4 edges: (0,1),(1,2),(2,3),(3,0); S3: (0,1),(1,2),(2,0).
    S3 uses its production linear MIN3 displacement field, including the
    rotation-dependent quadratic edge term. This is research-only small-linear
    kinematics, not a finite-surface or qualification declaration.
    """
    _edge_nodes = ((0,1),(1,2),(2,3),(3,0))
    _natural = ((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.))

    @classmethod
    def _layout(cls, element):
        if type(element) is QualifiedE4PLShellElement:
            return cls._edge_nodes, cls._natural
        if type(element) is NativeParityE4PLS3V2DShellElement:
            return ((0,1),(1,2),(2,0)), ((0.,0.),(1.,0.),(0.,1.))
        raise _trace_error('shell trace admits exact production E4-PL Q4 or S3 V2D only')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):
            raise AttributeError('ShellEdgeTrace is immutable; rebuild after changes')
        object.__setattr__(self,name,value)

    @_local_errstate
    def __init__(self, model, edges, *, offset=0.0, tolerance=1e-10):
        self.model = model
        try:
            pairs = tuple(tuple(pair) for pair in edges)
            _ = model.mesh.elements, model.mesh.dof_manager.total_dofs
        except (TypeError,ValueError,OverflowError,AttributeError) as exc:
            raise _trace_error('trace needs a native model and integer (element, edge) pairs') from exc
        if any(len(pair) != 2 for pair in pairs):
            raise _trace_error('trace edges must be (element, edge) pairs')
        # exact integers only: a fractional (or any float) identifier is refused, never truncated
        self.edges = tuple((_exact_index(e, what='trace element id', code=_CODE),
                            _exact_index(k, what='trace edge index', code=_CODE)) for e, k in pairs)
        self.offset=_finite_float(offset,what='trace offset')
        self.tolerance=_finite_float(tolerance,what='trace tolerance')
        tolerance=self.tolerance  # the converted value is the only one used below
        if len(set(self.edges)) != len(self.edges):
            raise _trace_error('duplicate shell trace edge')
        if not self.edges or not np.isfinite([self.offset,self.tolerance]).all() or not self.tolerance > 0:
            raise _trace_error('trace needs edges, a finite offset and positive physical tolerance')
        self.ndof = model.mesh.dof_manager.total_dofs
        vertices=[]
        normals=[]
        thicknesses=[]
        outward=[]
        self._segments=[]
        for eid, edge in self.edges:
            try:
                element=model.mesh.elements[eid]
            except (KeyError,IndexError,TypeError) as exc:
                raise _trace_error('shell trace refers to an unknown element') from exc
            edge_nodes,_=self._layout(element)
            if edge not in range(len(edge_nodes)):
                raise _trace_error('invalid production shell edge index')
            coordinates=_element_coordinates(model,element)
            a,b=edge_nodes[edge]
            start,end=coordinates[a],coordinates[b]
            # Norms and unit vectors use _scaled_norm/_scaled_unit (original formula in range, no
            # overflow/underflow loss outside it); comparisons are positive forms, so nan refuses.
            if not _scaled_norm(end-start)>tolerance:
                raise _trace_error('degenerate shell edge')
            if vertices and not _scaled_norm(start-vertices[-1])<=tolerance:
                raise _trace_error('edge chain must be ordered, connected and nonduplicated')
            if not vertices:
                vertices.append(start)
            vertices.append(end)
            geometric=np.cross(coordinates[1]-coordinates[0],coordinates[-1]-coordinates[0])
            if not np.isfinite(geometric).all():
                raise _trace_error('shell facet normal overflows: geometry outside the floating-point range')
            geometric=_scaled_unit(geometric) if _scaled_norm(geometric)>np.finfo(float).tiny else None
            if geometric is None:
                raise _trace_error('degenerate shell facet')
            director=element.physical_reference_director
            director=geometric if director is None else _finite_float_array(director,what='physical shell director')
            if director.shape!=(3,) or not np.isfinite(director).all() or not _scaled_norm(director)>np.finfo(float).tiny:
                raise _trace_error('physical shell director must be finite and nonzero')
            director=_scaled_unit(director)
            if director is None:
                raise _trace_error('physical shell director must be finite and nonzero')
            if not (abs(abs(geometric@director)-1)<=1e-10 and np.max(abs((coordinates-start)@director))<=tolerance):
                raise _trace_error('only planar facets with physical normal are supported')
            normals.append(director)
            thickness=_finite_float(element.thickness,what='each shell thickness')
            if thickness<=0:
                raise _trace_error('each shell thickness must be finite and positive')
            _finite_float(getattr(element,'reference_surface_offset',0.0),what='shell reference-surface offset')
            thicknesses.append(thickness)
            inward=coordinates.mean(axis=0)-(start+end)/2
            inward-=director*(director@inward)
            tangent=_scaled_unit(end-start)
            if tangent is None:
                raise _trace_error('degenerate shell edge')
            inward-=tangent*(tangent@inward)
            # Codex round-6 P2: a finite inward vector whose norm overflowed divided to a zero outward.
            unit_inward=_scaled_unit(inward) if _scaled_norm(inward)>tolerance else None
            if unit_inward is None:
                raise _trace_error('degenerate shell facet side')
            outward.append(unit_inward)
            self._segments.append((element,edge,
                np.frombuffer(np.ascontiguousarray(start).tobytes(),dtype=float),
                np.frombuffer(np.ascontiguousarray(end).tobytes(),dtype=float)))
        self.reference_start=np.array(vertices[0],float)
        delta=np.array(vertices[-1])-self.reference_start
        self.length=_scaled_norm(delta)
        tangent=_scaled_unit(delta)
        if not self.length>tolerance or tangent is None:
            raise _trace_error('zero-length interface')
        self.tangent=tangent
        self.normal=np.array(normals[0])
        self.conormal=np.cross(self.normal,self.tangent)
        self.solid_outward=np.array(outward[0])
        self.thickness=thicknesses[0]
        if self.thickness<=0 or not np.isfinite(self.thickness):
            raise _trace_error('thickness must be finite and positive')
        coordinates=np.array(vertices)
        self.breakpoints=(coordinates-self.reference_start)@self.tangent
        residual=coordinates-self.reference_start-self.breakpoints[:,None]*self.tangent
        if not (all(_scaled_norm(r)<=tolerance for r in residual) and np.all(np.diff(self.breakpoints)>tolerance)):
            raise _trace_error('only monotone straight edge chains are supported')
        if not (all(np.linalg.norm(n-self.normal)<=1e-10 for n in normals) and all(abs(t-self.thickness)<=tolerance for t in thicknesses) and all(np.linalg.norm(n-self.solid_outward)<=1e-10 for n in outward)):
            raise _trace_error('mixed directors, sides or thicknesses require separate qualification')
        if not all(np.isfinite(np.asarray(v,float)).all() for v in (self.reference_start,self.length,self.tangent,
                   self.normal,self.conormal,self.solid_outward,self.breakpoints,*normals,*outward)):
            raise _trace_error('shell trace geometry is outside the floating-point range')
        for name in ('reference_start','tangent','normal','conormal','solid_outward','breakpoints'):
            value=np.ascontiguousarray(getattr(self,name))
            setattr(self,name,np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape))
        self._segments=tuple(self._segments)
        self._signature=self.signature()
        self._sealed=True

    def signature(self):
        payload=[]
        for eid,edge in self.edges:
            e=self.model.mesh.elements[eid]
            payload.append((eid,edge,id(e),type(e).__module__,type(e).__qualname__,e.node_ids,
                            _element_coordinates(self.model,e).tolist(),_finite_float(e.thickness,what='shell thickness'),
                            None if e.physical_reference_director is None else
                            _finite_float_array(e.physical_reference_director,what='physical shell director').tolist(),
                            _finite_float(getattr(e,'reference_surface_offset',0.0),what='shell reference-surface offset'),
                            [list(self.model.mesh.dof_manager.get_node_dofs(node)) for node in e.node_ids]))
        return hashlib.sha256(json.dumps((payload,self.offset,self.tolerance,self.model.mesh.dof_manager.total_dofs),sort_keys=True).encode()).hexdigest()

    def validate(self):
        try:
            current=self.signature()
        except (KeyError,ValueError,TypeError,AttributeError) as exc:
            raise _trace_error('stale shell trace: referenced geometry or layout is unavailable') from exc
        if current!=self._signature:
            raise _trace_error('stale shell trace: rebuild after geometry/thickness/director/DOF changes')

    @_local_errstate
    def chart(self,points):
        """(s,t) chart coordinates, shape (N,2), of finite real points of shape exactly (N,3); any other input
        shape, a non-finite input or result refuses (SOLID_TRACE_INVALID)."""
        points=_points(points,what='trace points',columns=3)
        try:
            relative=points-self.reference_start
            result=np.column_stack((relative@self.tangent,relative@self.normal-self.offset))
        except (ValueError,TypeError,IndexError,OverflowError) as exc:
            raise _trace_error('trace points must be finite numeric coordinates') from exc
        if not np.isfinite(result).all():
            raise _trace_error('trace chart is not finite (non-finite points or overflow)')
        return result

    @_local_errstate
    def points(self,chart):
        """Points, shape (N,3), of finite real (s,t) chart coordinates of shape exactly (N,2); any other input
        shape, a non-finite input or result refuses (SOLID_TRACE_INVALID)."""
        chart=_points(chart,what='chart coordinates',columns=2)
        try:
            result=self.reference_start+chart[:,0,None]*self.tangent+(chart[:,1,None]+self.offset)*self.normal
        except (ValueError,TypeError,IndexError,OverflowError) as exc:
            raise _trace_error('chart coordinates must be a finite (N,2) numeric array') from exc
        if not np.isfinite(result).all():
            raise _trace_error('trace points are not finite (non-finite chart or overflow)')
        return result

    @_local_errstate
    def displacement_operator(self,points):
        self.validate()
        points=_points(points,what='trace points',columns=3)
        chart=self.chart(points)
        plane=(points-self.reference_start)@self.solid_outward
        # Positive forms: a nan chart or plane coordinate (overflowed relative position) refuses.
        if not (np.all(abs(plane)<=self.tolerance) and np.all(chart[:,0]>=-self.tolerance) and np.all(chart[:,0]<=self.length+self.tolerance) and np.all(abs(chart[:,1])<=self.thickness/2+self.tolerance)):
            raise _trace_error('points are outside the physical shell-solid interface')
        rows=[]; cols=[]; data=[]
        for p,(s,_) in enumerate(chart):
            segment=min(np.searchsorted(self.breakpoints,s,side='right')-1,len(self._segments)-1)
            segment=max(segment,0)
            element,edge,start,end=self._segments[segment]
            alpha=(s-self.breakpoints[segment])/(self.breakpoints[segment+1]-self.breakpoints[segment])
            edge_nodes,natural_nodes=self._layout(element)
            a,b=edge_nodes[edge]
            natural=(1-alpha)*np.array(natural_nodes[a])+alpha*np.array(natural_nodes[b])
            weights=element_interpolation_weights(element,natural)
            reference=weights@_element_coordinates(self.model,element)
            arm=points[p]-reference
            for local_node,(node,weight) in enumerate(zip(element.node_ids,weights)):
                dofs=self.model.mesh.dof_manager.get_node_dofs(node)
                for component in range(3):
                    rows.append(3*p+component); cols.append(dofs[component]); data.append(weight)
                for axis in range(3):
                    vector=weight*np.cross(np.eye(3)[axis],arm)
                    if type(element) is NativeParityE4PLS3V2DShellElement:
                        # Restriction of production w_h=sum(N*w-l*rx+m*ry).
                        # l/m vanish at vertices; retained rotations are linear.
                        sign=1. if local_node==a else -1. if local_node==b else 0.
                        vector=vector+sign*.5*alpha*(1-alpha)*self.normal*np.dot(
                            self.normal,np.cross(np.eye(3)[axis],end-start))
                    for component in range(3):
                        rows.append(3*p+component); cols.append(dofs[3+axis]); data.append(vector[component])
        if not np.isfinite(data).all():
            raise _trace_error('trace displacement operator is not finite')
        return sparse.csr_matrix((data,(rows,cols)),shape=(3*len(points),self.ndof))
