"""Physical paths for the developing normal-nodal, deforming S3 contact route.

No affine pressure coordinate is supplied. The full general contact contributor
owns nonlinear gap/work; this helper admits the constrained motion and maps the
actual supplied-float Hermite skins for conservative geometry checks.
"""
from dataclasses import dataclass,field,replace
from fractions import Fraction
import numpy as np
from scipy import sparse
from .fixed_projection_s3_events import _calculated, _parallel
from .s3_patch_static import GeneralS3PatchManager
from .shell_pair_hermite_sweep import shell_pair_hermite_lower
from .s3_overlap_hermite_gap import s3_overlap_hermite_gap_bounds
from .s3_pressure_hermite_bounds import _PreparationCache


def _indices(model,nodes):
    value = np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:6] for node in nodes],dtype=np.intp)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class NormalS3SkinPath:
    source_dofs: np.ndarray
    target_dofs: np.ndarray
    normal_exact: tuple[Fraction,Fraction,Fraction]
    full_size: int
    features: tuple[tuple[str,int],...]
    _preparation: _PreparationCache=field(default_factory=_PreparationCache,
        init=False,compare=False,repr=False)

    def require_motion(self,value):
        raw = np.asarray(value)
        if (raw.dtype.kind not in 'fiu' or raw.shape != (self.full_size,)
                or not np.all(np.isfinite(raw))):
            raise ValueError('normal S3 skin motion requires a finite full real vector')
        raw = _calculated(lambda:raw)
        represented = np.zeros(self.full_size,dtype=bool)
        for dofs in (self.source_dofs,self.target_dofs):
            represented[dofs.reshape(-1)] = True
            local = raw[dofs]
            if np.any(local[:,3:] != 0.):
                raise ValueError('normal S3 skin path requires zero shell rotations')
            for vector in local[:,:3]:
                exact = tuple(Fraction(float(x)) for x in vector)
                if not _parallel(exact,self.normal_exact):
                    raise ValueError('normal S3 skin path requires exact normal nodal translations')
        if np.any(raw[~represented] != 0.):
            raise ValueError('normal S3 skin path contains unrepresented motion')

    def skins(self,manager,start,end,v0,v1):
        for value in (start,end,v0,v1):
            self.require_motion(value)
        first = tuple(_calculated(lambda:value) for value in manager._positions(start))
        last = tuple(_calculated(lambda:value) for value in manager._positions(end))
        count = len(first[0])
        rates = []
        for value in (v0,v1):
            made = _calculated(lambda:manager.map@value).reshape(-1,3)
            rates.append((made[:count],made[count:]))
        return first[0],last[0],first[1],last[1],rates[0][0],rates[1][0],rates[0][1],rates[1][1]

    def sweep(self,manager,start,end,v0,v1,dt,*,cancel=None,event_gap_tolerance_m=None):
        skins = self.skins(manager,start,end,v0,v1)
        target_ids = manager.law.target_face_indices[0]
        lower = []
        for source_ids in manager.law.source_face_indices:
            args = [skin[source_ids if i in (0,1,4,5) else target_ids]
                    for i,skin in enumerate(skins)]
            proof = shell_pair_hermite_lower(*args,dt,
                maximum_panels=manager.settings.maximum_gap_triangles,
                maximum_depth=manager.settings.maximum_swept_subdivisions,
                gap_tolerance_m=manager.gap_tolerance,cancel=cancel,
                reference_source=manager.reference[0][source_ids],
                reference_target=manager.reference[1][target_ids],
                exclude_open_endpoint_approach_tolerance_m=event_gap_tolerance_m)
            if not proof.success:
                return None, 'normal S3 skin physical path unresolved: '+proof.detail
            lower.append(proof.lower_gap)
        return min(lower), None

    def pressure_sweep(self,manager,start,end,v0,v1,dt,*,active,gap_tolerance_m,cancel=None,
            polygon_bounds=False):
        from .s3_pressure_hermite_bounds import s3_pressure_hermite_bounds
        skins=self.skins(manager,start,end,v0,v1)
        ids=manager.law.target_face_indices[0]
        return s3_pressure_hermite_bounds(manager.reference[0],manager.reference[1][ids],
            manager.law.source_face_indices,*tuple(skin if i in (0,1,4,5) else skin[ids]
            for i,skin in enumerate(skins)),dt,active=active,gap_tolerance_m=gap_tolerance_m,
            maximum_time_panels=manager.settings.maximum_gap_triangles,
            maximum_spatial_cells=manager.settings.maximum_gap_triangles,
            maximum_depth=manager.settings.maximum_swept_subdivisions,cancel=cancel,
            _preparation_cache=self._preparation,_polygon_bounds=polygon_bounds)

    def overlap_sweep(self,manager,start,end,v0,v1,dt,*,cancel=None,event_gap_tolerance_m=None):
        skins=self.skins(manager,start,end,v0,v1)
        ids=manager.law.target_face_indices[0]
        proof=s3_overlap_hermite_gap_bounds(manager.reference[0],manager.reference[1][ids],
            manager.law.source_face_indices,*tuple(skin if i in (0,1,4,5) else skin[ids]
            for i,skin in enumerate(skins)),dt,gap_tolerance_m=manager.gap_tolerance,
            maximum_time_panels=manager.settings.maximum_gap_triangles,
            maximum_spatial_cells=manager.settings.maximum_gap_triangles,
            maximum_depth=manager.settings.maximum_swept_subdivisions,cancel=cancel,
            exclude_open_endpoint_approach_tolerance_m=event_gap_tolerance_m,
            _preparation_cache=self._preparation)
        if not proof.success:return None,'normal S3 matched physical path '+proof.status+': '+proof.detail
        return (float('inf') if proof.status=='no_overlap' else proof.lower_gap_m),None

    def overlap_endpoint(self,manager,full,result,*,cancel=None):
        """Driver-owned geometry result; production force/work remains unchanged."""
        self.require_motion(full)
        source,target=tuple(_calculated(lambda:value) for value in manager._positions(full))
        ids=manager.law.target_face_indices[0];target=target[ids]
        zeros_source,zeros_target=np.zeros_like(source),np.zeros_like(target)
        proof=s3_overlap_hermite_gap_bounds(manager.reference[0],manager.reference[1][ids],
            manager.law.source_face_indices,source,source,target,target,
            zeros_source,zeros_source,zeros_target,zeros_target,1.,
            gap_tolerance_m=manager.gap_tolerance,
            maximum_time_panels=manager.settings.maximum_gap_triangles,
            maximum_spatial_cells=manager.settings.maximum_gap_triangles,
            maximum_depth=manager.settings.maximum_swept_subdivisions,cancel=cancel,
            _preparation_cache=self._preparation)
        if not proof.success:return None,'normal S3 matched endpoint '+proof.status+': '+proof.detail
        lower=float('inf') if proof.status=='no_overlap' else proof.lower_gap_m
        return replace(result,minimum_gap=lower,gap_tolerance_met=lower>=-manager.gap_tolerance),None


def prepare_normal_s3_skin_path(model,manager,T):
    if (type(manager) is not GeneralS3PatchManager or manager.augmented
            or not manager.projected_work or manager.law.pressure_basis is not None
            or manager.law.geometry.differentiation != 'jet2_s3'
            or any(len(face) != 3 for face in manager.law.source_face_indices)
            or len(manager.law.target_face_indices) != 1
            or len(manager.law.target_face_indices[0]) != 3):
        raise ValueError('normal S3 skin paths require fine P1 Jet2 projected penalty and one S3 target')
    matrix = T.tocsc(copy=True) if sparse.issparse(T) else np.asarray(T)
    if sparse.issparse(matrix):
        matrix.sum_duplicates();matrix.eliminate_zeros()
    values = matrix.data if sparse.issparse(matrix) else matrix
    if (matrix.ndim != 2 or matrix.shape[0] != manager.n_dof or not matrix.shape[1]
            or values.dtype.kind not in 'fiu' or not np.all(np.isfinite(values)) or not np.any(values != 0)):
        raise ValueError('normal S3 skin paths require a finite nonempty constraint subspace')
    source,target = manager.reference
    original = source[manager.law.source_face_indices[0]]
    points = tuple(tuple(Fraction(float(x)) for x in row) for row in original)
    a = tuple(x-y for x,y in zip(points[1],points[0]))
    b = tuple(x-y for x,y in zip(points[2],points[0]))
    normal = tuple(a[i]*b[j]-a[j]*b[i] for i,j in ((1,2),(2,0),(0,1)))
    if not any(normal):
        raise ValueError('normal S3 skin reference source is degenerate')
    for body in (source,target):
        origin = tuple(Fraction(float(x)) for x in body[0])
        for point in body:
            if sum((Fraction(float(x))-y)*n for x,y,n in zip(point,origin,normal)) != 0:
                raise ValueError('normal S3 skin reference bodies require exact parallel planes')
    contract = NormalS3SkinPath(_indices(model,manager.law.source_node_ids),
        _indices(model,manager.law.target_node_ids),normal,manager.n_dof,
        tuple((manager.definition.interactions[0].name,int(node)) for node in manager.law.source_node_ids))
    for column in range(matrix.shape[1]):
        generator = matrix[:,column].toarray().reshape(-1) if sparse.issparse(matrix) else matrix[:,column]
        contract.require_motion(generator)
    return contract
