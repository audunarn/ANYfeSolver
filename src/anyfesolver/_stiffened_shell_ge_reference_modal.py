"""Experimental supported virgin Q4/native-GE point descriptor modes.

The full mixed pencil is retained. Physical half-cell rotations have inertia;
nodal beam rotation traces, eighteen resultants/member, Q4 normal spin and point
multipliers do not. No static/Guyan reduction, tangent averaging or mass floor.
Requested pairs are nearest the declared shift, not a complete lowest spectrum.
"""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, ArpackError

from .control import CancellationToken, SolveCancelled, cancellation_safe_point
from .linalg import MatrixClass, factorize_cached
from ._native_reference_modal import _owned
from ._ge_beam3_p5_centered.mass import reference_kinetic_factors
from ._stiffened_shell_physical_q4_mass import physical_mass
from ._stiffened_shell_physical_s3_mass import physical_mass as s3_physical_mass
from ._stiffened_shell_ge_mixed_modal_mass import native_physical_mass
from ._stiffened_shell_b2_q4_panel_experimental import _modal_scaled_residual

POLICY = "GE_Q4_POINT_SUPPORTED_VIRGIN_FULL_DESCRIPTOR_REFERENCE_MODES_V1_EXPERIMENTAL"
MASS_POLICY = "Q4_DIRECTOR_AND_CENTERED_NATIVE_FULL_CELL_INERTIA_24_POINT_V1"
GE_FREE_FREE_MODAL_ONLY_V1 = "GE_CONNECTED_Q4_POINT_FREE_FREE_MODAL_ONLY_V1_EXPERIMENTAL"
MIXED_POLICY = "GE_FLAT_Q4_S3_NATIVE_VIRGIN_FULL_DESCRIPTOR_REFERENCE_MODES_V1_EXPERIMENTAL"
MIXED_MASS_POLICY = "Q4_DIRECTOR_S3_LINEAR_NODAL_DIRECTOR_CENTERED_NATIVE_FULL_CELL_INERTIA_24_POINT_V1"
GE_MIXED_FREE_FREE_MODAL_ONLY_V1 = "GE_FLAT_Q4_S3_POINT_FREE_FREE_MODAL_ONLY_V1_EXPERIMENTAL"


@dataclass(frozen=True)
class GEReferenceModesRequest:
    count: int = 3
    shift_hz: float = 1.
    max_iterations: int = 600

    def __post_init__(self):
        if (type(self.count) is not int or self.count <= 0 or
                type(self.max_iterations) is not int or self.max_iterations <= 0 or
                type(self.shift_hz) not in (int, float) or
                not np.isfinite(self.shift_hz) or self.shift_hz <= 0.):
            raise ValueError("positive finite GE reference-mode controls required")


@dataclass(frozen=True)
class GEReferenceModesResult:
    status: str
    model_identity: str
    mass_identity: str = ""
    frequencies_hz: tuple[float, ...] = ()
    eigenvalues_rad2_s2: tuple[float, ...] = ()
    full_coordinates: tuple[np.ndarray, ...] = ()
    shell_modes: tuple[np.ndarray, ...] = ()
    beam_modes: tuple[np.ndarray, ...] = ()
    multiplier_modes: tuple[np.ndarray, ...] = ()
    support_reactions: tuple[np.ndarray, ...] = ()
    equation_residuals: tuple[float, ...] = ()
    constraint_residuals: tuple[float, ...] = ()
    physical_equation_residuals: tuple[float, ...] = ()
    zero_inertia_dofs: tuple[int, ...] = ()
    inverse_eigenvalues: tuple[complex, ...] = ()
    factorization_count: int = 0
    solve_count: int = 0
    shift_hz: float = 0.
    failure_reason: str | None = None
    policy: str = POLICY
    mass_policy: str = MASS_POLICY
    selection: str = "requested finite pairs nearest shift; completeness unqualified"
    production_qualified: bool = False


@dataclass(frozen=True)
class GEMixedReferenceModesRequest:
    """Explicit flat Q4/S3, native generalized/fibre, POINT/BL5 extension."""
    count: int = 3
    shift_hz: float = 1.
    max_iterations: int = 600

    def __post_init__(self):
        GEReferenceModesRequest.__post_init__(self)


@dataclass(frozen=True)
class GEMixedReferenceModesResult(GEReferenceModesResult):
    policy: str = MIXED_POLICY
    mass_policy: str = MIXED_MASS_POLICY


@dataclass(frozen=True)
class GEFreeReferenceModesResult:
    """Separate zero-mode space and flexible requested pairs; no legacy schema change."""
    status: str
    model_identity: str
    flexible: GEReferenceModesResult
    rigid_origin_m: tuple[float, float, float] = ()
    canonical_rigid_coordinates: tuple[np.ndarray, ...] = ()
    rigid_coordinates: tuple[np.ndarray, ...] = ()
    rigid_multipliers: tuple[np.ndarray, ...] = ()
    rigid_kinetic_gram: np.ndarray | None = None
    rigid_equation_residuals: tuple[float, ...] = ()
    rigid_constraint_residuals: tuple[float, ...] = ()
    rigid_flexible_mass_overlap: float = 0.
    spectral_shift_rad2_s2: float = 0.
    failure_reason: str | None = None
    policy: str = GE_FREE_FREE_MODAL_ONLY_V1
    selection: str = "six verified canonical rigid fields and flexible pairs nearest negative shift; completeness unqualified"
    production_qualified: bool = False


class _ModalFailure(RuntimeError):
    pass


def _admit(owner, *, mixed=False):
    owner._require_open()
    if owner._definition.distributing_footprints or owner._definition.finite_distributing_footprints:
        raise ValueError('spectral operations exclude reference-linear distributing footprints')
    if getattr(owner,"_curved_reference",False):
        raise ValueError("curved reference spectra are not admitted")
    definition = owner._definition
    if not mixed and definition.modal_boundary_policy == GE_MIXED_FREE_FREE_MODAL_ONLY_V1:
        raise ValueError("mixed free-free boundary requires GEMixedReferenceModesRequest")
    if (owner._accepted is not None or owner._finite_driver is not None or
            owner._generation != 0):
        raise ValueError("GE reference modes require a fresh virgin owner without accepted histories")
    if not mixed and (definition.lines or definition.footprints or definition.shell.s3_elements or
            definition.shell.j2_curve is not None or owner._has_physical_fibre or
            not definition.points or (not definition.fixed and not owner._modal_only)):
        raise ValueError("virgin elastic Q4/generalized-GE six-pose points required; lines, footprints, S3 and materials unsupported")
    if mixed:
        from ._stiffened_shell_l1_assembly_experimental import ExperimentalBL5Line
        from .e4_pl_s3_v2d_element import FORMULATION_ID
        if (definition.footprints or definition.shell.j2_curve is not None or
                not (definition.points or definition.lines) or
                (not definition.fixed and not owner._modal_only) or
                any(type(line) is not ExperimentalBL5Line for line in definition.lines) or
                owner._bl5_material or owner._bl5_shell_material or owner._bl5_native_material):
            raise ValueError("mixed reference modes require virgin elastic flat POINT/BL5; material BL5, shell J2 and footprints unsupported")
        if any(len(slots) == 18 and getattr(element, "formulation_id", None) != FORMULATION_ID
               for _, element, slots, _ in owner._shell_parts):
            raise ValueError("S3 linear nodal director mass requires actual direct18 V2D owner; bubble families unsupported")
    if owner._unsupported_modes:
        raise ValueError(owner._unsupported_modes[0])
    # The shell's compile seam already requires one connected cell graph.
    # Native components use declared node identity, never coordinate merging.
    neighbours = {node: set() for node in owner._beam_ids}
    for element, _, _ in owner._beam_parts:
        for node in element.node_ids:
            neighbours[node].update(element.node_ids)
    linked = {point.beam_node for point in definition.points}
    if mixed:
        for line in definition.lines:
            for member, _ in line.beam_halfcells:
                linked.update(owner._beam_parts[member][0].node_ids)
    unseen = set(neighbours)
    while unseen:
        start = min(unseen)
        component = {start}; pending = [start]
        while pending:
            for other in neighbours[pending.pop()]-component:
                component.add(other); pending.append(other)
        if not component.intersection(linked):
            raise ValueError("connected reference-modal panel requires every native beam component linked by a six-pose point to the shell")
        unseen.difference_update(component)
    # Bounded six-column physical rigid-field detectability, not a global
    # stiffness/kernel test. Reject partial/free supports in this first scope.
    rigid = np.zeros((owner._K.shape[0], 6))
    for offset, xyz in ((0, owner._shell_xyz),
                        (owner._shell_count, np.array([owner._beam_nodes[i][0]
                                                     for i in owner._beam_ids]))):
        for i, position in enumerate(xyz):
            row = offset+6*i
            rigid[row:row+3, :3] = np.eye(3)
            for j in range(3):
                rigid[row:row+3, 3+j] = np.cross(np.eye(3)[j], position)
            rigid[row+3:row+6, 3:] = np.eye(3)
    if not owner._modal_only:
        supported = rigid[owner._fixed]
        scales = np.linalg.norm(supported, axis=0)
        if np.any(scales == 0.):
            raise ValueError("all six physical rigid fields require supported detectability")
        singular = np.linalg.svd(supported/scales, compute_uv=False)
        if len(singular) < 6 or singular[-1] <= 64*np.finfo(float).eps*max(supported.shape)*singular[0]:
            raise ValueError("all six physical rigid fields require supported detectability")
    for inertia in owner._reference_modal_inertias:
        _physical_spatial_inertia(inertia)


def _physical_spatial_inertia(matrix):
    """New-operation physical spatial admission; never project legacy input.

    Relative 128-epsilon budgets cover the bounded 3x3 arithmetic and eigen
    evaluation. This is binary64 spatial realizability, not a planar-section
    geometry certificate. The captured native tensor is used without changes.
    """
    a = np.asarray(matrix)
    if a.shape != (6, 6) or not np.isfinite(a).all() or not np.array_equal(a, a.T):
        raise ValueError("explicit symmetric physical spatial inertia required")
    eps = 128*np.finfo(float).eps
    with np.errstate(over="ignore", invalid="ignore"):
        m = float(np.trace(a[:3, :3])/3.)
    if not np.isfinite(m):
        raise ValueError("physical spatial inertia scalar mass is not representable")
    translation_scale = max(float(np.max(abs(a[:3, :3]))), abs(m))
    if m <= 0. or np.max(abs(a[:3, :3]-m*np.eye(3))) > eps*translation_scale:
        raise ValueError("physical spatial inertia translation must equal positive mass times identity")
    cross = a[:3, 3:]
    cross_scale = float(np.max(abs(cross), initial=0.))
    if np.max(abs(cross+cross.T), initial=0.) > eps*cross_scale:
        raise ValueError("physical spatial inertia requires skew first-moment coupling")
    # cross=-[h]x. Do not symmetrize/project the supplied cross block.
    h = np.array((cross[1, 2], cross[2, 0], cross[0, 1]))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        c = h/m
    if not np.isfinite(c).all():
        raise ValueError("physical spatial inertia centroid is not representable")
    with np.errstate(over="ignore", invalid="ignore"):
        parallel = m*((c@c)*np.eye(3)-np.outer(c, c))
    if not np.isfinite(parallel).all():
        raise ValueError("physical spatial inertia parallel-axis term is not representable")
    with np.errstate(over="ignore", invalid="ignore"):
        central = a[3:, 3:]-parallel
        second = .5*np.trace(central)*np.eye(3)-central
    if not np.isfinite(central).all() or not np.isfinite(second).all():
        raise ValueError("physical spatial inertia central second moment is not representable")
    scale = max(float(np.linalg.norm(a[3:, 3:], 2)),
                float(np.linalg.norm(parallel, 2)))
    budget = eps*scale
    eigenvalues = np.linalg.eigvalsh(second)
    if not np.isfinite(budget) or not np.isfinite(eigenvalues).all():
        raise ValueError("physical spatial inertia realizability evaluation is not representable")
    if eigenvalues[0] < -budget:
        raise ValueError("physical spatial inertia violates central second-moment realizability")
    return m, c, central


def capabilities(owner):
    """Separate explicit route descriptor; legacy capability schema is unchanged."""
    reason = None
    try:
        _admit(owner)
    except (ValueError, RuntimeError) as exc:
        reason = str(exc)
    free = owner._modal_only
    return dict(policy=GE_FREE_FREE_MODAL_ONLY_V1 if free else POLICY, mass_policy=MASS_POLICY,
        status="EXPERIMENTAL_NOT_QUALIFIED", operation="modes", scope=("connected free-free virgin Q4/GE point panel modal-only" if free else "connected supported virgin Q4/GE point panel reference only"),
        eligible=reason is None, ineligibility_reason=reason,
        mass_source="explicit native definition inertia; explicit shell density and thickness",
        mass_admission="binary64 physical 3D spatial inertia; planar section realizability unqualified",
        inertia="nodal translations and six native half-cell rotations; tangential shell directors",
        algebraic="native nodal rotation traces, eighteen resultants/member, Q4 normal spin and multipliers",
        selection=("six canonical rigid fields and requested flexible pairs nearest negative shift; completeness unqualified" if free else "requested finite pairs nearest shift; completeness unqualified"),
        excluded=(("static solve", "finite solve", "restore", "chart state writes") if free else ("free-free",))+("prestress", "buckling", "finite dynamics", "accepted material histories", "S3", "line", "footprint"),
        production_qualified=False)


def mixed_capabilities(owner):
    reason = None
    try:
        _admit(owner, mixed=True)
    except (ValueError, RuntimeError) as exc:
        reason = str(exc)
    return dict(policy=GE_MIXED_FREE_FREE_MODAL_ONLY_V1 if owner._modal_only else MIXED_POLICY,
        mass_policy=MIXED_MASS_POLICY,
        status="EXPERIMENTAL_NOT_QUALIFIED", operation="modes",
        scope="virgin flat Q4/S3/mixed shell; generalized and axial/biaxial physical fibre; POINT/BL5",
        eligible=reason is None, ineligibility_reason=reason,
        boundary="six canonical rigid fields" if owner._modal_only else "supported",
        mass_source="explicit native 6x6 definition inertia; shell density and thickness",
        algebraic="shell normal spin, native nodal rotation traces, retained18 and multipliers",
        selection="requested finite pairs nearest shift; completeness unqualified",
        excluded=("accepted histories", "prestress", "active plasticity", "shell J2",
                  "material BL5", "chart state", "curved shell", "footprints"),
        production_qualified=False)


def _assemble_mass(owner, check=lambda stage: None, *, mixed=False):
    """Reuse physical local kinetic laws and the owner's authoritative slots."""
    cache_name = "_mixed_reference_modal_mass" if mixed else "_reference_modal_mass"
    cached = getattr(owner, cache_name, None)
    if cached is not None:
        check("cached_mass")
        return cached
    rows = []; columns = []; values = []
    zero = {6*i+5 for i in range(len(owner._shell_ids))}
    offset = owner._shell_count
    zero.update(offset+6*i+j for i in range(len(owner._beam_ids)) for j in (3, 4, 5))
    def scatter(slots, matrix):
        rows.extend(np.repeat(slots, len(slots)))
        columns.extend(np.tile(slots, len(slots)))
        values.extend(matrix.ravel())
    panel = owner._definition.shell
    for _, _, slots, _ in owner._shell_parts:
        check("shell_mass")
        vertices = owner._shell_xyz[np.asarray(slots)[::6]//6]
        law = s3_physical_mass if mixed and len(slots) == 18 else physical_mass
        scatter(slots, law(vertices, density_kg_m3=panel.density_kg_m3,
                                    thickness_m=panel.thickness_m))
    for inertia, (element, slots, _) in zip(owner._reference_modal_inertias, owner._beam_parts):
        check("beam_mass")
        _physical_spatial_inertia(inertia)
        if mixed:
            local = native_physical_mass(element.operator.reference, inertia)
        else:
            factors = reference_kinetic_factors(element.operator.reference,
                element.section.elastic, inertia, order=24)
            local = factors.full.T@factors.full
        if not np.isfinite(local).all():
            raise ValueError("finite explicit native reference inertia required")
        # slots are beam-local global incidences; the first24 retain nodal18
        # and inertial cell6. The following eighteen resultants stay massless.
        scatter(offset+slots[:24], local)
        zero.update(int(offset+slot) for slot in slots[24:])
    mass = sparse.coo_matrix((values, (rows, columns)), shape=owner._K.shape).tocsr()
    mass.sum_duplicates(); mass.eliminate_zeros(); mass.sort_indices()
    zero = tuple(sorted(zero))
    if (not np.isfinite(mass.data).all() or mass[:, zero].nnz or mass[zero, :].nnz):
        raise ValueError("declared reference algebraic inertia must be exactly absent")
    if np.any(mass.diagonal()[[i for i in range(mass.shape[0]) if i not in zero]] <= 0.):
        raise ValueError("positive physical reference coordinate inertia required")
    digest = sha256()
    digest.update(owner._identity.encode("ascii"))
    digest.update((MIXED_MASS_POLICY if mixed else MASS_POLICY).encode("ascii"))
    for array in (mass.indptr, mass.indices, mass.data): digest.update(array.tobytes())
    # Immutable byte-backed arrays; publication only after complete preparation.
    for name in ("data", "indices", "indptr"):
        array = getattr(mass, name)
        setattr(mass, name, np.frombuffer(array.tobytes(), dtype=array.dtype))
    check("mass_publish")
    result = (mass, zero, digest.hexdigest())
    setattr(owner, cache_name, result)
    return result


def _congruence(owner, A, B):
    # Reuse the established scale calculation with its correct native packing,
    # exclusively for this new route. No legacy solve or stored scale changes.
    from ._stiffened_shell_ge_experimental import _reference_scales
    _, shell, beam = _reference_scales(owner._shell_xyz, owner._beam_nodes,
        owner._beam_parts, owner._shell_count, owner._beam_count,
        footprint_retained_roles=True)
    primal = np.r_[shell, beam][owner._free]
    Dp = sparse.diags(primal)
    weighted = (Dp@owner._K[owner._free, :][:, owner._free]@Dp).tocsr()
    work = float(np.max(np.abs(weighted.data), initial=0.))
    if not np.isfinite(work) or work <= 0.:
        raise _ModalFailure("positive representable reference work scale required")
    scale = np.r_[primal/np.sqrt(work), np.sqrt(work)/owner._constraint_scale]
    if not np.isfinite(scale).all() or np.any(scale <= 0.):
        raise _ModalFailure("reference modal congruence is not representable")
    D = sparse.diags(scale)
    As, Bs = (D@A@D).tocsc(), (D@B@D).tocsc()
    for original, scaled in ((A, As), (B, Bs)):
        old = original.copy(); old.eliminate_zeros(); old.sort_indices()
        scaled.eliminate_zeros(); scaled.sort_indices()
        if (not np.isfinite(scaled.data).all() or old.nnz != scaled.nnz or
                not np.array_equal(old.indptr, scaled.indptr) or
                not np.array_equal(old.indices, scaled.indices)):
            raise _ModalFailure("modal matrix entries lost through congruence")
        denominator = float(np.max(np.abs(scaled.data), initial=0.))
        delta = scaled-scaled.T
        if denominator == 0. or np.max(np.abs(delta.data), initial=0.) > 1e-10*denominator:
            raise _ModalFailure("conservative reference pencil symmetry unresolved; no averaging permitted")
    return As, Bs, scale


def _free_rigid_space(owner, mass, scale, Bs, check):
    """Six physical common-body velocities; only a bounded 6x6 Gram solve."""
    origin = owner._shell_xyz.mean(axis=0)
    R = np.zeros((owner._K.shape[0], 6))
    for offset, xyz in ((0, owner._shell_xyz),
                        (owner._shell_count, np.array([owner._beam_nodes[i][0]
                                                     for i in owner._beam_ids]))):
        for i, position in enumerate(xyz):
            row = offset+6*i
            R[row:row+3, :3] = np.eye(3)
            for j in range(3):
                R[row:row+3, 3+j] = np.cross(np.eye(3)[j], position-origin)
            R[row+3:row+6, 3:] = np.eye(3)
    for _, slots, _ in owner._beam_parts:
        R[owner._shell_count+slots[18:21], 3:] = np.eye(3)
        R[owner._shell_count+slots[21:24], 3:] = np.eye(3)
    n = len(owner._free)
    equations = []; gaps = []
    for column in R.T:
        check("rigid_field")
        equation = _modal_scaled_residual(owner._K@column,
            abs(owner._K)@abs(column), scale[:n])
        gap = _modal_scaled_residual(owner._C@column,
            abs(owner._C)@abs(column), scale[n:])
        if equation > 1e-8 or gap > 1e-9:
            raise _ModalFailure("canonical free-free rigid field violates physical K or point C")
        equations.append(equation); gaps.append(gap)
    gram = R.T@mass@R
    if (not np.isfinite(gram).all() or
            np.max(abs(gram-gram.T)) > 1e-10*np.max(abs(gram))):
        raise _ModalFailure("canonical rigid kinetic Gram symmetry unresolved")
    try:
        lower = np.linalg.cholesky(gram)
        W = R@np.linalg.solve(lower.T, np.eye(6))
    except np.linalg.LinAlgError as exc:
        raise _ModalFailure("six physical rigid fields require positive kinetic Gram") from exc
    if (not np.isfinite(W).all() or
            np.max(abs(W.T@mass@W-np.eye(6))) > 1e-7):
        raise _ModalFailure("rigid kinetic normalization unresolved")
    Ws = np.vstack((W, np.zeros((owner._C.shape[0], 6))))/scale[:, None]
    if not np.isfinite(Ws).all() or np.max(abs(Ws.T@Bs@Ws-np.eye(6))) > 1e-7:
        raise _ModalFailure("rigid congruence reconstruction unresolved")
    return dict(origin=tuple(map(float, origin)), canonical=R, normalized=W,
                scaled=Ws, gram=gram, equations=tuple(equations), gaps=tuple(gaps))


def solve(owner, request, *, cancellation_token=None):
    mixed = type(request) is GEMixedReferenceModesRequest
    _admit(owner, mixed=mixed)
    if type(request) not in (GEReferenceModesRequest, GEMixedReferenceModesRequest):
        raise ValueError("typed GE reference-mode request required")
    if cancellation_token is not None and not isinstance(cancellation_token, CancellationToken):
        raise ValueError("CancellationToken required")
    n = len(owner._free); r = owner._C.shape[0]
    if request.count >= n+r-1:
        raise ValueError("reference mode count must leave sparse Arnoldi room")
    sigma = float((2*np.pi*request.shift_hz)**2)
    if not np.isfinite(sigma) or sigma <= 0.:
        raise ValueError("positive representable squared reference frequency required")
    free = owner._modal_only
    if free: sigma = -sigma
    handle = None; raw = (); mass_id = ""; zero = ()
    rigid = None; overlap = 0.
    def check(stage):
        cancellation_safe_point(cancellation_token, "ge_reference_modes."+stage)
    def finish(status, **kwargs):
        if free:
            kwargs.update(policy=GE_FREE_FREE_MODAL_ONLY_V1,
                          selection="flexible finite pairs nearest negative shift; completeness unqualified")
        if mixed:
            kwargs["policy"] = GE_MIXED_FREE_FREE_MODAL_ONLY_V1 if free else MIXED_POLICY
        result_type = GEMixedReferenceModesResult if mixed else GEReferenceModesResult
        flexible = result_type(status, owner._identity, mass_id,
            zero_inertia_dofs=zero, inverse_eigenvalues=raw,
            factorization_count=handle.factorization_count if handle else 0,
            solve_count=handle.solve_count if handle else 0,
            shift_hz=float(request.shift_hz), **kwargs)
        if not free: return flexible
        data = {}
        if status == "completed":
            data = dict(rigid_origin_m=rigid["origin"],
                canonical_rigid_coordinates=tuple(_owned(x) for x in rigid["canonical"].T),
                rigid_coordinates=tuple(_owned(x) for x in rigid["normalized"].T),
                rigid_multipliers=tuple(_owned(np.zeros(r)) for _ in range(6)),
                rigid_kinetic_gram=_owned(rigid["gram"]),
                rigid_equation_residuals=rigid["equations"],
                rigid_constraint_residuals=rigid["gaps"],
                rigid_flexible_mass_overlap=overlap)
        if mixed: data["policy"] = GE_MIXED_FREE_FREE_MODAL_ONLY_V1
        return GEFreeReferenceModesResult(status, owner._identity, flexible,
            spectral_shift_rad2_s2=sigma, failure_reason=flexible.failure_reason, **data)
    try:
        check("start")
        mass, zero, mass_id = _assemble_mass(owner, check, mixed=mixed)
        B = sparse.block_diag((mass[owner._free, :][:, owner._free],
                              sparse.csc_matrix((r, r))), format="csc")
        A = owner._A.copy()
        As, Bs, scale = _congruence(owner, A, B)
        if free:
            rigid = _free_rigid_space(owner, mass, scale, Bs, check)
        def project(vector):
            if not free: return vector
            return vector-rigid["scaled"]@(rigid["scaled"].T@(Bs@vector))
        check("before_factorization")
        handle = factorize_cached((As-sigma*Bs).tocsc(), MatrixClass.GENERAL)
        if handle.status != "ok":
            raise _ModalFailure("reference modal factorization failed: "+str(handle.failure_reason))
        def action(vector):
            check("iteration")
            try:
                answer = np.asarray(handle.solve(Bs@project(vector)), dtype=float)
            except RuntimeError as exc:
                if (str(exc) == "Sparse solve produced NaN/Inf values" or
                        str(exc).startswith("Cannot solve with failed factorization:")):
                    raise _ModalFailure("reference modal sparse solve failed: "+str(exc)) from exc
                raise
            if not np.isfinite(answer).all():
                raise _ModalFailure("finite reference modal inverse action required")
            check("iteration_complete")
            return project(answer)
        values, vectors = eigs(LinearOperator(A.shape, matvec=action, dtype=float),
            k=request.count, which="LM", tol=1e-10, maxiter=request.max_iterations,
            v0=np.linspace(1., 2., A.shape[0]))
        raw = tuple(map(complex, values))
        check("after_iteration")
        abs_A, abs_B = abs(A), abs(B)
        pairs = []
        for mu, vector in zip(values, vectors.T):
            if (not np.isfinite(mu) or mu.real == 0. or
                    abs(mu.imag) > 1e-8*abs(mu.real) or
                    np.max(abs(vector.imag)) > 1e-8*np.max(abs(vector.real))):
                raise _ModalFailure("nonphysical or unresolved finite reference pair")
            eigenvalue = sigma+1./float(mu.real)
            if free:
                # Retain the separate shift/reconstruction cancellation guard.
                roundoff_screen = 64*np.finfo(float).eps*A.shape[0]*(abs(sigma)+abs(1./float(mu.real)))
                if not np.isfinite(roundoff_screen) or eigenvalue <= roundoff_screen:
                    raise _ModalFailure("extra mechanism or unresolved near-zero flexible pair")
            x = scale*np.asarray(vector.real)
            q = np.zeros(owner._K.shape[0]); q[owner._free] = x[:n]
            if free:
                # Physical quadratic-work resolution screen on the reconstructed
                # candidate (the congruence inverse applied to the Krylov vector).
                # The pair is resolvable only when its inertial quadratic work
                # exceeds the abs-coefficient quadratic-work roundoff of this
                # candidate's elastic and descriptor constraint work. Both sides
                # are physical energy, so the decision is invariant under
                # equivalent diagonal unit/coordinate rescalings. Absolute-entry
                # bounds can vary under general frame rotations; this is a
                # conservative resolution screen, not a frame-invariance proof.
                # Its coefficient uncertainty uses no fitted eigenvalue or shift. Cancellation/roundoff
                # screen only: not a certified Arnoldi/LU eigenvalue error bound
                # or exactly-six nullity proof.
                kinetic = float(q@mass@q)
                work = eigenvalue*kinetic
                uncertainty = 64*np.finfo(float).eps*A.shape[0]*(
                    float((abs(owner._K)@abs(q))@abs(q))+
                    2.*float((abs(owner._C).T@abs(x[n:]))@abs(q)))
                if (not np.isfinite(work) or not np.isfinite(uncertainty) or
                        work <= uncertainty):
                    raise _ModalFailure("extra mechanism or unresolved near-zero flexible pair")
            norm = float(q@mass@q)
            if not np.isfinite(eigenvalue) or eigenvalue <= 0. or not np.isfinite(norm) or norm <= 0.:
                raise _ModalFailure("positive finite eigenvalue and physical modal mass required")
            x /= np.sqrt(norm); q /= np.sqrt(norm)
            if not np.isfinite(x).all() or not np.isfinite(q).all():
                raise _ModalFailure("physical reference mode normalization is not representable")
            # Sign uses physical inertial coordinates, not massless resultants.
            physical_slots = np.setdiff1d(np.arange(len(q)), zero)
            largest = physical_slots[int(np.argmax(np.abs(q[physical_slots])))]
            if q[largest] < 0.: x = -x; q = -q
            lam = x[n:]
            residual = A@x-eigenvalue*(B@x)
            bound = abs_A@abs(x)+abs(eigenvalue)*(abs_B@abs(x))
            equation = _modal_scaled_residual(residual, bound, scale)
            gap = owner._C@q
            constraint = _modal_scaled_residual(gap, abs(owner._C)@abs(q), scale[n:])
            reactions = owner._K@q+owner._C.T@lam-eigenvalue*(mass@q)
            physical_bound = abs(owner._K)@abs(q)+abs(owner._C.T)@abs(lam)+abs(eigenvalue)*(abs(mass)@abs(q))
            physical = _modal_scaled_residual(reactions[owner._free],
                physical_bound[owner._free], scale[:n])
            if equation > 1e-8 or physical > 1e-8 or constraint > 1e-9:
                raise _ModalFailure("reference modal physical equation or constraint residual failed")
            pairs.append((float(eigenvalue), _owned(q), _owned(lam), _owned(reactions),
                          equation, constraint, physical))
        pairs.sort(key=lambda item: item[0])
        modes = np.column_stack([pair[1] for pair in pairs])
        gram = modes.T@mass@modes
        if not np.isfinite(gram).all() or np.max(abs(gram-np.eye(request.count))) > 1e-7:
            raise _ModalFailure("physical reference modal mass orthogonality failed")
        if free:
            overlap = float(np.max(abs(rigid["normalized"].T@mass@modes)))
            if not np.isfinite(overlap) or overlap > 1e-7:
                raise _ModalFailure("rigid and flexible physical mode spaces not mass orthogonal")
        check("publish")
        return finish("completed", frequencies_hz=tuple(np.sqrt(p[0])/(2*np.pi) for p in pairs),
            eigenvalues_rad2_s2=tuple(p[0] for p in pairs),
            full_coordinates=tuple(p[1] for p in pairs),
            shell_modes=tuple(_owned(p[1][:owner._shell_count]) for p in pairs),
            beam_modes=tuple(_owned(p[1][owner._shell_count:]) for p in pairs),
            multiplier_modes=tuple(p[2] for p in pairs), support_reactions=tuple(p[3] for p in pairs),
            equation_residuals=tuple(p[4] for p in pairs),
            constraint_residuals=tuple(p[5] for p in pairs),
            physical_equation_residuals=tuple(p[6] for p in pairs))
    except SolveCancelled as exc:
        return finish("cancelled", failure_reason=str(exc))
    except (ArpackError, _ModalFailure) as exc:
        partial = getattr(exc, "eigenvalues", None)
        if partial is not None: raw = tuple(map(complex, partial))
        return finish("failed", failure_reason=str(exc))
    finally:
        # Do not retain a factor through an operation closure/error traceback.
        handle = None
