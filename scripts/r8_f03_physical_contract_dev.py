"""Prospective native F03 contract with exact local structural counterexamples.

Stdlib only. This is not an operator/norm construction or an actual-case run.
The equations distinguish signed mixed stationarity from physical work, and
record why the admitted external reference lacks the current transfer domain.
No threshold, multiplier policy, physics, execution or F03 acceptance is adopted.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/agent_plans/development/r8-f03-physical-contract-20261010"


def exact(value):
    if type(value) not in (int, Fraction):
        raise ValueError("exact integer/rational structural data required")
    return Fraction(value)


def scalar_stationary_identity(*, a, b, d, x, resultant_load=0):
    """One local complementary scalar, not a matrix factorization or solver."""
    a, b, d, x, rp = (exact(v) for v in (a, b, d, x, resultant_load))
    if d <= 0:
        raise ValueError("positive complementary scalar needed for this identity")
    p = (b*x-rp)/d
    s = a+b*b/d
    mixed = a*x*x+2*b*x*p-d*p*p
    stationary = x*x*s-rp*rp/d
    if mixed != stationary or b*x-d*p != rp:
        raise AssertionError("exact local stationarity/work identity failed")
    return {"x": x, "p": p, "stationary_S": s,
            "mixed_work": mixed, "signed_resultant_source": -rp*rp/d,
            "resultant_stationarity": b*x-d*p}


def spin_quotient_counterexample(*, completion_coefficient, spin=1):
    """Physical u/director are zero; total completion still sees theta_z."""
    coefficient, spin = exact(completion_coefficient), exact(spin)
    if coefficient <= 0 or spin == 0:
        raise ValueError("nonzero completion and spin counterexample required")
    return {"physical_displacement": (Fraction(0),)*3,
            "physical_director_change": (Fraction(0),)*3,
            "BL5_director_gap": (Fraction(0),)*2,
            "L1_relative_spin_gap": spin,
            "completion_work": coefficient*spin*spin,
            "total_work_quotient_descends": False}


def body_yaw_is_not_coupled_gauge(*, left, right, yaw=1):
    """u_y(s)=yaw*s, theta_z=yaw, shell=0 on a flat x-axis line.

    Beam root translations/rx/ry vanish, free root rz is admitted. A BL5
    translation row still sees the yaw. Director/offset spin terms are zero.
    """
    left, right, yaw = exact(left), exact(right), exact(yaw)
    if not 0 <= left < right or yaw == 0:
        raise ValueError("positive reference interval and nonzero yaw required")
    row = -yaw*(right*right-left*left)/2
    return {"P0_ty_gap_integral": row, "director_gap_integral": (Fraction(0),)*2,
            "body_null_does_not_imply_constraint_null": row != 0}


def constant_p1_intersection_obstruction(*, cell_tilts, nodal_tilts, root_tilt_fixed=True):
    """Exact scalar bonded tilt on aligned intervals, ignoring normal spin.

    A native cell constant equals an external P1 function everywhere iff its
    two endpoint values equal that constant. Shared endpoints imply all cell
    tilts equal. Fixed external root tilt then forces the common tilt to zero.
    """
    cells = tuple(exact(v) for v in cell_tilts)
    nodes = tuple(exact(v) for v in nodal_tilts)
    if not cells or len(nodes) != len(cells)+1:
        raise ValueError("one continuous P1 interval per declared cell required")
    matched = all(nodes[i] == cell == nodes[i+1] for i, cell in enumerate(cells))
    support_ok = not root_tilt_fixed or nodes[0] == 0
    if matched and support_ok and root_tilt_fixed and any(cells):
        raise AssertionError("nonzero supported tilt cannot lie in this exact intersection")
    return {"pointwise_equal": matched, "support_admitted": support_ok,
            "nonzero_supported_tilt_admitted": matched and support_ok and any(cells)}


def equal_average_not_equal_field(*, left_rotation, right_rotation):
    """P0 averages match, while a P1 tilt and its cell average differ pointwise."""
    left, right = exact(left_rotation), exact(right_rotation)
    if left == right:
        raise ValueError("nonconstant P1 counterexample required")
    average = (left+right)/2
    return {"native_cell_tilt": average, "external_end_tilts": (left, right),
            "P0_average_difference": Fraction(0),
            "left_pointwise_difference": average-left,
            "right_pointwise_difference": average-right,
            "same_physical_field": False}


def omitted_endpoint_rotation_is_not_physical_null():
    """Local scalar component of ell=sign*(theta_node-theta_cell).

    With zero translation/cell rotation and a free interior nodal rotation,
    the actual native trace pose and C action are zero. Complementary endpoint
    compatibility sees the rotation. Positive D gives positive stationary work.
    This rejects extending a trace-only auxiliary norm to the entire x space.
    """
    stationarity = scalar_stationary_identity(a=0, b=1, d=2, x=1)
    return {"translation_trace": 0, "cell_director_trace": 0,
            "Cx_action": 0, "selected_pose_trace_norm": 0,
            "endpoint_relative_rotation": 1,
            "stationary_mechanical_work": stationarity["mixed_work"],
            "physical_null": False}


def common_forty_density_dual_control():
    """Exact known support-major/channel-major physical five-row conventions.

    This tests dual row ordering/signs only, not an isomorphism of body fields.
    External gap order is (tx,ty,tz,rx,ry); native order is
    (-tx,-ty,-tz,-ry,+rx). Physical director densities stay in N here; any
    numerical director-length congruence is a separate operation.
    """
    source = tuple(Fraction(i+1) for i in range(40))
    gap = tuple(Fraction(40-i, 7) for i in range(40))
    native_density, native_gap = [], []
    for channel, sign in ((0, -1), (1, -1), (2, -1), (4, -1), (3, 1)):
        native_density.extend(sign*source[5*j+channel] for j in range(8))
        native_gap.extend(sign*gap[5*j+channel] for j in range(8))
    external_work = sum(a*b for a, b in zip(source, gap))
    native_work = sum(a*b for a, b in zip(native_density, native_gap))
    if native_work != external_work:
        raise AssertionError("physical 40-density transpose work convention failed")
    return {"coefficient_count": 40, "external_work": external_work,
            "native_work": native_work, "work_equal": True,
            "complete_body_field_equivalence": False}


def common_p0_yaw_balance(*, intervals, ty_density, applied_Mz):
    """Actual flat P0 force first moment; no body solve or gauge force."""
    cells = tuple((exact(a), exact(b)) for a, b in intervals)
    density = tuple(exact(v) for v in ty_density)
    mz = exact(applied_Mz)
    if (not cells or len(cells) != len(density) or any(not 0 <= a < b for a, b in cells) or
            any(cells[i][1] != cells[i+1][0] for i in range(len(cells)-1))):
        raise ValueError("ordered contiguous P0 supports and matching physical density required")
    moment = sum(value*(b*b-a*a)/2 for value, (a, b) in zip(density, cells))
    return {"shell_force_first_moment_Nm": moment, "applied_Mz_Nm": mz,
            "beam_yaw_compatibility_residual_Nm": mz-moment,
            "prescribed_body_load_balanced": mz == moment,
            "unbalanced_load_may_be_gauged_away": False}


def continuous_L1_vs_BL5_shear_control():
    """Straight zero-lift, e0=0 local conditional attachment proof.

    Unit projected tangent t=(4/5,3/5,0), director ez, raw chord derivative
    tau=(4/5,3/5,1/7). Full L1 aligns Ue1=t and Ue2=ez cross t;
    BL5 permits U=I because its director is still ez. No matrix/trig is used.
    """
    tx, ty, normal_derivative = Fraction(4, 5), Fraction(3, 5), Fraction(1, 7)
    if tx*tx+ty*ty != 1:
        raise AssertionError("unit projected tangent identity failed")
    full_pose_gamma_y = -ty*tx+tx*ty
    return {"full_L1_gamma_y": full_pose_gamma_y,
            "full_L1_gamma_z": normal_derivative,
            "BL5_director_gap": (Fraction(0),)*2,
            "BL5_gamma_y": ty, "BL5_gamma_z": normal_derivative,
            "BL5_field_admitted_by_full_L1": False,
            "scope": "continuous full-pose tie, straight zero lift, zero correspondence e0; not finite weak-row enforcement"}


def prospective_contract():
    """Fixed equations and unanswered admission predicates, not a norm recipe."""
    return {
        "status": "PROSPECTIVE_NATIVE_PHYSICAL_WORK_CONTRACT_NOT_ADOPTED",
        "qualification": False,
        "domain": "unprestressed flat reference BL5/director bond; six-row L1 is distinct",
        "variables": {
            "x": "shell nodal physical coordinates, beam nodal coordinates, independent cell rotations",
            "p": "18 native resultants per member (six force and twelve moment coordinates)",
            "lambda": "three translation-density and two director-density P0 coefficients"},
        "units": {"translation": "m", "rotation": "rad/dimensionless",
                  "resultants": "N (first six), Nm (remaining twelve)",
                  "translation_density": "N/m", "director_density": "N",
                  "integrated_translation_gap": "m^2", "integrated_director_gap": "m",
                  "work": "J"},
        "body_equations": {
            "potential": "p.k(x)-Psi*(p)+shell_physical_core+shell_completion",
            "tangent": "K=[[A,B.T],[B,-D]], D=Psi*''",
            "constraint": "C=[Cx,0]; multiplier saddle zero block",
            "local_elimination": "p=D^-1(Bx-rp) only when local D is proven SPD",
            "stationary_operator": "S=A+B.T D^-1 B",
            "stationary_quadratic": "q.T K q=x.T S x-rp.T D^-1 rp",
            "actual_line_load_roles": "BL5 Cp=0 => rp=0; retained cell-rotation loads stay in x",
            "static_condensation_is_not_mass": True},
        "physical_work": {
            "continuous_dual": "integral(lambda_t.dot(delta gt)+lambda_d.dot(delta gd)) ds",
            "common_variation": "v -> Pn v, Pe v on an explicitly justified shared physical field domain",
            "dual_pullback": "Pn.T fn and Pe.T fe; no inverse between unlike spaces",
            "supported_common_work": "ell_n(Pn v)=ell_e(Pe v) only for admitted fields and explicitly bounded remainder",
            "coordinate_congruence": "q=U*qhat; lambda=V*lamhat; Khat=U.T*K*U/E0; Chat=V.T*C*U/E0",
            "congruence_limits": "invertible role-aware U,V are units/bases only; do not alter nulls or infer physical mass"},
        "stability": {
            "full_mixed_SPD": "REJECTED: p-only directions have negative complementary work",
            "mechanical_domain": "fixed-supported ker(Cx); retain all physical cell rotations",
            "bilinear_form": "x.T Ssym y; raw skew and local-elimination uncertainty reported separately",
            "criterion": "coercivity only relative to an independently justified positive physical H; coupling inf-sup only with justified H and dual M",
            "coercivity_equation": "inf(v in supported kerCx / proven N) a_sym(v,v)/H(v,v) >= alpha0 > 0 uniformly over the registered growing family",
            "coupling_equation": "inf(mu!=0) sup(v!=0) |mu.T Cx v|/(sqrt(H(v,v))*sqrt(M(mu,mu))) >= beta0 > 0 uniformly over the registered growing family",
            "quotient_admission": "each removed N must satisfy support*N=0, Cx*N=0 and Ssym*N=0; completion-sensitive spin cannot be silently removed",
            "coupling_only_quotient": "Cx may descend to a trace quotient when its removed directions are Cx-null; this does not make S descend or certify mechanical stability on that quotient",
            "body_vs_coupled_null": "body yaw can have Cx*yaw !=0; body resolvent gauge is not a coupled-kernel quotient",
            "current_H_M_adoption": "UNRESOLVED; existing auxiliary trace/coefficient controls do not select a norm",
            "actual_auxiliary_embedding_scope": "existing actual_flat_primal_work omits nodal beam rotations; those are C-invisible but not mechanical nulls (endpoint compatibility remains)",
            "thresholds": "existing registered thresholds and seven ratios unchanged; no new measured-value floor"},
        "reference_obstruction": {
            "native_trace": "independent half-cell director Ucell*R0(s), endpoint translations plus physical lift",
            "external_trace": "continuous P1 beam rotations and translations",
            "pointwise_common_tilt": "only constant tilt across connected intervals; fixed root rx/ry => zero",
            "native_bending": "nodal-minus-cell endpoint rotations also enter native kinematics/complementary work",
            "missing_observable": "reference virtual action on nontrivial native retained tilt/endpoint-compatibility field, or adopted reconstructed continuum field with explicit remainder",
            "tangent_completion": "external all-column physical tangent and core/completion semantics UNESTABLISHED",
            "existing_evidence": "three density responses and total saved K support bounded work observations, not arbitrary-field equivalence"},
        "constructive_common_density_route": {
            "action_maps": "Ln=-Cn.T*T and Le=-Ce.T for common40 source densities; T is the admitted signed ordering map, not a primal transfer",
            "existing_data": "native full C plus external original40 integrated rows/MP scales supply every action column; no new external model needed",
            "body_load": "rj=fj+Lj*lambda, with cell-rotation action retained in native x",
            "rigid_compatibility": "Zj.T*rj=0 for each free body rigid mode; flat yaw reduces to sum(lambda_ty*(b²-a²)/2)=applied_Mz",
            "unbalanced_density": "refuse prescribed-load body solve or retain yaw/lambda in original coupled saddle; never gauge away nonzero physical rigid residual",
            "matched_compliance": "Wj(lambda)=rj.T*qj (equal to stationary total signed quadratic work when local stationary conditions hold); compare as approximate engineering responses, no exact primal equality required",
            "reference_gate": "arbitrary common-density compliance needs all-column physical K semantics or independent element-action application on each proposed load-generated variation; saved3actions do not certify this",
            "remaining_remainder": "distinct strain/completion reconstruction and beam approximation; physical/convergence bounds unresolved, no new observed-value metric",
            "independent_shell_side": "signed recovered P0 physical traction+director couple can act on common supported shell Q1 edge fields; nodal P1 dual values alone do not uniquely extend to arbitrary retained director jumps",
            "stability_separate": "balanced-load compliance comparisons do not prove the mixed-system mechanical/coupling stability inequalities"},
        "decision": "reject full-space transfer/unsupported quotient; specify missing reference field action before any new physical comparison",
        "preserved_obligations": ["seven ratios", "growing interface spaces", "three refinement levels",
                                  "locking/shear/drilling", "signed mechanical stability", "formal F03 unconsumed"]}


def exact_controls():
    return {
        "stationary_zero_source": scalar_stationary_identity(a=1, b=2, d=3, x=2),
        "stationary_nonzero_source": scalar_stationary_identity(a=1, b=2, d=3, x=2, resultant_load=1),
        "full_mixed_negative_direction": {"x": 0, "p": 1, "complementary_D": 3, "work": -3},
        "negative_stationary_mechanics": scalar_stationary_identity(a=-2, b=1, d=1, x=1),
        "spin_completion_quotient": spin_quotient_counterexample(completion_coefficient=2),
        "body_yaw_vs_coupled_kernel": body_yaw_is_not_coupled_gauge(left=0, right=Fraction(1, 8)),
        "nontrivial_supported_pointwise_tilt_refused": constant_p1_intersection_obstruction(cell_tilts=(1, 2), nodal_tilts=(0, 1, 2)),
        "zero_supported_pointwise_tilt": constant_p1_intersection_obstruction(cell_tilts=(0, 0), nodal_tilts=(0, 0, 0)),
        "P0_average_not_field_equality": equal_average_not_equal_field(left_rotation=0, right_rotation=2),
        "omitted_endpoint_rotation_is_not_physical_null": omitted_endpoint_rotation_is_not_physical_null(),
        "forty_density_dual_work": common_forty_density_dual_control(),
        "balanced_common_density_yaw": common_p0_yaw_balance(
            intervals=tuple((Fraction(i, 8), Fraction(i+1, 8)) for i in range(8)),
            ty_density=(200,)*8, applied_Mz=100)}


def encode(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(type(value).__name__)


def write_record(output, *, controls=False):
    output = Path(output)
    if output.exists():
        raise ValueError("unique output path required to preserve evidence")
    paths = [Path(__file__), ROOT / "src/anyfesolver/_ge_beam3_retained_generalized.py",
             ROOT / "src/anyfesolver/_ge_beam3_generalized_static_boundary.py",
             ROOT / "src/anyfesolver/_stiffened_shell_l1_local_experimental.py",
             ROOT / "src/anyfesolver/_stiffened_shell_bl5_local_experimental.py",
             ROOT / "src/anyfesolver/_stiffened_shell_l1_assembly_experimental.py",
             ROOT / "src/anyfesolver/e4_pl_element.py"]
    record = {"contract": prospective_contract(),
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "execution": "stdlib literal source/IO only" if not controls else "stdlib exact scalar/rational structural controls only; no numerical imports/actual matrices"}
    if controls:
        record["exact_structural_controls"] = exact_controls()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=encode)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exact-controls", action="store_true")
    args = parser.parse_args()
    write_record(args.output, controls=args.exact_controls)
