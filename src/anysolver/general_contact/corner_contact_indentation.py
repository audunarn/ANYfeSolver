"""Development-only coupled elastic sphere / stiffened panel corner-contact statics.

Both bodies stay the delivered nonrigid models (320 S3 sphere, 100 original Q4
plate faces, rotating beam-shell ties, gravity on both). Unknowns are the reduced
coordinates of both bodies, six guide multipliers (reference-mass-weighted rigid
rows of the sphere, which carry the sphere's free rigid motion only during these
quasi-static increments, not any future drop) and one compressive force
multiplier per physical sphere corner. Nothing is committed unless EVERY gate
holds; a failed trial, cancellation or exception leaves the accepted state exact.
Development-only: no dynamics, friction, plasticity, impact or restart.
"""
from __future__ import annotations

import copy

import numpy as np

from anysolver import LoadCase, assemble_load_vector, assemble_mass_matrix

from .corner_contact_certificate import (
    CornerContactRefusal, PENETRATION_LIMIT_M, first_touch_lift, whole_face_certificates,
)
from .corner_contact_geometry import corner_gaps
from .corner_contact_solver import (
    Cancelled, Evaluation, MAX_ITERATIONS, MAX_REDUCTIONS, NewtonFailure, solve_ncp,
)
from .structural_state_bridge import BridgeState, TwoBodyStateBridge

GRAVITY = np.array([0.0, 0.0, -9.81])
MAX_BISECTIONS = 4
LOAD_DEPTH_M = 1e-3
LOAD_INCREMENTS = 4
STATUS_COMPLETED = "indentation_cycle_completed"
STATUS_PARTIAL = "indentation_partial"
STATUS_REFUSED = "indentation_refused"


class GateFailure(RuntimeError):
    def __init__(self, code, message, detail=None):
        self.code = code
        self.detail = detail
        super().__init__(message)


def _translation_sum(model, vector):
    total = np.zeros(3)
    for node in model.mesh.nodes.values():
        total += vector[np.asarray(node.dofs[:3], dtype=int)]
    return total


def _wrench(model, displacement, vector):
    force, moment = np.zeros(3), np.zeros(3)
    for node in model.mesh.nodes.values():
        d = np.asarray(node.dofs[:6], dtype=int)
        f = vector[d[:3]]
        force += f
        moment += np.cross(np.asarray(node.coords()) + displacement[d[:3]], f) + vector[d[3:]]
    return force, moment


class CoupledIndentation:
    """Accepted state, structural/contact evaluation and gated increments."""

    def __init__(self, case):
        self.case = case
        self.sphere, self.panel = case.sphere, case.panel
        self.sphere_map, self.panel_map = case.sphere_map, case.panel_map
        for model in (self.sphere, self.panel):
            model.apply_boundary_conditions()
        self.bridge = TwoBodyStateBridge(
            self.sphere, self.panel, case.poses["near"], case.panel_q,
            panel_history=dict(case.result.element_states))
        self.n_s, self.n_p = self.bridge.sizes
        self.total = self.n_s + self.n_p
        loads = [assemble_load_vector(m, LoadCase(name="gravity", gravity=GRAVITY))[0]
                 for m in (self.sphere, self.panel)]
        self.f_ext_parts = [np.asarray(v, dtype=float).reshape(-1) for v in loads]
        self.f_ext = np.concatenate(self.f_ext_parts)
        self.weight = {"sphere": float(-_translation_sum(self.sphere, self.f_ext_parts[0])[2]),
                       "panel": float(-_translation_sum(self.panel, self.f_ext_parts[1])[2])}
        self.f_ref = self.weight["sphere"] + self.weight["panel"]
        self.force_tolerance = 1e-8 * max(1.0, self.f_ref)
        self._build_guides()
        accepted = self.bridge.accepted()
        self.q = self.bridge.reduced_coordinates(accepted)
        self.mu = np.zeros(6)
        self.lam = {}
        self.target = self.guide_rows_full @ accepted.displacement
        self._trial_target = self.target.copy()
        self.z_open = float(self.target[2])
        self.cancel = None
        self.penetration_limit = PENETRATION_LIMIT_M
        self.bisections_left = MAX_BISECTIONS
        self.max_iterations = MAX_ITERATIONS
        # Sphere controls are affine reference-mass rows; panel coordinates do
        # not enter these rows. Preserve them during globalization of contact.
        self.affine_guide_filter = True
        self.last_result = None
        self.timing = dict(evaluations=0, structural_s=0.0, geometry_s=0.0,
                           tangent_reduction_s=0.0)
        self.panel_gravity_displacement = np.array(case.panel_q, copy=True)

    def _split(self, vector):
        vector = np.asarray(vector, dtype=float)
        return vector[:self.n_s], vector[self.n_s:]

    # -- guides ---------------------------------------------------------------
    def _build_guides(self):
        """Rows W = G^-1 R^T M of reference rigid fields (m for translation, rad for spin)."""
        from scipy import sparse
        mass = sparse.csr_matrix(assemble_mass_matrix(self.sphere)[0])
        rigid = np.zeros((self.n_s, 6))
        for node in self.sphere.mesh.nodes.values():
            d = np.asarray(node.dofs[:6], dtype=int)
            x = np.asarray(node.coords(), dtype=float)
            for a in range(3):
                rigid[d[a], a] = 1.0
                axis = np.eye(3)[a]
                rigid[d[:3], 3 + a] = np.cross(axis, x)
                rigid[d[3 + a], 3 + a] = 1.0
        gram = rigid.T @ (mass @ rigid)
        if not np.all(np.isfinite(gram)) or np.linalg.cond(gram) > 1e12:
            raise ValueError("rigid-field Gram system is singular or ill conditioned")
        rows = np.linalg.solve(gram, rigid.T @ (mass @ np.eye(self.n_s)))
        self.rigid_fields = rigid
        self.guide_rows_sphere = rows
        self.guide_rows_full = np.hstack((rows, np.zeros((6, self.n_p))))
        self.gram_condition = float(np.linalg.cond(gram))

    # -- evaluation -----------------------------------------------------------
    def _gradient_row(self, i, info, sphere_pose, panel_pose):
        u = info["direction"]
        row = np.zeros(self.total)
        row[:self.n_s] = sphere_pose.coordinate_map[3 * i:3 * i + 3, :].T @ u
        a, _ = self.panel_map.face_slices[info["face"]]
        for k, w in enumerate(info["weights"]):
            if w != 0.0:
                row[self.n_s:] -= w * (
                    panel_pose.coordinate_map[3 * (a + k):3 * (a + k) + 3, :].T @ u)
        return row

    def evaluate_trial(self, q, mu, lam):
        """Complete trial residual with no structural/contact iteration tangent."""
        return self.evaluate(q, mu, lam, tangent=False)

    def _contact_geometry(self, full, *, multipliers=None, return_audit=False):
        u_s, u_p = full[:self.n_s], full[self.n_s:]
        sphere_pose = self.sphere_map.pose(u_s)
        panel_pose = self.panel_map.pose(u_p)
        outcome = corner_gaps(
            sphere_pose.positions, self.panel_map, panel_pose,
            multipliers=multipliers, query_position_errors=sphere_pose.position_errors,
            return_audit=return_audit)
        if return_audit:
            infos, audit = outcome
            return sphere_pose, panel_pose, infos, audit
        return sphere_pose, panel_pose, outcome

    def _initial_guess(self):
        """Admit the speculative guide predictor or retain the accepted iterate.

        A refused predictor is not a refused accepted state.  In particular an
        overclosed rigid guess can have two closest seam points even when the
        coupled elastic path has an admissible Newton step from the accepted pose.
        No distance, normal or work branch is manufactured for that guess.
        """
        guess = self.q.copy()
        guess[:self.n_s] += self.rigid_fields[:, 2] * (
            self._trial_target[2] - self.target[2])
        try:
            self._contact_geometry(self.bridge.expand(guess))
        except CornerContactRefusal as refusal:
            if refusal.code != "nonunique_closest_point":
                raise
            from .corner_contact_solver import _observe
            _observe("predictor_refused", code=refusal.code,
                     restart="last_accepted_iterate", target_z=float(self._trial_target[2]))
            return self.q.copy()
        return guess

    def evaluate(self, q, mu, lam, *, tangent=True):
        import time
        bridge = self.bridge
        t0 = time.perf_counter()
        full = bridge.expand(q)
        t1 = time.perf_counter()
        state = BridgeState(bridge._owner, full, {}, {})
        sphere_pose, panel_pose, infos, inactive_audit = self._contact_geometry(
            full, multipliers=lam, return_audit=True)
        ids = sorted(infos)
        rows = np.array([self._gradient_row(i, infos[i], sphere_pose, panel_pose)
                         for i in ids]).reshape(len(ids), self.total)
        t2 = time.perf_counter()
        # Refuse inadmissible geometry before the expensive element response.
        f_int, k_blocks, histories = bridge.response(full, tangent=tangent)
        structural_end = time.perf_counter()
        lam_arr = np.array([lam.get(i, 0.0) for i in ids])
        f_contact = lam_arr @ rows if ids else np.zeros(self.total)
        f_guide = self.guide_rows_full.T @ mu
        r_full = f_int - self.f_ext - f_contact - f_guide
        # J^T f is exactly NonlinearRotatingMPCMap.reduce_force (identity map for the sphere).
        jac = bridge.block_jacobian(full)
        residual = np.asarray(jac.T @ r_full).reshape(-1)
        stiffness = (bridge.reduce_tangent_at(full, k_blocks, r_full).toarray()
                     if tangent else None)
        gradients = (np.asarray(jac.T @ rows.T).T if ids else np.zeros((0, q.size)))
        guide_red = np.asarray((jac.T @ self.guide_rows_full.T).T)
        guide_error = self.guide_rows_full @ full - self._trial_target
        t3 = time.perf_counter()
        self.timing["evaluations"] += 1
        self.timing["structural_s"] += (t1 - t0) + (structural_end - t2)
        self.timing["geometry_s"] += t2 - t1
        self.timing["tangent_reduction_s"] += t3 - structural_end
        extra = dict(inactive_corner_proofs=inactive_audit,
                     state=state, histories=histories, f_int=f_int, r_full=r_full,
                     f_contact=f_contact, f_guide=f_guide, infos=infos, rows=rows,
                     sphere_pose=sphere_pose, panel_pose=panel_pose,
                     lam=dict(zip(ids, lam_arr.tolist())))
        # Guard again AFTER the batched Jacobian products: the model identity must be
        # unchanged between the validated state and every derived quantity.
        bridge._check_models()
        scale = float(np.linalg.norm(np.asarray(jac.T @ (self.f_ext + f_guide)).reshape(-1)))
        extra["force_scale"] = scale
        return Evaluation(residual, stiffness, ids, np.array([infos[i]["gap"] for i in ids]),
                          gradients, guide_error, guide_red, extra, force_scale=scale)

    # -- acceptance -----------------------------------------------------------
    def _gates(self, result):
        ev = result["ev"]
        x = ev.extra
        tol = self.force_tolerance
        cert = whole_face_certificates(self.sphere_map, x["sphere_pose"],
                                       self.panel_map, x["panel_pose"])
        if cert["unavailable_pairs"] or cert["minimum_bound_m"] < -self.penetration_limit:
            raise GateFailure("whole_face_certificate", "whole-face penetration bound not "
                              "certified", cert)
        lam = result["lam"]
        if any(v < 0.0 for v in lam.values()):
            raise GateFailure("negative_multiplier", "accepted multiplier is negative")
        u_s, u_p = self._split(x["state"].displacement)
        from examples.elastic_sphere_stiffened_panel import (
            FORCE_BALANCE_TOLERANCE, MOMENT_BALANCE_TOLERANCE, REACTION_AGREEMENT_TOLERANCE,
            _corotational_internal_force, _named_support_resultants,
        )
        f_c_s, f_c_p = self._split(x["f_contact"])
        f_g_s, _ = self._split(x["f_guide"])
        _, f_i_p = self._split(x["f_int"])
        imbalance_p = f_i_p - self.f_ext_parts[1] - f_c_p
        rotating = self.bridge._bodies[1].rotating

        def transfer(imbalance):
            return (imbalance if rotating is None else
                    rotating.transfer_slave_imbalance_to_masters(u_p, imbalance))
        transferred = transfer(imbalance_p)
        manager = self.panel.mesh.dof_manager
        reaction_vector = np.zeros(self.n_p)
        for dof in range(self.n_p):
            if manager.is_constrained(dof):
                reaction_vector[dof] = transferred[dof]
        # Independent recovery: the public corotational element force law, evaluated
        # without the solver's histories, against the reaction from the coupled assembly.
        recovered = transfer(_corotational_internal_force(self.panel, u_p)
                             - self.f_ext_parts[1] - f_c_p)
        reported_named = _named_support_resultants(self.panel, transferred)
        recovered_named = _named_support_resultants(self.panel, recovered)
        deviation = max((abs(a - b) / max(1.0, abs(a))
                         for name, values in reported_named.items()
                         for a, b in zip(values, recovered_named[name])), default=0.0)
        if set(reported_named) != set(recovered_named) or deviation > REACTION_AGREEMENT_TOLERANCE:
            raise GateFailure("reaction_agreement", "independent support recovery disagrees",
                              dict(deviation=deviation))
        # Pair wrench: corner forces on the sphere and the equal/opposite weighted
        # panel corner forces, pulled back through the spin maps.
        fs = np.zeros((len(self.sphere_map.dofs), 3))
        fp = np.zeros((len(self.panel_map.dofs), 3))
        resultant = np.zeros(3)
        for i, value in lam.items():
            info = x["infos"][i]
            fs[i] += value * info["direction"]
            a, _ = self.panel_map.face_slices[info["face"]]
            for k, w in enumerate(info["weights"]):
                fp[a + k] -= value * w * info["direction"]
            resultant += value * info["direction"]
        ws = _wrench(self.sphere, u_s, self.sphere_map.spin_pullback(x["sphere_pose"], fs))
        wp = _wrench(self.panel, u_p, self.panel_map.spin_pullback(x["panel_pose"], fp))
        # Physical (current-position, spatial-spin) wrenches; external, guide, contact and
        # support loads only -- no internal-force resultant can hide an error here.
        gravity_s = _wrench(self.sphere, u_s, self.f_ext_parts[0])
        gravity_p = _wrench(self.panel, u_p, self.f_ext_parts[1])
        guide = _wrench(self.sphere, u_s, f_g_s)
        supports = _wrench(self.panel, u_p, reaction_vector)
        sphere_w = [a + b + c for a, b, c in zip(gravity_s, ws, guide)]
        panel_w = [a + b + c for a, b, c in zip(gravity_p, wp, supports)]
        total_w = [a + b for a, b in zip(sphere_w, panel_w)]
        pair_w = [a + b for a, b in zip(ws, wp)]
        norms = {name: (float(np.linalg.norm(w[0])), float(np.linalg.norm(w[1])))
                 for name, w in (("sphere", sphere_w), ("panel", panel_w),
                                 ("whole_coupled", total_w), ("contact_pair", pair_w))}
        checks = dict(
            wrench_force_N={k: v[0] for k, v in norms.items()},
            wrench_moment_Nm={k: v[1] for k, v in norms.items()},
            force_tolerance_N=FORCE_BALANCE_TOLERANCE,
            moment_tolerance_Nm=MOMENT_BALANCE_TOLERANCE,
            reaction_agreement=dict(max_relative_deviation=deviation,
                                    tolerance=REACTION_AGREEMENT_TOLERANCE),
            named_support_resultants=reported_named,
            sphere_weight_N=self.weight["sphere"], panel_weight_N=self.weight["panel"],
            guide_force_N=guide[0].tolist(), contact_on_sphere_N=ws[0].tolist(),
            contact_on_panel_N=wp[0].tolist(), support_force_N=supports[0].tolist())
        if (any(v[0] > FORCE_BALANCE_TOLERANCE for v in norms.values())
                or any(v[1] > MOMENT_BALANCE_TOLERANCE for v in norms.values())):
            raise GateFailure("balance", "force/moment balance exceeds the frozen case bound",
                              checks)
        return cert, checks, resultant, supports[0], None

    # -- increments -----------------------------------------------------------
    def solve_to(self, z_target):
        """Solve one increment; commit only after every gate passes."""
        self._trial_target = self.target.copy()
        self._trial_target[2] = float(z_target)
        try:
            # The prescribed guide increment is a rigid vertical translation,
            # not a new accepted state.  W R = I by the reference-mass guide
            # construction; elastic coordinates and the panel preload stay free.
            # Starting at the new guide height exposes closing gaps to the active
            # set instead of asking a free, guide-only Newton step to cross contact.
            guess = self._initial_guess()
            result = solve_ncp(self, guess, self.mu, self.lam, f_ref=self.f_ref, l_ref=1.0,
                               force_tolerance=self.force_tolerance, cancel=self.cancel,
                               max_iterations=self.max_iterations)
            self.last_result = result
            cert, checks, resultant, reaction, _ = self._gates(result)
            x = result["ev"].extra
            state = x["state"]
            self.bridge.commit(state, x["histories"])
        except BaseException:
            self._trial_target = self.target.copy()
            raise
        self.q, self.mu, self.lam = result["q"], result["mu"], result["lam"]
        self.target = self._trial_target.copy()
        return self._record(result, cert, checks, resultant, reaction)

    def advance(self, z_goal):
        """Increment toward ``z_goal`` with at most four bisections over the cycle."""
        records, targets = [], [float(z_goal)]
        while targets:
            try:
                records.append(self.solve_to(targets[-1]))
                targets.pop()
            except GateFailure as failure:
                # A converged state failed scientific acceptance. Do not obscure
                # that result behind automatic increment retries or a later timeout.
                self.last_gate_failure = dict(code=failure.code,detail=failure.detail,
                                              target_z=float(targets[-1]))
                from .corner_contact_solver import _observe
                _observe("acceptance_refused", **self.last_gate_failure)
                raise
            except (NewtonFailure, CornerContactRefusal):
                if self.bisections_left <= 0:
                    raise
                self.bisections_left -= 1
                targets.append(0.5 * (self.target[2] + targets[-1]))
        return records

    # -- reporting ------------------------------------------------------------
    def nonrigid_sphere_deformation_m(self, displacement=None):
        u_s = self._split((self.bridge.accepted().displacement
                                  if displacement is None else displacement))[0]
        elastic = u_s - self.rigid_fields @ (self.guide_rows_sphere @ u_s)
        return float(max(np.linalg.norm(elastic[np.asarray(n.dofs[:3], dtype=int)])
                         for n in self.sphere.mesh.nodes.values()))

    def panel_change_m(self):
        u_p = self._split(self.bridge.accepted().displacement)[1]
        delta = u_p - self.panel_gravity_displacement
        return float(np.max(np.abs(delta[np.asarray(
            [n.dofs[2] for n in self.panel.mesh.nodes.values()], dtype=int)])))

    def _record(self, result, cert, checks, resultant, reaction):
        ev = result["ev"]
        x = ev.extra
        active = sorted(result["lam"])
        return dict(
            guide_z_m=float(self.target[2]), iterations=result["iterations"],
            active_corners=len(active), candidate_corners=len(ev.ids),
            inactive_corner_proofs=x.get("inactive_corner_proofs", []),
            multipliers_N={int(i): float(result["lam"][i]) for i in active},
            accepted_gaps_m={int(i): float(g) for i, g in zip(ev.ids, ev.gaps) if i in active},
            minimum_candidate_gap_m=float(ev.gaps.min()) if len(ev.ids) else None,
            contact_resultant_N=resultant.tolist(),
            guide_multipliers=result["mu"].tolist(),
            whole_face_certificate=dict(minimum_bound_m=cert["minimum_bound_m"],
                                        pairs_certified=cert["pairs_certified"],
                                        pairs_pruned=cert["pairs_pruned_disjoint"],
                                        unavailable=len(cert["unavailable_pairs"])),
            balance=checks, panel_support_reaction_N=reaction.tolist(),
            nonrigid_sphere_deformation_m=self.nonrigid_sphere_deformation_m(
                x["state"].displacement),
            panel_vertical_change_from_gravity_m=float(np.max(np.abs(
                self._split(x["state"].displacement)[1][np.asarray(
                    [n.dofs[2] for n in self.panel.mesh.nodes.values()], dtype=int)]
                - self.panel_gravity_displacement[np.asarray(
                    [n.dofs[2] for n in self.panel.mesh.nodes.values()], dtype=int)]))),
            newton_trace=result["trace"])

    def first_touch_height(self):
        """Guide z at which the equilibrated open sphere first touches the ACTUAL panel."""
        state = self.bridge.accepted()
        u_s, u_p = self._split(state.displacement)
        sphere = self.sphere_map.pose(u_s)
        panel = self.panel_map.pose(u_p)
        lift, corners, lifts = first_touch_lift(sphere.positions, panel.positions,
                                                self.panel_map.face_slices)
        if len(lifts) > 1 and not np.isfinite(lifts[0] - lifts[1]):
            raise CornerContactRefusal("ambiguous_onset", "first-touch height is not finite")
        return float(self.target[2] + lift), corners, lifts

    def prepare_open(self):
        """Open equilibrium at the existing nominal 5 mm clearance (both gravities)."""
        record = self.advance(self.z_open)[-1]
        weight = self.weight["sphere"]
        guide_force = _translation_sum(self.sphere, self._split(
            self.guide_rows_full.T @ self.mu)[0])
        record["open_sphere_weight_N"] = weight
        record["open_guide_reaction_N"] = guide_force.tolist()
        record["guide_balances_consistent_gravity"] = bool(
            abs(guide_force[2] - weight) <= self.force_tolerance)
        return record


def run_cycle(case=None, *, cancel=None, stages=None):
    """Five-target path open, first touch, -1 mm, first touch, original open."""
    from examples.elastic_sphere_panel_contact_work import prepare_work_case
    case = prepare_work_case() if case is None else case
    sim = CoupledIndentation(case)
    sim.cancel = cancel
    report = dict(status=STATUS_PARTIAL, development_only=True, impact="unsupported",
                  restart_checkpoint=False, dynamics=False, friction=False, plasticity=False,
                  scope="quasi-static open/touch/load/unload only; no drop, no restart",
                  settings=dict(max_iterations=MAX_ITERATIONS, max_reductions=MAX_REDUCTIONS,
                                max_bisections=MAX_BISECTIONS, force_ref_N=sim.f_ref,
                                length_ref_m=1.0, penetration_limit_m=sim.penetration_limit,
                                gap_tolerance_m=1e-12, force_tolerance_N=sim.force_tolerance,
                                load_depth_m=LOAD_DEPTH_M, load_increments=LOAD_INCREMENTS),
                  increments=[])
    try:
        open_record = sim.prepare_open()
        q_open = sim.q.copy()
        report["open"] = open_record
        z_touch, corners, lifts = sim.first_touch_height()
        report["first_touch"] = dict(guide_z_m=z_touch, lowest_corner_ids=corners,
                                     lifts_m=lifts)
        path = [("to_first_touch", z_touch, 1),
                ("load", z_touch - LOAD_DEPTH_M, LOAD_INCREMENTS),
                ("unload", z_touch, LOAD_INCREMENTS),
                ("to_open", sim.z_open, 1)]
        if stages is not None:
            path = path[:stages]
        for name, goal, count in path:
            start = sim.target[2]
            for k in range(1, count + 1):
                for record in sim.advance(start + (goal - start) * k / count):
                    record["stage"] = name
                    report["increments"].append(record)
        if stages is None:
            error = np.abs(sim.q - q_open) - (1e-10 + 1e-8 * np.abs(q_open))
            report["open_recovery"] = dict(
                rtol=1e-8, atol=1e-10, maximum_excess=float(np.max(error)),
                recovered=bool(np.all(error <= 0.0)))
            if not report["open_recovery"]["recovered"]:
                raise GateFailure("open_recovery", "final open state does not recover")
            report["status"] = STATUS_COMPLETED
    except Cancelled:
        report["failure"] = "cancelled"
    except (NewtonFailure, CornerContactRefusal, GateFailure) as error:
        report["status"] = STATUS_REFUSED
        report["failure"] = dict(code=getattr(error, "code", type(error).__name__),
                                 message=str(error))
    report["bisections_used"] = MAX_BISECTIONS - sim.bisections_left
    return report, sim
