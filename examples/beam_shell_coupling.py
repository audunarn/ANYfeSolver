"""Standalone public beam-shell coupling example: registered pre-yield case.

This example builds the registered eccentric beam-shell coupling case with
public ``anyfesolver`` APIs only and solves it with the registered pre-yield
force-controlled controls. It reports a truthful JSON summary: ``completed``
is true only when the solver actually reached the requested load factor
within 1e-8 and the existing constraint postcheck passed; anything else
(including cooperative cancellation) is reported as non-completion and the
process exit status is nonzero.

Geometry and material follow the registered Slice 3 continuation fixture:
L = 1 m panel of width W = 0.24 m, t = 0.012 m, a 2x1 shell grid, one
10x40 mm stiffener at eccentricity e = +0.04 m, canonical S355 curve,
-2000 N uz on every shell-tip node, 6 load steps, at most 30 iterations per
step, tolerance 1e-8, corotational kinematics. ``direct`` couples each beam
node to one shell edge node; ``interpolated`` (default) couples genuinely
interpolated mid-span beam nodes to weighted shell masters.

Command line::

    python examples/beam_shell_coupling.py [--coupling direct|interpolated]
                                          [--case preyield|plastic]
                                          [--stage full|prefix]
                                          [--restart CHECKPOINT_PATH]
                                          [--checkpoint-out CHECKPOINT_PATH]
                                          [--output results.json]

The registered plastic case (``--case plastic``) solves the same model with
public displacement control on the tip probe node ``uz``: the full target
-11/60 m in 11 increments, or a completed prefix of -0.10 m in 6
increments (``--stage prefix``), each with at most 40 iterations per
increment and tolerance 1e-8. A prefix checkpoint written with
``--checkpoint-out`` can finish the remaining 5 increments to -11/60 m in
a later process with ``--restart PATH``. The restart checkpoint is saved
through ``result.restart_checkpoint_bytes()`` only when the solver
actually completed the requested target; cancellation, failure and
refused restarts never write or overwrite a checkpoint file. A restart
checkpoint that does not end at a continuable completed target is
refused by the solver and reported truthfully with a nonzero exit
status.

The JSON summary is always written to stdout; with ``--output`` the same
document is also written to the given path. Exit status 0 means valid
completion only; incompatible flag combinations are rejected with exit
status 2 before any solve runs.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from anyfesolver import (
    BoundaryCondition,
    CoupledBeamShellElement,
    CancellationToken,
    DisplacementControl,
    FEModel,
    InterpolatedBeamShellMPCElement,
    LoadCase,
    create_shell_element,
    solve_static_nonlinear,
    SolveCancelled,
)
from anyfesolver.elements import BeamElement
from anyfesolver.material_curves import DNVC208MaterialCurve, dnv_c208_steel_curve

L = 1.0
W = 0.24
T = 0.012
NX = 2
NY = 1
ECCENTRICITY = np.array([0.0, 0.0, 0.04])
BEAM_SECTION = {
    "area": 4.0e-4,
    "Iy": 0.01 * 0.04**3 / 12.0,
    "Iz": 0.04 * 0.01**3 / 12.0,
    "J": 1.0e-8,
    "b": 0.01,
    "h": 0.04,
    "fiber_plasticity": True,
}
TIP_LOAD_PER_NODE = -2000.0
NUM_STEPS = 6
MAX_ITERATIONS = 30
TOLERANCE = 1.0e-8
REQUESTED_LOAD_FACTOR = 1.0
COMPLETION_LOAD_FACTOR_TOL = 1.0e-8

COUPLINGS = ("direct", "interpolated")
DEFAULT_COUPLING = "interpolated"

# Registered plastic displacement-control case (Slice 3 continuation
# fixture): tip probe uz to -11/60 m in 11 increments, or a completed
# prefix of -0.10 m in 6 increments resumed with 5 increments, at most 40
# iterations per increment, tolerance 1e-8.
DC_TARGET = -11.0 / 60.0
DC_FULL_STEPS = 11
DC_PREFIX_TARGET = -0.10
DC_PREFIX_STEPS = 6
DC_RESUME_STEPS = 5
DC_MAX_ITERATIONS = 40
DC_TOLERANCE = 1.0e-8
COMPLETION_CONTROL_TOL = 1.0e-8
CASES = ("preyield", "plastic")
STAGES = ("full", "prefix")


@dataclass
class BeamShellCase:
    """Built, constrained and loaded model plus its probe node id."""

    coupling: str
    model: FEModel
    load: LoadCase
    probe_node_id: int


def _add_s355_material(model: FEModel) -> None:
    """Canonical S355 curve at the panel thickness (registered scale 1.0)."""
    base = dnv_c208_steel_curve("S355", T).as_dict()
    hardening = DNVC208MaterialCurve(
        base["sigma_prop"],
        base["sigma_yield"],
        base["sigma_yield_2"],
        base["eps_p_y1"],
        base["eps_p_y2"],
        base["K"],
        base["n"],
    )
    model.add_material(
        "steel",
        2.1e11,
        0.3,
        density=7850.0,
        yield_stress=base["sigma_yield"],
        hardening_curve=hardening,
    )


def build_model(coupling: str = DEFAULT_COUPLING) -> BeamShellCase:
    """Build the registered pre-yield case for ``direct`` or ``interpolated``.

    Node identities are deterministic construction ids: shell grid nodes
    1..(NX+1)*(NY+1), beam nodes 100+ib, MPC elements 200+ib and beam
    elements 300+ib. ``direct`` uses 2 beam elements on shell edge nodes;
    ``interpolated`` uses 3 beam elements with genuinely interpolated
    mid-span beam nodes.
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"unknown coupling {coupling!r}; expected one of {COUPLINGS}")
    model = FEModel(f"beam_shell_{coupling}")
    _add_s355_material(model)
    grid: dict[tuple[int, int], int] = {}
    node_id = 1
    for ix, x in enumerate(np.linspace(0.0, L, NX + 1)):
        for iy, y in enumerate(np.linspace(-W / 2.0, W / 2.0, NY + 1)):
            model.add_node(node_id, float(x), float(y), 0.0)
            grid[(ix, iy)] = node_id
            node_id += 1
    element_id = 1
    for ix in range(NX):
        for iy in range(NY):
            model.add_element(
                element_id,
                create_shell_element(
                    element_id,
                    [
                        grid[(ix, iy)],
                        grid[(ix + 1, iy)],
                        grid[(ix + 1, iy + 1)],
                        grid[(ix, iy + 1)],
                    ],
                    "steel",
                    thickness=T,
                ),
            )
            element_id += 1
    n_beam = 2 if coupling == "direct" else 3
    beam_nodes: list[int] = []
    for ib, xb in enumerate(np.linspace(0.0, L, n_beam + 1)):
        beam_node = 100 + ib
        model.add_node(beam_node, float(xb), -W / 2.0, float(ECCENTRICITY[2]))
        beam_nodes.append(beam_node)
        sx = xb / (L / NX)
        ix = min(int(np.floor(sx)), NX - 1)
        xi = sx - ix
        masters = [(grid[(ix, 0)], 1.0 - xi), (grid[(ix + 1, 0)], xi)]
        masters = [(n, w) for n, w in masters if w > 1.0e-12]
        mpc_id = 200 + ib
        if len(masters) == 1:
            model.add_element(
                mpc_id,
                CoupledBeamShellElement(
                    mpc_id, beam_node, masters[0][0], "steel",
                    eccentricity=ECCENTRICITY,
                ),
            )
        else:
            weights = np.array([w for _, w in masters], dtype=float)
            model.add_element(
                mpc_id,
                InterpolatedBeamShellMPCElement(
                    mpc_id, beam_node, [n for n, _ in masters], weights,
                    eccentricity=ECCENTRICITY,
                ),
            )
    for ib in range(n_beam):
        model.add_element(
            300 + ib,
            BeamElement(300 + ib, [beam_nodes[ib], beam_nodes[ib + 1]],
                        "steel", dict(BEAM_SECTION)),
        )
    model.add_boundary_condition(
        BoundaryCondition(
            "root",
            [grid[(0, iy)] for iy in range(NY + 1)],
            {name: 0.0 for name in ("ux", "uy", "uz", "rx", "ry", "rz")},
        )
    )
    model.apply_boundary_conditions()
    load = LoadCase("tip")
    for iy in range(NY + 1):
        load.add_nodal_load(
            grid[(NX, iy)],
            [0.0, 0.0, TIP_LOAD_PER_NODE, 0.0, 0.0, 0.0],
        )
    model.add_load_case(load)
    return BeamShellCase(
        coupling=coupling, model=model, load=load, probe_node_id=grid[(NX, 0)]
    )


class _AcceptedHistory:
    """Count accepted increments through public progress events."""

    def __init__(self, observer: Optional[Callable[[Any], None]] = None) -> None:
        self.accepted_increments = 0
        self.last_accepted_load_factor: Optional[float] = None
        self.last_control_value: Optional[float] = None
        self._observer = observer

    def __call__(self, event: Any) -> None:
        if event.get("type") == "nonlinear_static_step":
            self.accepted_increments += 1
            load_factor = event.get("load_factor")
            if load_factor is not None:
                self.last_accepted_load_factor = float(load_factor)
            if event.get("control_value") is not None:
                self.last_control_value = float(event["control_value"])
        if self._observer is not None:
            self._observer(event)


def _probe_uz(case: BeamShellCase, displacements: np.ndarray) -> float:
    dofs = case.model.mesh.dof_manager.get_node_dofs(case.probe_node_id)
    return float(np.asarray(displacements)[dofs[2]])


def run(
    coupling: str = DEFAULT_COUPLING,
    *,
    case: Optional[BeamShellCase] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[Callable[[Any], None]] = None,
) -> dict:
    """Solve the registered pre-yield case and return the JSON payload.

    The payload is truthful: ``completed`` requires solver status
    ``completed``, an achieved load factor within
    ``COMPLETION_LOAD_FACTOR_TOL`` of the requested one, and a passing
    existing constraint postcheck. A cancelled solve reports the accepted
    increment information visible through public progress events and never
    claims completion.
    """
    if case is None:
        case = build_model(coupling)
    history = _AcceptedHistory(progress_callback)
    try:
        result = solve_static_nonlinear(
            case.model,
            case.load,
            kinematics="corotational",
            num_steps=NUM_STEPS,
            max_iterations=MAX_ITERATIONS,
            tolerance=TOLERANCE,
            record_increment_snapshots=True,
            cancellation_token=cancellation_token,
            progress_callback=history,
        )
    except SolveCancelled as cancelled:
        return {
            "coupling": case.coupling,
            "status": "cancelled",
            "stop_reason": str(cancelled.reason),
            "requested_load_factor": REQUESTED_LOAD_FACTOR,
            "achieved_load_factor": history.last_accepted_load_factor,
            "probe_displacement": None,
            "accepted_increments": history.accepted_increments,
            "constraint_postcheck": None,
            "completed": False,
        }
    postcheck = result.info.get("constraint_postcheck")
    achieved = float(result.load_factor)
    completed = bool(
        result.status == "completed"
        and abs(achieved - REQUESTED_LOAD_FACTOR) <= COMPLETION_LOAD_FACTOR_TOL
        and isinstance(postcheck, dict)
        and postcheck.get("status") == "passed"
    )
    return {
        "coupling": case.coupling,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "requested_load_factor": REQUESTED_LOAD_FACTOR,
        "achieved_load_factor": achieved,
        "probe_displacement": _probe_uz(case, result.displacements),
        "accepted_increments": len(result.steps),
        "constraint_postcheck": postcheck,
        "completed": completed,
    }


def run_plastic(
    coupling: str = DEFAULT_COUPLING,
    *,
    stage: str = "full",
    restart_checkpoint=None,
    checkpoint_out=None,
    case: Optional[BeamShellCase] = None,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[Callable[[Any], None]] = None,
) -> dict:
    """Continue from a completed prefix through the existing checkpoint API.

    A failed or cancelled solve never replaces ``checkpoint_out``. The solver
    remains responsible for validating the checkpoint and refusing incomplete
    terminal states; this example does not reconstruct initial material state.
    """
    if stage not in ("full", "prefix"):
        raise ValueError("stage must be full or prefix")
    if restart_checkpoint is not None and stage != "full":
        raise ValueError("restart requires the full plastic stage")
    if case is None:
        case = build_model(coupling)
    target = DC_PREFIX_TARGET if stage == "prefix" else DC_TARGET
    increments = (DC_PREFIX_STEPS if stage == "prefix" else
                  DC_RESUME_STEPS if restart_checkpoint is not None else DC_FULL_STEPS)
    history = _AcceptedHistory(progress_callback)
    payload = {
        "case": "plastic", "coupling": case.coupling,
        "requested_control": target, "achieved_control": None,
        "control_error": None, "probe_displacement": None,
        "achieved_load_factor": None, "accepted_increments": 0,
        "constraint_postcheck": None, "completed": False,
        "checkpoint_written": False,
    }
    try:
        result = solve_static_nonlinear(
            case.model, case.load, control="displacement",
            displacement_control=DisplacementControl(
                node_id=case.probe_node_id, dof="uz", target_displacement=target),
            kinematics="corotational", num_steps=increments,
            max_iterations=DC_MAX_ITERATIONS, tolerance=DC_TOLERANCE,
            record_increment_snapshots=True, emit_restart_checkpoint=True,
            restart_checkpoint=restart_checkpoint,
            cancellation_token=cancellation_token, progress_callback=history,
        )
    except SolveCancelled as exc:
        payload.update(status="cancelled", stop_reason=exc.reason,
                       achieved_control=history.last_control_value,
                       probe_displacement=history.last_control_value,
                       achieved_load_factor=history.last_accepted_load_factor,
                       accepted_increments=history.accepted_increments)
        return payload
    except ValueError as exc:
        payload.update(status="refused", stop_reason=str(exc))
        return payload
    achieved = _probe_uz(case, result.displacements)
    postcheck = result.info.get("constraint_postcheck")
    error = abs(achieved - target)
    completed = bool(result.status == "completed" and error <= 1e-8
                     and isinstance(postcheck, dict) and postcheck.get("status") == "passed")
    payload.update(status=result.status, stop_reason=result.stop_reason,
                   achieved_control=achieved, probe_displacement=achieved,
                   control_error=error, achieved_load_factor=float(result.load_factor),
                   accepted_increments=len(result.steps), constraint_postcheck=postcheck,
                   completed=completed)
    if completed and checkpoint_out is not None:
        # Serialize before opening a file, and replace atomically only after the
        # complete checkpoint has been flushed. Existing recovery files survive
        # solver refusal, cancellation and serialization/write failure.
        data = result.restart_checkpoint_bytes()
        destination = Path(checkpoint_out)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
            payload["checkpoint_written"] = True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return payload


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registered pre-yield beam-shell coupling example"
    )
    parser.add_argument(
        "--coupling", choices=COUPLINGS, default=DEFAULT_COUPLING,
        help="beam-to-shell coupling law (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default=None,
        help="optional path that additionally receives the JSON document",
    )
    parser.add_argument("--case", choices=("preyield", "plastic"), default="preyield")
    parser.add_argument("--stage", choices=("full", "prefix"), default="full")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    args = parser.parse_args(argv)
    if args.case == "preyield" and (args.stage != "full" or args.restart or args.checkpoint_out):
        parser.error("stage/restart/checkpoint options require --case plastic")
    if args.restart and args.stage != "full":
        parser.error("--restart requires --stage full")
    if args.case == "plastic":
        restart = args.restart.read_bytes() if args.restart else None
        payload = run_plastic(args.coupling, stage=args.stage,
                              restart_checkpoint=restart, checkpoint_out=args.checkpoint_out)
    else:
        payload = run(args.coupling)
    document = json.dumps(payload, indent=2, allow_nan=False)
    print(document)
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(document + "\n")
    return 0 if payload["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
