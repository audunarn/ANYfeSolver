"""Owned physical actions for the experimental six-channel nodal pose law."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PointTransferWrench:
    """Connection loads exerted on each domain, in global spatial components.

    Actions are (Fx, Fy, Fz, Mx, My, Mz), with moments about the respective
    port positions. They are MINUS the connection's resisting residual, not
    material-basis multipliers or support reactions. Positions retain the
    mechanics' separate low part. Reference-linear actions use reference
    ports; finite actions use current ports. The enclosing accepted result
    supplies the generation identity. No torque-response qualification follows.
    """
    point_index: int
    shell_node_id: int
    beam_node_id: int
    configuration: str
    shell_position_m: tuple[float, float, float]
    beam_position_m: tuple[float, float, float]
    shell_position_low_m: tuple[float, float, float]
    beam_position_low_m: tuple[float, float, float]
    shell_action: tuple[float, float, float, float, float, float]
    beam_action: tuple[float, float, float, float, float, float]
    gap: tuple[float, float, float, float, float, float]


def point_record(owner, index, positions, position_low, resisting, gap, configuration):
    """Capture already evaluated values without element or joint evaluation."""
    point = owner._point_incidence[index]
    return PointTransferWrench(
        index, point.shell_node, point.beam_node, configuration,
        tuple(float(v) for v in positions[0]), tuple(float(v) for v in positions[1]),
        tuple(float(v) for v in position_low[0]), tuple(float(v) for v in position_low[1]),
        tuple(-float(v) for v in resisting[:6]), tuple(-float(v) for v in resisting[6:12]),
        tuple(float(v) for v in gap))


def reference_point_records(owner, q, multipliers):
    """Use the actual frozen reference map of the linear problem."""
    records = []
    for index, (joint, slots) in enumerate(zip(owner._connections, owner._joint_slots)):
        rows = owner._C[6*index:6*index+6]
        resisting = rows[:, slots].T @ multipliers[6*index:6*index+6]
        records.append(point_record(
            owner, index, joint.reference_positions, ((0.,)*3, (0.,)*3),
            resisting, rows @ q, "reference_linear"))
    return tuple(records)
