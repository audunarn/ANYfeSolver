"""Selected cached finite outputs; no assembly or material-history replay."""
from dataclasses import dataclass, fields, replace
import numpy as np


@dataclass(frozen=True)
class GERecoverySelection:
    """Ordered model node/element IDs, not execution-array row numbers.

    Material element IDs refer to the IDs reported by accepted recovery. Empty
    groups request no output. Repeated IDs and untyped containers are rejected.
    """
    shell_nodes: tuple = ()
    beam_nodes: tuple = ()
    shell_material_elements: tuple = ()
    beam_material_elements: tuple = ()

    def __post_init__(self):
        groups = tuple(getattr(self, f.name) for f in fields(self))
        if not any(groups):
            raise ValueError("nonempty finite recovery selection required")
        for group in groups:
            if (type(group) is not tuple or any(type(v) is not int for v in group)
                    or len(set(group)) != len(group)):
                raise ValueError("unique ordered integer model IDs required")


@dataclass(frozen=True)
class SelectedGERecovery:
    model_identity: str
    generation: int
    status: str
    selection: GERecoverySelection
    shell_displacement: np.ndarray
    shell_frames: np.ndarray
    beam_positions: np.ndarray
    beam_frames: np.ndarray
    shell_support_coordinate_covectors: np.ndarray
    beam_support_coordinate_covectors: np.ndarray
    shell_material: tuple
    beam_station_accumulated: tuple
    load_position: float | None
    displacement_target: float | None
    amplitude: float | None
    production_qualified: bool = False
    beam_generalized_material: tuple = ()


def _owned(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def recover(owner, result, selection):
    if type(selection) is not GERecoverySelection:
        raise ValueError("typed finite recovery selection required")
    if owner._finite_driver is None:
        raise ValueError("selected finite recovery requires an accepted finite state")
    # Validate all IDs before producing output. These dictionaries contain only
    # indices/references; no full response array is copied.
    shell = owner._shell_index
    beam = owner._beam_index
    materials = {r.element_id: r for r in
                 getattr(result, "shell_q4_material", ()) +
                 getattr(result, "shell_s3_material", ())}
    accumulated = result.beam_station_accumulated
    beam_material = {e.element_id: i for i, (e, _, _) in enumerate(owner._beam_parts)
                     if i < len(accumulated)}
    for ids, available in ((selection.shell_nodes, shell),
                          (selection.beam_nodes, beam),
                          (selection.shell_material_elements, materials),
                          (selection.beam_material_elements, beam_material)):
        if any(i not in available for i in ids):
            raise ValueError("unknown or unavailable finite recovery ID")
    si = np.array([shell[i] for i in selection.shell_nodes], dtype=int)
    bi = np.array([beam[i] for i in selection.beam_nodes], dtype=int)
    records = []
    for element_id in selection.shell_material_elements:
        record = materials[element_id]
        records.append(replace(record, **{f.name: _owned(getattr(record, f.name))
            for f in fields(record) if isinstance(getattr(record, f.name), np.ndarray)}))
    reactions = result.support_reactions
    displacement = result.shell_displacement.reshape(-1, 6)[si]
    if result.shell_frames is None:
        from scipy.spatial.transform import Rotation
        frames = (Rotation.from_rotvec(displacement[:, 3:]).as_matrix()
                  if len(si) else np.empty((0, 3, 3)))
    else:
        frames = result.shell_frames[si]
    position = getattr(result, "load_position", None)
    if position is None:
        position = getattr(result, "load_factor", None)
    generalized={r.element_id:r for r in getattr(result,"beam_generalized_material",())}
    generalized_records=[]
    for element_id in selection.beam_material_elements:
        if element_id in generalized:
            record=generalized[element_id]
            stations=tuple(replace(row,**{f.name:_owned(getattr(row,f.name))
                for f in fields(row) if isinstance(getattr(row,f.name),np.ndarray)})
                for row in record.stations)
            generalized_records.append(replace(record,stations=stations))
    return SelectedGERecovery(owner._identity, result.generation, result.status,
        selection, _owned(displacement),
        _owned(frames), _owned(result.beam_positions[bi]),
        _owned(result.beam_frames[bi]),
        _owned(reactions[:owner._shell_count].reshape(-1, 6)[si]),
        _owned(reactions[owner._shell_count:owner._shell_count+6*len(beam)].reshape(-1, 6)[bi]),
        tuple(records), tuple(accumulated[beam_material[i]]
            for i in selection.beam_material_elements),
        position,
        getattr(result, "target", None), getattr(result, "amplitude", None),
        beam_generalized_material=tuple(generalized_records))
