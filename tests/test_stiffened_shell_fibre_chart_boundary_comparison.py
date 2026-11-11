"""Controlled yield-boundary tangent-selection comparison for cyclic fibre charts.

Cross-chart cyclic comparison of recovered native physical fibre records.
Every selected tangent/branch is re-verified against the actual unchanged
constitutive law and the record's own captured accepted predecessor. Two
verification routes exist, per record:

- Exact reconstruction (independent counterexample units): when the
  recorded station strain pairs reproduce the law's fibre strain exactly
  (Decimal prec 80, the product response's own fibre map and summation
  order), the recorded recovery must be precisely the law's output and
  the captured accepted history must be exactly the predecessor plus the
  recorded increment.
- Exact local cell replay (actual cell-recovery captures): recover_material
  invokes the deterministic coupled cell response once per fibre beam -
  cell.response(p18, accepted predecessor, include_energy=True) on the
  retained resultants p18 - and the underlying cell computes the station
  strain as a Decimal80 value, publishes pair(strain) per component and
  constructs the FibreRecovery from the exact Decimal80 fibre strain, so
  every published pair loses the 80-digit tail and no reconstruction from
  the pairs can re-derive the law's inputs or outputs. When the caller
  supplies the capture's actual retained-fibre operators (cell_operators),
  the comparator re-runs that same deterministic response from the
  capture's own retained resultants and accepted predecessor rows and
  requires every recorded station field, fibre record, accepted history
  row and energy record to be exactly its output - the true inputs, never
  the rounded station pairs. This is local constitutive replay only: no
  global equilibrium is solved and no acceptance criterion is claimed
  here. Compensated operators and malformed captures are refused with a
  precise obstruction rather than dropping verification.

The earlier derived representation-error interval route was removed after
the root review rejected its proof: strain-unit pads were added to stress
rounding bounds, the yield-boundary pin solved the wrong equation
(radius/E - z0 instead of z0 + radius/E, excluding valid negative
boundaries and nonzero accepted plastic strain), and a heuristic 32-ULP
margin stood in for directed rounding. It is preserved verbatim in
.restart-ci/chart-comparison-delivery/interval-route-04-preserved.py.

Equality stays required on smooth and same-branch selections through the
existing 1e-9 physical comparator; only a pair whose own signed yield
functions straddle zero (opposing signs, or an exact zero on one side, on
the same hardening segment) AND whose selected branches are an elastic-side
selection against a plastic-side selection, or a valid ELASTIC vs
YIELD_BOUNDARY pair, may report the two valid one-sided tangents. No
surface-distance band, no hardening-knot exemption (a HARDENING_KNOT
selection never pairs with a different branch), no forged labels. No
solver runs, no product APIs, no constitutive change; physical 1e-9
state/energy scales and exact reset/restored checks elsewhere are
untouched. Historical failures stay immutable.
"""
from dataclasses import dataclass, replace
from decimal import Decimal as D, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from anyfesolver._ge_beam3_fibre_cell import CellHistory
from anyfesolver._ge_beam3_fibre_section import (
    Fibre, FibreHistory, FlowCurve, PhysicalFibreSection)
from anyfesolver._ge_beam3_retained_fibre import RetainedFibreOperator
from anyfesolver._ge_beam3_station_resultant_cell import history_row, pair
from anyfesolver._stiffened_shell_ge_finite_experimental import (
    BeamFibreEnergyRecovery, BeamFibreStationRecovery)
from test_stiffened_shell_staged_material_history import _material_close


# Root decision: no additional distance threshold. A straddle is admitted
# only by the paired constitutive evidence itself - opposing or exactly zero
# signed yield-function values on the same hardening segment plus the
# existing 1e-9 physical agreement. (The early review's 1.2e-11 relative was
# an arithmetic error - 5e-6 Pa / 4.14e8 Pa is about 1.2e-14 - and no
# distance number is used as a criterion here.)
# Root amendment: no hardening-knot exemption. The selected branch pair
# itself must be an elastic-side selection against a plastic-side selection,
# or a valid ELASTIC vs YIELD_BOUNDARY pair; a HARDENING_KNOT selection
# never pairs with a different branch, so a true knot branch mismatch stays
# refused, and same-branch matches keep the existing comparison.
ELASTIC_SIDE = ('ELASTIC', 'YIELD_BOUNDARY')
PLASTIC_SIDE = ('PLASTIC_POSITIVE', 'PLASTIC_NEGATIVE')
YOUNG = 210e9
YIELD_STRESS = 250e6
HARDENING = 210e9


@dataclass(frozen=True)
class _ShellStandIn:
    young_pa: float = YOUNG
    thickness_m: float = 5e-3
    poisson: float = .3


@dataclass(frozen=True)
class _DefinitionStandIn:
    shell: _ShellStandIn = _ShellStandIn()


DEFINITION = _DefinitionStandIn()


@dataclass(frozen=True)
class _Selection:
    """The law's own re-evaluated selection from the accepted predecessor.

    signed is the law's |drive|-radius as an exact enclosure (slo, shi): a
    point (v, v) on the exact route; on the cell-replay route an exact sign
    enclosure determined by the replay-verified branch selection itself
    (ELASTIC strictly inside, YIELD_BOUNDARY exactly on the surface,
    plastic strictly beyond) - no observed value, no margin. segment and
    knot are exact in both routes.
    """
    branch: str
    signed: tuple
    segment: int
    knot: bool


@dataclass(frozen=True)
class _CellOperator:
    """One beam's actual retained-fibre operator and its capture row."""
    operator: RetainedFibreOperator
    index: int


@dataclass(frozen=True)
class _StationPlan:
    record: object
    section: PhysicalFibreSection
    rows_after: np.ndarray
    rows_before: np.ndarray
    operator: _CellOperator = None


def _paired(value):
    high, low = pair([value])
    return (high[0], low[0])


def fibre_sections(model):
    """Declared physical fibre sections of the model, by element id."""
    sections = {}
    for native in model.beams:
        element, _ = native.instantiate()
        if type(element.section) is PhysicalFibreSection:
            sections[element.element_id] = element.section
    return sections


def cell_operators(model):
    """Actual retained-fibre operators of the declared model, by element id.

    The mixed owner assembles its beam parts in definition.beams order
    (_beam_reference appends one part per defined beam), so a beam's part
    index is its position in the definition and the captured
    cell_resultants rows follow the same order. No owner, solve or
    product API is constructed here; the operators are the definition's
    own local cell compilers.
    """
    cells = {}
    for index, native in enumerate(model.beams):
        element, _ = native.instantiate()
        if type(element.section) is PhysicalFibreSection:
            if type(element.operator) is not RetainedFibreOperator:
                raise ValueError('declared fibre beam has no retained fibre operator')
            cells[element.element_id] = _CellOperator(element.operator, index)
    return cells


def _replay_beam(group, capture, cell_operator):
    """Exact local replay of one beam's actual cell from the capture itself.

    Re-runs the beam's own retained-fibre operator - the same deterministic
    recover_material that produced the capture's records: one coupled cell
    response on the capture's retained resultants p18 with the accepted
    predecessor rebuilt from the previous capture's accepted rows,
    include_energy=True - and requires every recorded station field, fibre
    record, accepted history row and energy record to be exactly its
    output. The true inputs are the capture's own retained resultants and
    predecessor rows, never the rounded station pairs, so the lost 80-digit
    tail is irrelevant: the record must be precisely the replay's output.
    Compensated (paired-update) operators are refused - only uncompensated
    captures have an unchanged single-argument replay.
    """
    operator = cell_operator.operator
    if operator.arithmetic_policy is not None:
        raise ValueError('compensated retained-fibre captures have no '
            'uncompensated local cell replay')
    section = group[0].section
    if len(group) != len(operator.cell.stations):
        raise ValueError('captured station inventory does not match the declared operator')
    origin = CellHistory(operator.cell.identity, tuple(
        FibreHistory(section.identity,
            tuple(tuple(float(v) for v in row) for row in entry.rows_before))
        for entry in group))
    resultants = np.asarray(capture['result'].cell_resultants, dtype=float)
    frames = np.asarray(capture['result'].cell_frames, dtype=float)
    if (resultants.ndim != 2 or resultants.shape[1] != 18
            or resultants.shape[0] <= cell_operator.index
            or frames.shape != (resultants.shape[0], 2, 3, 3)):
        raise ValueError('capture lacks the declared beam retained resultants and cell frames')
    response = operator.recover_material(frames[cell_operator.index],
        resultants[cell_operator.index], origin=origin)
    if len(response.stations) != len(group):
        raise AssertionError('replayed station inventory does not match the capture')
    energies = [record for record in capture['result'].beam_fibre_energy
        if record.element_id == group[0].record.element_id]
    if len(energies) != 1:
        raise AssertionError('capture lacks the beam unique energy record')
    energy = energies[0]
    assert energy.stored_energy_j == response.stored_energy \
        and energy.previous_hardening_energy_j == response.previous_hardening_energy \
        and energy.incremental_dissipation_j == response.incremental_dissipation \
        and energy.incremental_material_potential_j == response.material_potential \
        and energy.energy_recovery_station_updates == response.energy_recovery_station_updates, \
        'recorded beam energy record is not the replayed cell output'
    rows = {}
    for entry, row in zip(group, response.stations):
        record = entry.record
        assert (record.cell, record.station, record.xi, record.measure) == (
            row['cell'], row['station'], row['xi'], row['measure']), \
            'recorded station does not match the replayed cell response'
        for name in ('strain', 'strain_low', 'resultants', 'resultants_low',
                'reference_frame', 'current_frame'):
            assert np.array_equal(np.asarray(getattr(record, name)),
                np.asarray(row[name])), \
                'recorded station field is not the replayed cell output'
        assert record.fibres == row['fibres'], \
            'recorded fibre recovery is not the replayed cell output'
        assert tuple(float(v) for v in entry.rows_after.ravel()) == \
            tuple(float(v) for fibre_row in row['history'].rows for v in fibre_row), \
            'accepted history is not the replayed cell output'
        rows[(record.cell, record.station)] = row
    return rows


def _replay_selection(fibre, row_before, replayed):
    """Sign-exact selection derived from the replay-verified law output.

    _replay_beam proved the record is precisely the law's output for the
    true retained resultants and accepted predecessor, so the law's own
    branch label determines the signed yield function exactly: ELASTIC is
    strictly inside the surface, YIELD_BOUNDARY exactly on it, and every
    plastic selection strictly beyond it. The enclosure carries the sign
    only - no magnitude is observed or compared - and the hardening segment
    stays the exact predecessor segment.
    """
    with localcontext() as context:
        context.prec = 80
        p0 = history_row([float(v) for v in row_before])[1]
        x, _, _ = fibre.curve._segments()
        segment = sum(k <= p0 for k in x)-1
    branch = replayed.branch
    signed = {'ELASTIC': (-1., -1.), 'YIELD_BOUNDARY': (0., 0.)}.get(
        branch, (1., 1.))
    return _Selection(branch, signed, segment, branch == 'HARDENING_KNOT')


def _exact_selection(fibre, row_before, row_after, recovery, strain, strain_low):
    """Exact re-evaluation of the unchanged law from the accepted predecessor.

    Reconstructs the fibre strain exactly as PhysicalFibreSection.response
    does - Decimal prec 80, the section's own fibre map and summation order -
    from the recorded station strain pair, never from the recorded recovery
    pair, which compresses the 80-digit polynomial to two binary64 floats.
    Raises unless the recorded recovery is precisely the law's output for the
    reconstructed predecessor (z0, p0) at that strain, and the captured
    accepted history is exactly the predecessor plus the recorded increment.
    """
    with localcontext() as context:
        context.prec = 80
        z0, p0 = history_row([float(v) for v in row_before])
        young = D.from_float(fibre.young)
        e = [D.from_float(h)+D.from_float(l) for h, l in zip(strain, strain_low)]
        b = PhysicalFibreSection._map(fibre)
        ef = sum((a*x for a, x in zip(b, e)), D(0))
        assert _paired(ef) == tuple(recovery.total_strain), \
            'recorded fibre strain is not the station strain reconstruction'
        delta, modulus, branch = fibre.curve._return(young, ef, z0, p0)
        # The record must be exactly the law's own selection, not close to it;
        # a mismatched label is not forgiven by an equal elastic output.
        assert branch == recovery.branch, \
            'recorded branch is not the law selection'
        assert _paired(delta) == tuple(recovery.plastic_increment), \
            'recorded plastic increment is not the law selection'
        assert _paired(modulus) == tuple(recovery.algorithmic_modulus), \
            'recorded algorithmic modulus is not the law selection'
        # Accepted history continuity: exactly the predecessor plus the
        # recorded increment, as the product itself encodes the rows.
        assert _paired(z0+delta) == (row_after[0], row_after[1]) \
            and _paired(p0+abs(delta)) == (row_after[2], row_after[3]), \
            'accepted history is not the predecessor plus the recorded increment'
        ee = ef-(z0+delta)
        assert _paired(ee) == tuple(recovery.elastic_strain), \
            'recorded elastic strain is not the law selection'
        assert _paired(young*ee) == tuple(recovery.stress), \
            'recorded fibre stress is not the law selection'
        x, y, slopes = fibre.curve._segments()
        first = sum(k <= p0 for k in x)-1
        drive = young*(ef-z0)
        radius = y[first]+slopes[first]*(p0-x[first])
        knot = delta != 0 and any(p0+abs(delta) == k for k in x[1:])
        assert knot == (branch == 'HARDENING_KNOT'), \
            'hardening knot label does not match an exact knot landing'
        signed = abs(drive)-radius
        return _Selection(branch, (signed, signed), first, knot)


def _pair_decision(first, second):
    """'equal' or 'straddle'; any other pairing is refused.

    A straddle is admitted only by the paired constitutive evidence itself:
    the selected branch pair must be an elastic-side selection against a
    plastic-side selection, or a valid ELASTIC vs YIELD_BOUNDARY pair - no
    hardening-knot exemption, a HARDENING_KNOT selection never pairs with a
    different branch - and the two sides' own signed yield functions (the
    law's |drive|-radius at each side's reconstructed state and predecessor,
    exact points on the exact route and branch-derived sign enclosures on
    the cell-replay route) must straddle zero - there must exist consistent
    true values with one side at or below zero and the other at or above
    zero - and both predecessors must sit on the same hardening segment, so
    the yield surface lies between the physically agreeing states by
    continuity. No surface-distance band is consulted; same-sided pairs
    are refused here and physically distant pairs are refused by the
    existing 1e-9 agreement.
    """
    if first.branch == second.branch:
        return 'equal'
    assert ((first.branch in ELASTIC_SIDE and second.branch in PLASTIC_SIDE)
            or (first.branch in PLASTIC_SIDE and second.branch in ELASTIC_SIDE)
            or {first.branch, second.branch} == {'ELASTIC', 'YIELD_BOUNDARY'}), \
        'paired branch selections cannot straddle a yield surface'
    straddle = ((first.signed[0] <= 0 and second.signed[1] >= 0)
                or (second.signed[0] <= 0 and first.signed[1] >= 0))
    assert straddle, 'paired signed yield functions do not straddle zero'
    assert first.segment == second.segment, \
        'paired predecessors are not on the same hardening segment'
    return 'straddle'


def _station_plan(sections, operators, capture, previous):
    """Bind each recorded station to its captured accepted/predecessor rows."""
    if previous is None:
        raise ValueError('accepted predecessor capture is required, not declared')
    records = capture['result'].beam_fibre_material
    rows_after = np.asarray(capture['beam']['accepted_fibres'], dtype=float)
    rows_before = np.asarray(previous['beam']['accepted_fibres'], dtype=float)
    if (rows_after.ndim != 2 or rows_after.shape[1] != 4
            or rows_before.shape != rows_after.shape):
        raise ValueError('complete paired accepted fibre histories required')
    plan = []; cursor = 0
    for record in records:
        section = sections.get(record.element_id)
        if section is None or type(section) is not PhysicalFibreSection:
            raise ValueError('recorded station has no declared physical fibre section')
        count = len(record.fibres)
        if count != len(section.fibres) or cursor+count > len(rows_after):
            raise ValueError('recorded fibre inventory does not match the captured history')
        for fibre, recovery in zip(section.fibres, record.fibres):
            if recovery.fibre_id != fibre.fibre_id:
                raise ValueError('recorded fibre order does not match the declared section')
        plan.append(_StationPlan(record, section,
            rows_after[cursor:cursor+count], rows_before[cursor:cursor+count],
            operators.get(record.element_id) if operators else None))
        cursor += count
    if cursor != len(rows_after):
        raise ValueError('unassigned accepted fibre history rows')
    return plan


def cyclic_fibre_boundary_hook(sections, operators, a, b, a_previous, b_previous):
    """Opt-in comparator for paired beam_fibre_material records across charts.

    a, b are the compared captures and a_previous, b_previous each branch's
    own accepted predecessor capture. operators (cell_operators(model))
    supplies each fibre beam's actual retained-fibre operator; a record
    whose exact station-pair reconstruction fails is then verified by exact
    local cell replay from its capture's own retained resultants and
    accepted predecessor. Returns the hook for _same's explicit
    fibre_material route; every other caller of _same keeps the default.
    """
    operators = {} if operators is None else operators
    plan_a = _station_plan(sections, operators, a, a_previous)
    plan_b = _station_plan(sections, operators, b, b_previous)
    replayed = {}
    def beam_rows(side, plan, capture, entry):
        element_id = entry.record.element_id
        if (side, element_id) not in replayed:
            group = [e for e in plan if e.record.element_id == element_id]
            replayed[(side, element_id)] = _replay_beam(group, capture,
                entry.operator)
        return replayed[(side, element_id)]
    def verified(side, plan, capture, entry, recovery, index):
        # Exact route first: a record whose station pairs reproduce the
        # law's fibre strain must be precisely the law's output. Only a
        # law-mismatch AssertionError falls back to the capture's own cell
        # replay; malformed histories raise ValueError and stay fatal.
        fibre = entry.section.fibres[index]
        try:
            return _exact_selection(fibre, entry.rows_before[index],
                entry.rows_after[index], recovery, entry.record.strain,
                entry.record.strain_low)
        except AssertionError:
            if entry.operator is None:
                raise
            rows = beam_rows(side, plan, capture, entry)
            return _replay_selection(fibre, entry.rows_before[index],
                rows[(entry.record.cell, entry.record.station)]['fibres'][index])
    def hook(xx, yy, definition):
        index = next((i for i, entry in enumerate(plan_a) if entry.record is xx), None)
        assert index is not None, 'compared record does not belong to the capture'
        entry_a, entry_b = plan_a[index], plan_b[index]
        assert entry_b.record is yy, 'compared records are not the paired stations'
        assert entry_a.section is entry_b.section, 'paired stations declare different sections'
        assert len(xx.fibres) == len(yy.fibres) == len(entry_a.section.fibres)
        fibres_x, fibres_y = [], []
        for j, fibre in enumerate(entry_a.section.fibres):
            rx, ry = xx.fibres[j], yy.fibres[j]
            assert rx.fibre_id == ry.fibre_id == fibre.fibre_id, 'fibre origin mismatch'
            selected_x = verified('a', plan_a, a, entry_a, rx, j)
            selected_y = verified('b', plan_b, b, entry_b, ry, j)
            decision = _pair_decision(selected_x, selected_y)
            if decision == 'equal':
                fibres_x.append(rx); fibres_y.append(ry)
                continue
            # The derivative does not exist at the surface: cross-record
            # equality of the discontinuous leaves is not asserted. Each side
            # was verified above against its own law and predecessor; the
            # elastic-side selection stands in for both on the compared
            # copies so every continuous leaf keeps its existing 1e-9 scale.
            elastic = rx if selected_x.branch in ELASTIC_SIDE else ry
            fibres_x.append(replace(rx, algorithmic_modulus=elastic.algorithmic_modulus,
                branch=elastic.branch))
            fibres_y.append(replace(ry, algorithmic_modulus=elastic.algorithmic_modulus,
                branch=elastic.branch))
        compared_x = xx if all(a is b for a, b in zip(fibres_x, xx.fibres)) \
            else replace(xx, fibres=tuple(fibres_x))
        compared_y = yy if all(a is b for a, b in zip(fibres_y, yy.fibres)) \
            else replace(yy, fibres=tuple(fibres_y))
        _material_close(compared_x, compared_y, definition)
    return hook


# ---------------------------------------------------------------------------
# Controlled fixtures: genuine records from the actual law, no solver runs.
# ---------------------------------------------------------------------------

def _section(curve=None):
    fibre = Fibre('f0', 0., 0., 4e-4, YOUNG,
        curve if curve is not None else FlowCurve((0.,), (YIELD_STRESS,), HARDENING))
    return PhysicalFibreSection((fibre,), np.eye(6)*1e6)


def _dyadic_section(knot=None):
    """Exact-arithmetic control law: E=2^40, yield 2^32, flat first segment.

    Dyadic values make Decimal return-map landings exactly reproducible, so
    genuine YIELD_BOUNDARY and HARDENING_KNOT selections are constructible
    offline without any solver or tolerance.
    """
    young, yield_stress, tail = float(2**40), float(2**32), float(2**40)
    curve = (FlowCurve((0., float(knot)), (yield_stress, yield_stress), tail)
             if knot is not None else FlowCurve((0.,), (yield_stress,), tail))
    return PhysicalFibreSection((Fibre('f0', 0., 0., 4e-4, young, curve),),
        np.eye(6)*1e6)


def _plastic_origin(section):
    """Genuine accepted plastic history from one actual law response."""
    response = section.response(np.array((2e-3, 0., 0., 0., 0., 0.)),
        origin=section.virgin())
    assert response.fibres[0].branch == 'PLASTIC_POSITIVE'
    return response.history


def _boundary(section, origin):
    """Decimal reload-yield boundary (z0 + radius/E) of the accepted origin."""
    with localcontext() as context:
        context.prec = 80
        z0, p0 = history_row([float(v) for v in origin.rows[0]])
        x, y, slopes = section.fibres[0].curve._segments()
        first = sum(k <= p0 for k in x)-1
        radius = y[first]+slopes[first]*(p0-x[first])
        return z0, p0, radius, z0+radius/D.from_float(section.fibres[0].young)


def _response_at(section, origin, ef_decimal, offset):
    strain = np.array((float(ef_decimal+D.from_float(offset)), 0., 0., 0., 0., 0.))
    return section.response(strain, origin=origin)


def _record(response, element_id=2):
    return BeamFibreStationRecovery(element_id, 0, 0, 0., 1.,
        np.array(response.strain), np.array(response.strain_low),
        np.array(response.resultants), np.array(response.resultants_low),
        np.eye(3), np.eye(3), response.fibres)


def _stored(response):
    """The response's stored energy exactly as its binary64 pair encodes it.

    The pair is the product's own encoding of the 80-digit value; a float
    sum of it at a 2e6 magnitude would only measure representation
    rounding, not the energy.
    """
    high, low = response.stored_energy
    return D.from_float(high)+D.from_float(low)


def _capture(record, rows):
    return {'beam': {'accepted_fibres': np.array(rows, dtype=float).reshape(-1, 4)},
            'result': SimpleNamespace(beam_fibre_material=(record,))}


def _predecessor_capture(record, rows):
    return {'beam': {'accepted_fibres': np.array(rows, dtype=float)},
            'result': SimpleNamespace(beam_fibre_material=(record,))}


def _straddle_pair(section=None, offset=1e-14):
    """Genuine elastic-side/plastic-side records straddling the reload surface."""
    section = section if section is not None else _section()
    origin = _plastic_origin(section)
    _, _, _, ef = _boundary(section, origin)
    elastic = _response_at(section, origin, ef, -offset)
    plastic = _response_at(section, origin, ef, offset)
    assert elastic.fibres[0].branch == 'ELASTIC'
    assert plastic.fibres[0].branch == 'PLASTIC_POSITIVE'
    sections = {2: section}
    predecessor = _predecessor_capture(_record(elastic), origin.rows)
    pair_a = (_capture(_record(elastic), elastic.history.rows), predecessor)
    pair_b = (_capture(_record(plastic), plastic.history.rows), predecessor)
    return sections, pair_a, pair_b, DEFINITION


def _run_hook(sections, pair_a, pair_b, definition=DEFINITION, operators=None):
    hook = cyclic_fibre_boundary_hook(sections, operators, pair_a[0], pair_b[0],
        pair_a[1], pair_b[1])
    hook(pair_a[0]['result'].beam_fibre_material[0],
         pair_b[0]['result'].beam_fibre_material[0], definition)


def _cell_recovery_control():
    """Genuine local cell-recovery capture from the actual chart definition.

    The real chart definition's own physical-fibre member (beams index 1,
    element id 2) supplies the actual RetainedFibreOperator and cell; the
    operator responds once through recover_material - the exact product
    path that produces beam_fibre_material and beam_fibre_energy records,
    one coupled cell response with include_energy - to the prespecified
    retained resultants p18 (constant axial force [3e6, 0, 0, 3e6, 0, 0]
    on the first six force components, all twelve moment components zero)
    from the cell's actual virgin predecessor, with identity cell frames.
    The capture is built from that response alone, exactly as the driver's
    _fibre_recovery builds its records: every station row becomes a
    BeamFibreStationRecovery with the member's element id, one
    BeamFibreEnergyRecovery carries the response's energy fields, the
    accepted rows are the flattened response history and the predecessor's
    rows the flattened virgin history, and the retained resultants/cell
    frames arrays carry the member at its definition index (the generalized
    member's row stays the driver's zero dummy). No owner is constructed
    and no equilibrium is solved - local constitutive response only.
    """
    from test_stiffened_shell_native_plastic_chart import definition
    model = definition()
    element, _ = model.beams[1].instantiate()
    assert element.element_id == 2 and type(element.section) is PhysicalFibreSection
    operator = element.operator
    section = element.section
    origin = operator.cell.virgin()
    resultants = np.zeros(18)
    resultants[:6] = (3e6, 0., 0., 3e6, 0., 0.)
    frames = np.tile(np.eye(3), (2, 1, 1))
    response = operator.recover_material(frames, resultants, origin=origin)
    records = []
    for row in response.stations:
        records.append(BeamFibreStationRecovery(element.element_id, row['cell'],
            row['station'], row['xi'], row['measure'],
            *(np.asarray(row[name], dtype=float) for name in (
                'strain', 'strain_low', 'resultants', 'resultants_low',
                'reference_frame', 'current_frame')), row['fibres']))
    energy = BeamFibreEnergyRecovery(element.element_id, response.stored_energy,
        response.previous_hardening_energy, response.incremental_dissipation,
        response.material_potential, response.energy_recovery_station_updates)
    accepted = np.array([row for station in response.history.stations
        for row in station.rows], dtype=float)
    virgin = np.array([row for station in origin.stations
        for row in station.rows], dtype=float)
    cell_resultants = np.zeros((len(model.beams), 18))
    cell_resultants[1] = resultants
    cell_frames = np.zeros((len(model.beams), 2, 3, 3))
    cell_frames[1] = frames
    capture = {'beam': {'accepted_fibres': accepted},
        'result': SimpleNamespace(beam_fibre_material=tuple(records),
            beam_fibre_energy=(energy,), cell_resultants=cell_resultants,
            cell_frames=cell_frames)}
    previous = {'beam': {'accepted_fibres': virgin},
        'result': SimpleNamespace(beam_fibre_material=(), beam_fibre_energy=(),
            cell_resultants=cell_resultants, cell_frames=cell_frames)}
    return {element.element_id: section}, \
        {element.element_id: _CellOperator(operator, 1)}, capture, previous


def test_flow_curve_yield_boundary_tangent_counterexample():
    """The actual FlowCurve law disproves uniform tangent equality at reload.

    From one accepted predecessor, the two one-sided trials at the reload
    yield boundary hold committed stress continuous at the physical 1e-9
    scale while the selected tangent jumps by exactly E-E*H/(E+H). This is
    the recorded Linux35 position-4.0 mechanism reproduced from the law
    itself. The stored-energy gap between the one-sided responses is proven
    continuous by the law's own potential/work bound - proportional to the
    offset and vanishing with it - and NOT by the coupled/chart 1e-9 J
    energy criterion: this controlled section energy is per length, its
    scale is set by the arbitrary declared background factor (F^T F adds
    1e12 axial stiffness, about 2e6 per-length stored energy here), the
    original beam energies are integrated joules, and a finite chosen
    offset meeting 1e-9 J would qualify nothing. The position-2.5 stored
    energy failure stays exactly as recorded; this counterexample does not
    claim it fixed.
    """
    section = _section()
    origin = _plastic_origin(section)
    z0, p0, radius, ef = _boundary(section, origin)
    fibre = section.fibres[0]
    plastic_modulus = YOUNG*HARDENING/(YOUNG+HARDENING)
    with localcontext() as context:
        context.prec = 80
        # Declared work-bound constants of this controlled fixture: the
        # fibre sits at y=z=0 (unit axial map, the station strain IS the
        # fibre strain) and only strain component 0 is excited, so the
        # background axial stiffness is the declared factor's own first
        # column inner product and the work bound is one-dimensional.
        axial = sum(D.from_float(v)**2 for v in section.background_factor[:, 0])
        area, young = D.from_float(fibre.area), D.from_float(fibre.young)
        flow = D.from_float(fibre.curve.flow_stress[0])
        previous_gap = None
        for offset in (1e-12, 1e-14, 1e-16):
            elastic = _response_at(section, origin, ef, -offset)
            plastic = _response_at(section, origin, ef, offset)
            inside, outside = elastic.fibres[0], plastic.fibres[0]
            assert inside.branch == 'ELASTIC' and inside.algorithmic_modulus[0] == YOUNG
            assert inside.plastic_increment == (0., 0.)
            assert outside.branch == 'PLASTIC_POSITIVE'
            assert outside.algorithmic_modulus[0] == plastic_modulus
            # The tangent selection jump is exactly the two one-sided tangents'
            # difference and does not shrink with the state gap.
            assert inside.algorithmic_modulus[0]-outside.algorithmic_modulus[0] \
                == YOUNG-plastic_modulus
            # Committed stress stays continuous at the physical 1e-9 scale...
            np.testing.assert_allclose(sum(inside.stress)/YOUNG,
                sum(outside.stress)/YOUNG, rtol=0., atol=1e-9)
            # ...while the state gap shrinks linearly toward zero.
            assert abs(sum(inside.stress)-sum(outside.stress)) <= 2.*YOUNG*offset
            assert elastic.derivative_kind == plastic.derivative_kind \
                == 'CLASSICAL_SMOOTH_BRANCH'
            # Stored-energy gap, exactly as the response encodes it.
            gap = abs(_stored(elastic)-_stored(plastic))
            # The law's potential/work bound, proportional to the offset:
            # the gap is at most the work of the peak trial resultant over
            # the strain gap plus the plastic commit work at the boundary
            # flow stress. Every term is a declared law quantity or the
            # law's own recorded output; nothing is fitted to the gap.
            e_in = D.from_float(elastic.strain[0])
            e_out = D.from_float(plastic.strain[0])
            commit = abs(D.from_float(outside.plastic_increment[0])
                + D.from_float(outside.plastic_increment[1]))
            peak = axial*abs(e_out)+area*young*abs(e_out-z0)
            work = abs(e_out-e_in)*peak+area*commit*(flow+young*abs(e_out-ef))
            assert gap <= work, 'stored-energy gap exceeds the work bound'
            if previous_gap is not None:
                assert gap < previous_gap
            previous_gap = gap


def test_true_elastic_plastic_straddle_admitted():
    sections, pair_a, pair_b, definition = _straddle_pair()
    _run_hook(sections, pair_a, pair_b, definition)


def test_lost_tail_cell_recovery_record_control():
    """A genuine local cell-recovery capture is admitted; tampering is refused.

    The capture comes from the actual chart definition's own retained-fibre
    operator (no owner, no equilibrium solve). The comparator's local cell
    replay re-runs the same deterministic recover_material from the
    capture's own retained resultants and accepted predecessor rows and
    must reproduce every recorded station field, fibre record, accepted
    history row and energy record exactly. Lost-tail necessity is
    measured, not presumed: if the genuine cell's published station pairs
    lose the 80-digit tail (at least one exact-route mismatch), the exact
    route alone cannot verify the capture and only the replay admits it;
    if the prespecified force reproduces exactly, nothing is forged and
    the lost-tail obstruction is proven on the actual preserved example
    data by the saved script. Tampered stress, modulus and accepted
    predecessor are refused against the replayed cell output.
    """
    sections, operators, capture, previous = _cell_recovery_control()
    genuine = (capture, previous)
    plan = _station_plan(sections, operators, capture, previous)
    # The local replay itself must reproduce the genuine capture exactly.
    for element_id in {entry.record.element_id for entry in plan}:
        group = [entry for entry in plan if entry.record.element_id == element_id]
        _replay_beam(group, capture, group[0].operator)
    exact_failures = 0
    for entry in plan:
        for index, fibre in enumerate(entry.section.fibres):
            try:
                _exact_selection(fibre, entry.rows_before[index],
                    entry.rows_after[index], entry.record.fibres[index],
                    entry.record.strain, entry.record.strain_low)
            except AssertionError:
                exact_failures += 1
    if exact_failures:
        # Lost-tail necessity established by the genuine cell itself: the
        # exact station-pair route alone cannot verify these records.
        with pytest.raises(AssertionError):
            _run_hook(sections, genuine, genuine, DEFINITION)
    _run_hook(sections, genuine, genuine, DEFINITION, operators=operators)
    records = capture['result'].beam_fibre_material
    recovery = records[0].fibres[0]

    def forged_capture(forged_record):
        replaced = tuple(forged_record if index == 0 else record
            for index, record in enumerate(records))
        return ({'beam': capture['beam'], 'result': SimpleNamespace(
            **{**capture['result'].__dict__, 'beam_fibre_material': replaced})},
            previous)

    for forged in (
        replace(records[0], fibres=(replace(recovery,
            stress=(recovery.stress[0]*(1.+1e-12), recovery.stress[1])), *records[0].fibres[1:])),
        replace(records[0], fibres=(replace(recovery,
            algorithmic_modulus=(recovery.algorithmic_modulus[0]*(1.+1e-12),
                recovery.algorithmic_modulus[1])), *records[0].fibres[1:])),
    ):
        with pytest.raises(AssertionError):
            _run_hook(sections, genuine, forged_capture(forged),
                DEFINITION, operators=operators)
    tampered = np.array(previous['beam']['accepted_fibres'], dtype=float)
    tampered[0, 0] += 1e-12
    tampered[0, 2] += 1e-12  # admissible wrong predecessor; p >= abs(z)
    forged_previous = {'beam': {'accepted_fibres': tampered},
        'result': previous['result']}
    with pytest.raises(AssertionError):
        _run_hook(sections, genuine, (capture, forged_previous),
            DEFINITION, operators=operators)


def test_smooth_matching_pairs_still_require_equality():
    sections, _, _, definition = _straddle_pair()
    section = sections[2]
    origin = _plastic_origin(section)
    _, _, _, ef = _boundary(section, origin)
    smooth = _response_at(section, origin, ef, 1e-9)
    assert smooth.fibres[0].branch == 'PLASTIC_POSITIVE'
    predecessor = _predecessor_capture(_record(smooth), origin.rows)
    smooth_pair = (_capture(_record(smooth), smooth.history.rows), predecessor)
    _run_hook(sections, smooth_pair, smooth_pair, definition)
    # A wrong modulus on a smooth same-branch pair is refused even though its
    # magnitude is close: the record must be exactly the law selection.
    wrong = replace(_record(smooth), fibres=(replace(smooth.fibres[0],
        algorithmic_modulus=(smooth.fibres[0].algorithmic_modulus[0]*(1.+1e-12),
            smooth.fibres[0].algorithmic_modulus[1])),))
    forged = (_capture(wrong, smooth.history.rows), predecessor)
    with pytest.raises(AssertionError):
        _run_hook(sections, smooth_pair, forged, definition)
    with pytest.raises(AssertionError):
        _run_hook(sections, forged, smooth_pair, definition)


def test_wrong_modulus_at_straddle_refused():
    sections, pair_a, pair_b, definition = _straddle_pair()
    record = pair_b[0]['result'].beam_fibre_material[0]
    recovery = record.fibres[0]
    for wrong in (105e9*(1.+1e-12), 157.5e9, 210e9):
        if wrong == recovery.algorithmic_modulus[0]:
            continue
        forged = replace(record, fibres=(replace(recovery,
            algorithmic_modulus=(wrong, recovery.algorithmic_modulus[1])),))
        capture = _capture(forged, pair_b[0]['beam']['accepted_fibres'])
        with pytest.raises(AssertionError):
            _run_hook(sections, pair_a, (capture, pair_b[1]), definition)
    # A wrong recorded stress is refused the same way: against the actual law,
    # never by magnitude or by the elastic branch's equal output.
    forged = replace(record, fibres=(replace(recovery,
        stress=(recovery.stress[0]*(1.+1e-12), recovery.stress[1])),))
    capture = _capture(forged, pair_b[0]['beam']['accepted_fibres'])
    with pytest.raises(AssertionError):
        _run_hook(sections, pair_a, (capture, pair_b[1]), definition)


def test_invalid_predecessor_refused():
    sections, pair_a, pair_b, definition = _straddle_pair()
    origin_rows = pair_b[1]['beam']['accepted_fibres']
    with localcontext() as context:
        context.prec = 80
        z0, p0 = history_row([float(v) for v in origin_rows[0]])
        # A normalized but wrong predecessor: accepted plastic state shifted.
        high, low = pair([z0, p0+D.from_float(1e-9)])
        shifted = np.array((high[0], low[0], high[1], low[1])).reshape(1, 4)
    record = pair_b[0]['result'].beam_fibre_material[0]
    wrong = _predecessor_capture(record, shifted)
    with pytest.raises(AssertionError):
        _run_hook(sections, pair_a, (pair_b[0], wrong), definition)
    # The plastic side's own accepted history is not its predecessor either.
    accepted = _predecessor_capture(record, pair_b[0]['beam']['accepted_fibres'])
    with pytest.raises(AssertionError):
        _run_hook(sections, pair_a, (pair_b[0], accepted), definition)


def test_incompatible_physical_state_refused():
    sections, pair_a, pair_b, definition = _straddle_pair()
    record = pair_b[0]['result'].beam_fibre_material[0]
    # A station strain component outside the fibre map: the fibre recovery
    # still reproduces exactly, and the pair is refused only by the existing
    # 1e-9 physical strain agreement - no new criterion.
    strain = np.array(record.strain, dtype=float)
    strain[1] += 1e-6
    forged = replace(record, strain=strain)
    capture = _capture(forged, pair_b[0]['beam']['accepted_fibres'])
    with pytest.raises(AssertionError):
        _run_hook(sections, pair_a, (capture, pair_b[1]), definition)
    # A station strain component inside the fibre map: the recorded fibre
    # strain no longer reproduces from the station strain (recovery
    # inconsistency) and the record is refused against the law.
    strain = np.array(record.strain, dtype=float)
    strain[0] += 1e-6
    forged = replace(record, strain=strain)
    capture = _capture(forged, pair_b[0]['beam']['accepted_fibres'])
    with pytest.raises(AssertionError):
        _run_hook(sections, pair_a, (capture, pair_b[1]), definition)


def test_branch_mismatch_away_from_yield_refused():
    sections, pair_a, _, definition = _straddle_pair()
    section = sections[2]
    origin = _plastic_origin(section)
    _, _, _, ef = _boundary(section, origin)
    far = _response_at(section, origin, ef, 1e-4)
    assert far.fibres[0].branch == 'PLASTIC_POSITIVE'
    predecessor = _predecessor_capture(_record(far), origin.rows)
    far_pair = (_capture(_record(far), far.history.rows), predecessor)
    # The far plastic trial straddles the surface by signed functions, but
    # the pair is refused by the existing 1e-9 physical agreement - never by
    # an arbitrary distance from the surface.
    with pytest.raises(AssertionError):
        _run_hook(sections, pair_a, far_pair, definition)
    # Both plastic with opposite drive signs: same-sided signed yield
    # functions, not a straddle.
    negative = section.response(
        np.array((float(-ef-D.from_float(1e-14)), 0., 0., 0., 0., 0.)), origin=origin)
    assert negative.fibres[0].branch == 'PLASTIC_NEGATIVE'
    negative_pair = (_capture(_record(negative), negative.history.rows),
        _predecessor_capture(_record(negative), origin.rows))
    with pytest.raises(AssertionError):
        _run_hook(sections, far_pair, negative_pair, definition)
    with pytest.raises(AssertionError):
        _run_hook(sections, pair_a, negative_pair, definition)


def test_invalid_history_refused():
    sections, pair_a, pair_b, definition = _straddle_pair()
    record = pair_b[0]['result'].beam_fibre_material[0]
    rows = pair_b[1]['beam']['accepted_fibres']
    def predecessor_with(new_rows):
        return _predecessor_capture(record, new_rows)
    with pytest.raises(ValueError):  # malformed row width
        _run_hook(sections, pair_a,
            (pair_b[0], predecessor_with([rows[0][:3]])), definition)
    with pytest.raises(ValueError):  # unnormalized pairing
        _run_hook(sections, pair_a,
            (pair_b[0], predecessor_with([(rows[0][1], rows[0][0], rows[0][2], rows[0][3])])),
            definition)
    with pytest.raises(ValueError):  # inadmissible p < |z|
        _run_hook(sections, pair_a,
            (pair_b[0], predecessor_with(
                [(rows[0][0], rows[0][1], abs(rows[0][0])/4., 0.)])), definition)


def test_hardening_knot_not_blanket_excused():
    # A genuine exact knot landing is NOT a straddle: no hardening-knot
    # exemption, a HARDENING_KNOT selection never pairs with a different
    # branch. This actual consistent law pair - real records, opposing
    # signed yield functions, same segment, physically agreeing - is
    # refused on the branch pair alone, in both orders.
    near_knot = _dyadic_section(knot=2**-48)
    origin = near_knot.virgin()
    elastic = near_knot.response(
        np.array((float(2**-8-2**-48), 0., 0., 0., 0., 0.)), origin=origin)
    landing = near_knot.response(
        np.array((float(2**-8+2**-48), 0., 0., 0., 0., 0.)), origin=origin)
    assert elastic.fibres[0].branch == 'ELASTIC'
    assert landing.fibres[0].branch == 'HARDENING_KNOT'
    sections = {2: near_knot}
    predecessor = _predecessor_capture(_record(elastic), origin.rows)
    elastic_pair = (_capture(_record(elastic), elastic.history.rows), predecessor)
    landing_pair = (_capture(_record(landing), landing.history.rows), predecessor)
    with pytest.raises(AssertionError):
        _run_hook(sections, elastic_pair, landing_pair, DEFINITION)
    with pytest.raises(AssertionError):
        _run_hook(sections, landing_pair, elastic_pair, DEFINITION)
    # Same-branch knot matches keep the existing comparison: genuine knot
    # landings from the same predecessor compare through the unchanged
    # equal route and the 1e-9 physical comparator.
    _run_hook(sections, landing_pair, landing_pair, DEFINITION)
    # A genuine knot landing far from the paired elastic state is refused
    # (here already by the branch pair - no knot exemption - never by a
    # distance band).
    far_knot = _dyadic_section(knot=2**-10)
    far_origin = far_knot.virgin()
    far_elastic = far_knot.response(
        np.array((float(2**-8-2**-48), 0., 0., 0., 0., 0.)), origin=far_origin)
    far_jump = far_knot.response(
        np.array((float(2**-8+2**-10), 0., 0., 0., 0., 0.)), origin=far_origin)
    assert far_elastic.fibres[0].branch == 'ELASTIC'
    assert far_jump.fibres[0].branch == 'HARDENING_KNOT'
    far_sections = {2: far_knot}
    far_predecessor = _predecessor_capture(_record(far_elastic), far_origin.rows)
    far_elastic_pair = (_capture(_record(far_elastic), far_elastic.history.rows),
        far_predecessor)
    far_jump_pair = (_capture(_record(far_jump), far_jump.history.rows), far_predecessor)
    with pytest.raises(AssertionError):
        _run_hook(far_sections, far_elastic_pair, far_jump_pair, DEFINITION)
    # A fabricated knot label on a smooth plastic return is refused by the law.
    smooth = far_knot.response(
        np.array((float(2**-8+2**-11), 0., 0., 0., 0., 0.)), origin=far_origin)
    assert smooth.fibres[0].branch == 'PLASTIC_POSITIVE'
    forged = replace(_record(smooth), fibres=(replace(smooth.fibres[0],
        branch='HARDENING_KNOT'),))
    forged_pair = (_capture(forged, smooth.history.rows), far_predecessor)
    with pytest.raises(AssertionError):
        _run_hook(far_sections, far_elastic_pair, forged_pair, DEFINITION)


def test_surface_label_coin_admitted_only_at_the_surface():
    # Exact dyadic boundary: the law itself selects YIELD_BOUNDARY and flags
    # the derivative as a semismooth branch selection, not a smooth branch.
    section = _dyadic_section()
    origin = section.virgin()
    exact = section.response(
        np.array((float(2**-8), 0., 0., 0., 0., 0.)), origin=origin)
    near = section.response(
        np.array((float(2**-8-2**-48), 0., 0., 0., 0., 0.)), origin=origin)
    deep = section.response(
        np.array((float(2**-8-2**-10), 0., 0., 0., 0., 0.)), origin=origin)
    assert exact.fibres[0].branch == 'YIELD_BOUNDARY'
    assert exact.derivative_kind == 'SEMISMOOTH_BRANCH_SELECTION'
    assert near.fibres[0].branch == deep.fibres[0].branch == 'ELASTIC'
    sections = {2: section}
    predecessor = _predecessor_capture(_record(exact), origin.rows)
    exact_pair = (_capture(_record(exact), exact.history.rows), predecessor)
    near_pair = (_capture(_record(near), near.history.rows), predecessor)
    # Both records are the law's actual selections; the pair is admitted as a
    # straddle with an exact zero on one side, not by a label exception.
    _run_hook(sections, exact_pair, near_pair, DEFINITION)
    # A YIELD_BOUNDARY label claimed away from the surface is a forged label:
    # the law's actual selection there is ELASTIC, and the record is refused.
    forged = replace(_record(deep), fibres=(replace(deep.fibres[0],
        branch='YIELD_BOUNDARY'),))
    deep_pair = (_capture(forged, deep.history.rows), predecessor)
    with pytest.raises(AssertionError):
        _run_hook(sections, near_pair, deep_pair, DEFINITION)


def test_hook_requires_actual_predecessor_captures():
    sections, pair_a, pair_b, _ = _straddle_pair()
    with pytest.raises(ValueError):
        cyclic_fibre_boundary_hook(sections, None, pair_a[0], pair_b[0], None, pair_b[1])
    with pytest.raises(ValueError):
        cyclic_fibre_boundary_hook(sections, None, pair_a[0], pair_b[0], pair_a[1], None)
