"""Private heterogeneous GE-Beam3 station transaction foundation.

This module owns no graph equilibrium and exposes no public formulation route.
It composes three already-authored pure constitutive laws behind one atomic
accepted-state pointer.  G4 graph, solver and restart qualification remain
separate.
"""
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock

import numpy as np

from ._ge_beam3_fibre_section import (
    FibreHistory,
    PhysicalFibreSection,
    canonical,
)
from ._ge_beam3_g1_elastic import ElasticSection, owned
from ._ge_beam3_generalized_ellipsoid_section import (
    EllipsoidalGeneralizedSection,
    GeneralizedHistory,
)


POLICY = "GE_BEAM3_G4A_MIXED_STATION_TRANSACTION_V1"
STATE_SCHEMA = "GE_BEAM3_G4A_STATION_TRANSACTION_STATE_V1"
RESULTANT_ONLY = "RESULTANT_ONLY_NO_FIBRE_STRESSES"
PHYSICAL_FIBRES = "PHYSICAL_FIBRE_STRESSES_AVAILABLE"
_FAMILIES = ("EXACT_ELASTIC", "GENERALIZED_ELLIPSOID", "PHYSICAL_FIBRE")
_CAPABILITIES = (RESULTANT_ONLY, RESULTANT_ONLY, PHYSICAL_FIBRES)


def _sha(value):
    return sha256(canonical(value)).hexdigest()


def _pair(value):
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("normalized high/low scalar pair required")
    if any(type(item) is not float or not np.isfinite(item) for item in value):
        raise ValueError("finite binary64 scalar pair required")
    return value


def _law_descriptor(law):
    if type(law) is ElasticSection:
        return law.descriptor()
    if type(law) is EllipsoidalGeneralizedSection:
        law.guard()
        return law.descriptor()
    if type(law) is PhysicalFibreSection:
        return dict(policy="GE_BEAM3_PHYSICAL_FIBRE_INCREMENTAL_POTENTIAL_V1",
                    identity=law.identity, fibres=law.fibres,
                    background_factor=law.background_factor)
    raise ValueError("exact admitted G4a section law required")


def _law_identity(law):
    return law.identity


@dataclass(frozen=True)
class StationDefinition:
    station_id: str
    law: object

    def __post_init__(self):
        if (type(self.station_id) is not str or not self.station_id.isascii()
                or not 1 <= len(self.station_id) <= 128):
            raise ValueError("bounded ASCII station ID required")
        _law_descriptor(self.law)


@dataclass(frozen=True)
class StationResponse:
    station_id: str
    family: str
    capability: str
    law_identity: str
    origin: object
    history: object
    strain: np.ndarray
    strain_low: np.ndarray
    resultants: np.ndarray
    resultants_low: np.ndarray
    tangent: np.ndarray
    tangent_low: np.ndarray
    incremental_potential: tuple
    stored_energy: tuple
    dissipation: tuple
    fibres: tuple
    derivative_kind: str


@dataclass(frozen=True)
class AcceptedState:
    owner_identity: str
    epoch: int
    strains: tuple
    histories: tuple
    responses: tuple
    previous_sha256: object
    state_sha256: str

    def descriptor(self):
        return dict(schema=STATE_SCHEMA, policy=POLICY,
                    owner_identity=self.owner_identity, epoch=self.epoch,
                    strains=self.strains, histories=self.histories,
                    responses=self.responses, previous_sha256=self.previous_sha256,
                    state_sha256=self.state_sha256,
                    production_qualified=False)


@dataclass(frozen=True)
class Proposal:
    token: object
    owner_identity: str
    epoch: int
    strains: tuple
    origins: tuple
    responses: tuple
    proposal_sha256: str


@dataclass(frozen=True)
class Prepared:
    token: object
    proposal: Proposal
    prepared_sha256: str


class MixedStationTransaction:
    """One-writer atomic owner for the frozen elastic/generalized/fibre order."""

    def __init__(self, definitions):
        self._definitions = tuple(definitions)
        if len(self._definitions) != 3 or any(
                type(item) is not StationDefinition for item in self._definitions):
            raise ValueError("exact three-station G4a definition required")
        if len({item.station_id for item in self._definitions}) != 3:
            raise ValueError("distinct station IDs required")
        if tuple(type(item.law) for item in self._definitions) != (
                ElasticSection, EllipsoidalGeneralizedSection, PhysicalFibreSection):
            raise ValueError("elastic/generalized/fibre station order required")
        self._definition_body = tuple(dict(
            station_id=item.station_id, family=family, capability=capability,
            law=_law_descriptor(item.law), law_identity=_law_identity(item.law))
            for item, family, capability in zip(
                self._definitions, _FAMILIES, _CAPABILITIES))
        self.identity = _sha(dict(policy=POLICY, stations=self._definition_body))
        self._law_hashes = tuple(_sha(_law_descriptor(item.law))
                                 for item in self._definitions)
        self._lock = Lock()
        self._pending = None
        self._prepared = None
        zero = tuple(owned(np.zeros(6), (6,)) for _ in self._definitions)
        origins = self._virgin_histories()
        responses = self._evaluate(zero, origins)
        self._accepted = self._state(0, zero,
                                     tuple(row.history for row in responses),
                                     responses, None)

    @property
    def definitions(self):
        return self._definitions

    @property
    def accepted(self):
        self._guard()
        return self._accepted

    def _enter(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent G4a station writer")

    def _guard(self):
        if _sha(dict(policy=POLICY, stations=self._definition_body)) != self.identity:
            raise ValueError("G4a owner definition changed")
        for definition, expected in zip(self._definitions, self._law_hashes):
            if _sha(_law_descriptor(definition.law)) != expected:
                raise ValueError("G4a station law changed")
        state = self._accepted
        body = dict(owner_identity=state.owner_identity, epoch=state.epoch,
                    strains=state.strains, histories=state.histories,
                    responses=state.responses, previous_sha256=state.previous_sha256)
        if (state.owner_identity != self.identity or type(state.epoch) is not int
                or state.epoch < 0 or state.state_sha256 != _sha(body)):
            raise ValueError("G4a accepted state changed")

    def _virgin_histories(self):
        return ((), self._definitions[1].law.virgin(),
                self._definitions[2].law.virgin())

    def _response(self, index, strain, origin):
        definition = self._definitions[index]
        family, capability = _FAMILIES[index], _CAPABILITIES[index]
        strain = owned(strain, (6,))
        if index == 0:
            if type(origin) is not tuple or origin:
                raise ValueError("elastic history must be empty")
            raw = definition.law.response(strain, origin)
            return StationResponse(
                definition.station_id, family, capability,
                definition.law.identity, origin, (), strain,
                owned(np.zeros(6), (6,)), owned(raw["resultants"], (6,)),
                owned(np.zeros(6), (6,)), owned(raw["tangent"], (6, 6)),
                owned(np.zeros((6, 6)), (6, 6)),
                (float(raw["potential"]), 0.0),
                (float(raw["potential"]), 0.0), (0.0, 0.0), (),
                "LINEAR_EXACT")
        if index == 1:
            if type(origin) is not GeneralizedHistory:
                raise ValueError("owned generalized origin required")
            raw = definition.law.response(strain, origin=origin)
            return StationResponse(
                definition.station_id, family, capability, raw.section_identity,
                raw.origin, raw.history, strain, owned(np.zeros(6), (6,)),
                raw.resultants, raw.resultants_low, raw.tangent, raw.tangent_low,
                _pair(raw.incremental_potential), _pair(raw.stored_energy),
                _pair(raw.dissipation), (), raw.derivative_kind)
        if type(origin) is not FibreHistory:
            raise ValueError("owned physical-fibre origin required")
        raw = definition.law.response(strain, origin=origin)
        return StationResponse(
            definition.station_id, family, capability, raw.section_identity,
            raw.origin, raw.history, strain, owned(np.zeros(6), (6,)),
            raw.resultants, raw.resultants_low, raw.tangent, raw.tangent_low,
            _pair(raw.incremental_potential), _pair(raw.stored_energy),
            _pair(raw.dissipation), raw.fibres, raw.derivative_kind)

    def _evaluate(self, strains, origins, check=None):
        if type(strains) is not tuple or len(strains) != 3:
            raise ValueError("ordered three-station strain tuple required")
        if type(origins) is not tuple or len(origins) != 3:
            raise ValueError("ordered three-station origin tuple required")
        rows = []
        for index, (strain, origin) in enumerate(zip(strains, origins)):
            if check is not None:
                check("prepare", index)
            rows.append(self._response(index, strain, origin))
        return tuple(rows)

    def _state(self, epoch, strains, histories, responses, previous):
        body = dict(owner_identity=self.identity, epoch=epoch, strains=strains,
                    histories=histories, responses=responses,
                    previous_sha256=previous)
        return AcceptedState(self.identity, epoch, strains, histories, responses,
                             previous, _sha(body))

    def trial_all(self, strains, check=None):
        self._enter()
        try:
            self._guard()
            if self._pending is not None or self._prepared is not None:
                raise RuntimeError("pending G4a station transaction")
            if type(strains) is not tuple or len(strains) != 3:
                raise ValueError("ordered three-station strain tuple required")
            data = tuple(owned(value, (6,)) for value in strains)
            origins = self._accepted.histories
            responses = []
            for index, (strain, origin) in enumerate(zip(data, origins)):
                responses.append(self._response(index, strain, origin))
                if check is not None:
                    check("trial", index)
            body = dict(owner_identity=self.identity, epoch=self._accepted.epoch,
                        strains=data, origins=origins, responses=tuple(responses))
            proposal = Proposal(object(), self.identity, self._accepted.epoch,
                                data, origins, tuple(responses), _sha(body))
            self._pending = proposal
            return proposal
        finally:
            self._lock.release()

    def prepare_all(self, proposal, check=None):
        self._enter()
        try:
            self._guard()
            if self._pending is not proposal or self._prepared is not None:
                raise ValueError("foreign stale or already prepared G4a proposal")
            body = dict(owner_identity=proposal.owner_identity, epoch=proposal.epoch,
                        strains=proposal.strains, origins=proposal.origins,
                        responses=proposal.responses)
            if (proposal.owner_identity != self.identity
                    or proposal.epoch != self._accepted.epoch
                    or proposal.proposal_sha256 != _sha(body)
                    or canonical(proposal.origins) != canonical(self._accepted.histories)):
                raise ValueError("G4a proposal authority changed")
            rebuilt = self._evaluate(proposal.strains, proposal.origins, check)
            if canonical(rebuilt) != canonical(proposal.responses):
                raise ValueError("G4a proposal response changed")
            token = object()
            prepared_sha = _sha(dict(proposal_sha256=proposal.proposal_sha256,
                                     owner_identity=self.identity,
                                     epoch=self._accepted.epoch))
            prepared = Prepared(token, proposal, prepared_sha)
            self._prepared = prepared
            return prepared
        except BaseException:
            self._pending = None
            self._prepared = None
            raise
        finally:
            self._lock.release()

    def commit_all(self, prepared, check=None):
        self._enter()
        published = False
        try:
            self._guard()
            if self._prepared is not prepared or self._pending is not prepared.proposal:
                raise ValueError("foreign stale or replayed G4a prepared set")
            proposal = prepared.proposal
            expected = _sha(dict(proposal_sha256=proposal.proposal_sha256,
                                 owner_identity=self.identity,
                                 epoch=self._accepted.epoch))
            if prepared.prepared_sha256 != expected:
                raise ValueError("G4a prepared authority changed")
            rebuilt = self._evaluate(proposal.strains, proposal.origins)
            if canonical(rebuilt) != canonical(proposal.responses):
                raise ValueError("G4a prepared response changed")
            if check is not None:
                check("before_publication", None)
            old = self._accepted
            made = self._state(old.epoch + 1, proposal.strains,
                               tuple(row.history for row in rebuilt), rebuilt,
                               old.state_sha256)
            self._accepted = made
            published = True
            self._pending = None
            self._prepared = None
            if check is not None:
                check("after_publication", None)
            return made
        except BaseException:
            self._pending = None
            self._prepared = None
            raise
        finally:
            self._lock.release()
            if published:
                self._guard()

    def discard(self, proposal):
        self._enter()
        try:
            if self._pending is not proposal:
                raise ValueError("foreign stale or replayed G4a proposal")
            self._pending = None
            self._prepared = None
        finally:
            self._lock.release()

    def recover(self, state=None):
        self._guard()
        state = self._accepted if state is None else state
        if state is not self._accepted:
            raise ValueError("only the issued current G4a accepted state may recover")
        rows = []
        for index, response in enumerate(state.responses):
            rebuilt = self._response(index, response.strain, response.origin)
            if (canonical(rebuilt) != canonical(response)
                    or canonical(response.history) != canonical(state.histories[index])):
                raise ValueError("accepted G4a recovery replay mismatch")
            row = dict(station_id=response.station_id, family=response.family,
                       capability=response.capability,
                       law_identity=response.law_identity,
                       strain=response.strain, strain_low=response.strain_low,
                       resultants=response.resultants,
                       resultants_low=response.resultants_low,
                       tangent=response.tangent, tangent_low=response.tangent_low,
                       incremental_potential=response.incremental_potential,
                       stored_energy=response.stored_energy,
                       dissipation=response.dissipation,
                       history=response.history,
                       derivative_kind=response.derivative_kind)
            if response.capability == PHYSICAL_FIBRES:
                row["fibres"] = response.fibres
            rows.append(row)
        self._guard()
        return tuple(rows)

    def snapshot(self):
        self._guard()
        return canonical(self._accepted.descriptor())
