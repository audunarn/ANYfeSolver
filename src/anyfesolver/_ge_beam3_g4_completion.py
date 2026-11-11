"""Private authenticated G4 heterogeneous accepted-history envelope.

The envelope owns no mechanics.  It can only replay a chain through a fresh
``MixedStationTransaction`` whose frozen definition is supplied by the caller.
Serialized bytes therefore cannot select a class, law, graph, or formulation.
"""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from ._ge_beam3_g1_elastic import owned
from ._ge_beam3_g4a_station_transaction import MixedStationTransaction
from ._ge_beam3_p5_seeded.core import canonical, sha
from ._ge_beam3_p5_seeded.codec import _load, MAX_BYTES


POLICY = "GE_BEAM3_G4_GENERAL_STATIC_MATERIAL_STATE_V1"
SCHEMA = "GE_BEAM3_G4_ACCEPTED_HISTORY_CHECKPOINT_V1"


def _bounded_ascii(value, name):
    if (type(value) is not str or not value.isascii()
            or not 1 <= len(value) <= 512):
        raise ValueError("bounded ASCII " + name + " required")
    return value


@dataclass(frozen=True)
class RestartAuthority:
    graph_identity: str
    formulation_identity: str
    section_identity: str
    constraint_identity: str
    load_identity: str
    control_identity: str
    provenance_identity: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _bounded_ascii(getattr(self, name), name)

    def descriptor(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def identity(self):
        return sha(dict(policy=POLICY, authority=self.descriptor()))


class AcceptedHistoryEnvelope:
    """Append-only canonical chain over one heterogeneous station owner."""

    def __init__(self, owner, authority):
        if type(owner) is not MixedStationTransaction:
            raise ValueError("exact heterogeneous station owner required")
        if type(authority) is not RestartAuthority:
            raise ValueError("exact G4 restart authority required")
        if owner.accepted.epoch != 0:
            raise ValueError("fresh virgin station owner required")
        self.owner = owner
        self.authority = authority
        self.identity = sha(dict(policy=POLICY, owner=owner.identity,
                                 authority=authority.descriptor()))
        self._records = ()
        self._capture = self._definition()

    def _definition(self):
        return canonical(dict(policy=POLICY, owner_identity=self.owner.identity,
                              authority=self.authority.descriptor(),
                              authority_identity=self.authority.identity,
                              envelope_identity=self.identity))

    def guard(self):
        self.owner._guard()
        if self._definition() != self._capture:
            raise ValueError("G4 restart authority changed")
        if self.owner.accepted.epoch != len(self._records):
            raise ValueError("G4 accepted cursor differs from record chain")

    @property
    def records(self):
        self.guard()
        return self._records

    def accept(self, strains, check=None):
        self.guard()
        if type(strains) is not tuple or len(strains) != 3:
            raise ValueError("exact heterogeneous strain set required")
        values = tuple(owned(row, (6,)) for row in strains)
        before = self.owner.accepted
        proposal = self.owner.trial_all(values, check=check)
        try:
            prepared = self.owner.prepare_all(proposal, check=check)
            accepted = self.owner.commit_all(prepared, check=check)
        except BaseException:
            self.guard()
            raise
        recovery = self.owner.recover(accepted)
        body = dict(epoch=accepted.epoch,
                    previous_sha256=before.state_sha256,
                    strains=values,
                    state_sha256=accepted.state_sha256,
                    histories_sha256=sha(accepted.histories),
                    recovery_sha256=sha(recovery),
                    production_qualified=False)
        record = {**body, "record_sha256": sha(body)}
        self._records = (*self._records, record)
        self.guard()
        return accepted

    def checkpoint(self):
        self.guard()
        body = dict(schema=SCHEMA, policy=POLICY,
                    envelope_identity=self.identity,
                    owner_identity=self.owner.identity,
                    authority=self.authority.descriptor(),
                    authority_identity=self.authority.identity,
                    completed_epochs=len(self._records),
                    records=self._records,
                    production_qualified=False)
        raw = canonical({**body, "checkpoint_sha256": sha(body)})
        if len(raw) > MAX_BYTES:
            raise ValueError("G4 checkpoint byte bound")
        return raw

    @classmethod
    def restore(cls, owner, authority, raw, *, expected_sha256):
        if (type(raw) is not bytes or type(expected_sha256) is not str
                or sha256(raw).hexdigest() != expected_sha256):
            raise ValueError("external G4 checkpoint authority mismatch")
        value = _load(raw.decode("ascii"))
        if canonical(value) != raw:
            raise ValueError("canonical G4 checkpoint required")
        keys = {"schema", "policy", "envelope_identity", "owner_identity",
                "authority", "authority_identity", "completed_epochs",
                "records", "production_qualified", "checkpoint_sha256"}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("exact G4 checkpoint schema")
        body = {key: item for key, item in value.items()
                if key != "checkpoint_sha256"}
        # Replay into a private scratch owner.  The caller-supplied owner remains
        # virgin if even the final record is malformed; only the returned owner
        # becomes visible after the complete chain has reproduced exactly.
        scratch = MixedStationTransaction(owner.definitions)
        if scratch.identity != owner.identity:
            raise ValueError("G4 scratch owner identity mismatch")
        made = cls(scratch, authority)
        if (value["schema"] != SCHEMA or value["policy"] != POLICY
                or value["production_qualified"] is not False
                or value["owner_identity"] != owner.identity
                or value["authority"] != authority.descriptor()
                or value["authority_identity"] != authority.identity
                or value["envelope_identity"] != made.identity
                or value["checkpoint_sha256"] != sha(body)):
            raise ValueError("G4 checkpoint definition binding")
        count = value["completed_epochs"]
        if (type(count) is not int or not 0 <= count <= 16
                or type(value["records"]) is not list
                or len(value["records"]) != count):
            raise ValueError("G4 checkpoint record extent")
        record_keys = {"epoch", "previous_sha256", "strains", "state_sha256",
                       "histories_sha256", "recovery_sha256",
                       "production_qualified", "record_sha256"}
        for index, row in enumerate(value["records"], 1):
            if (type(row) is not dict or set(row) != record_keys
                    or type(row["epoch"]) is not int or row["epoch"] != index
                    or row["production_qualified"] is not False):
                raise ValueError("exact G4 accepted record schema")
            record_body = {key: item for key, item in row.items()
                           if key != "record_sha256"}
            if row["record_sha256"] != sha(record_body):
                raise ValueError("G4 accepted record hash")
            prior = made.owner.accepted
            if row["previous_sha256"] != prior.state_sha256:
                raise ValueError("G4 accepted predecessor chain")
            strains = tuple(owned(item, (6,)) for item in row["strains"])
            accepted = made.accept(strains)
            actual = made._records[-1]
            if canonical(actual) != canonical(row):
                raise ValueError("G4 accepted state/history/recovery replay mismatch")
            if accepted.epoch != index:
                raise ValueError("G4 replay cursor")
        if made.checkpoint() != raw:
            raise ValueError("G4 checkpoint replay is not byte identical")
        return made
