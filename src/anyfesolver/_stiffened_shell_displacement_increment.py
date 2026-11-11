"""Private signed displacement cutback state; not yet dispatched publicly."""
from dataclasses import dataclass,asdict,replace
from math import isfinite
from ._stiffened_shell_increment_control import AdaptiveIncrementControl

MAX_EVENT_WORK = 2**63-1


@dataclass(frozen=True)
class DisplacementAttempt:
    displacement: float
    accepted_from: float
    requested_target: float
    accepted: bool
    evaluations: int
    linear_solves: int
    rejected_trials: int
    event_cell_evaluations: int = 0


class DisplacementIncrementLimit(RuntimeError):
    def __init__(self,message,state):
        super().__init__(message)
        self.state=state


@dataclass(frozen=True)
class DisplacementIncrementState:
    limits: AdaptiveIncrementControl
    targets: tuple
    initial: float
    attempts: tuple = ()

    def __post_init__(self):
        if type(self.limits) is not AdaptiveIncrementControl:
            raise ValueError("typed displacement increment limits required")
        if (type(self.initial) is not float or not isfinite(self.initial) or
                type(self.targets) is not tuple or not 1<=len(self.targets)<=1024 or
                any(type(v) is not float or not isfinite(v) for v in self.targets) or
                any(a==b for a,b in zip((self.initial,)+self.targets,self.targets)) or
                type(self.attempts) is not tuple or len(self.attempts)>self.limits.max_attempts):
            raise ValueError("bounded signed displacement history required")
        self.progress()

    def progress(self):
        """Replay only bounded controller metadata, never mechanical history."""
        cursor=0;depth=0;origin=self.initial;accepted=self.initial
        for row in self.attempts:
            if type(row) is not DisplacementAttempt or cursor==len(self.targets):
                raise ValueError("invalid displacement attempt sequence")
            target=self.targets[cursor]
            step=(target-origin)/(2**depth)
            expected=target if abs(step)>=abs(target-accepted) else accepted+step
            if (type(row.accepted) is not bool or row.displacement!=expected or
                    row.accepted_from!=accepted or row.requested_target!=target or
                    any(type(v) is not float or not isfinite(v) for v in
                        (row.displacement,row.accepted_from,row.requested_target)) or
                    any(type(v) is not int or v<0 for v in
                        (row.evaluations,row.linear_solves,row.rejected_trials)) or
                    type(row.event_cell_evaluations) is not int or
                    not 0<=row.event_cell_evaluations<=MAX_EVENT_WORK or
                    row.evaluations<1 or
                    depth>self.limits.max_cutbacks):
                raise ValueError("inconsistent displacement attempt")
            if row.accepted:
                accepted=row.displacement
                if accepted==target:cursor+=1;origin=accepted;depth=0
            else:depth+=1
        return cursor,depth,origin,accepted

    @property
    def completed(self):
        return self.progress()[0]==len(self.targets)

    def next_target(self):
        cursor,depth,origin,accepted=self.progress()
        if cursor==len(self.targets):raise ValueError("displacement sequence complete")
        if len(self.attempts)>=self.limits.max_attempts:
            raise DisplacementIncrementLimit("displacement attempt limit",self)
        if depth>self.limits.max_cutbacks:
            raise DisplacementIncrementLimit("displacement cutback limit",self)
        target=self.targets[cursor];step=(target-origin)/(2**depth)
        proposed=target if abs(step)>=abs(target-accepted) else accepted+step
        if not isfinite(proposed) or proposed==accepted:
            raise DisplacementIncrementLimit("displacement increment not representable",self)
        return proposed

    def record(self,accepted,*,evaluations,linear_solves,rejected_trials,event_cell_evaluations=0):
        proposed=self.next_target();cursor,_,_,previous=self.progress()
        row=DisplacementAttempt(proposed,previous,self.targets[cursor],accepted,
                                evaluations,linear_solves,rejected_trials,event_cell_evaluations)
        return replace(self,attempts=self.attempts+(row,))

    def payload(self):
        eventful=any(row.event_cell_evaluations for row in self.attempts)
        rows=[asdict(row) for row in self.attempts]
        if not eventful:
            for row in rows:row.pop("event_cell_evaluations")
        return dict(policy="DISPLACEMENT_INCREMENT_V2" if eventful else "DISPLACEMENT_INCREMENT_V1",
                    limits=asdict(self.limits),targets=list(self.targets),initial=self.initial,attempts=rows)

    @classmethod
    def from_payload(cls,value):
        if (type(value) is not dict or set(value)!={"policy","limits","targets","initial","attempts"} or
                value["policy"] not in ("DISPLACEMENT_INCREMENT_V1","DISPLACEMENT_INCREMENT_V2") or
                type(value["limits"]) is not dict or set(value["limits"])!={"max_cutbacks","max_attempts"} or
                type(value["targets"]) is not list or type(value["attempts"]) is not list or
                len(value["targets"])>1024 or len(value["attempts"])>1024):
            raise ValueError("invalid displacement controller payload")
        eventful=value["policy"]=="DISPLACEMENT_INCREMENT_V2"
        keys={"displacement","accepted_from","requested_target","accepted","evaluations","linear_solves","rejected_trials"}
        if eventful:keys.add("event_cell_evaluations")
        if any(type(row) is not dict or set(row)!=keys for row in value["attempts"]):
            raise ValueError("invalid displacement attempt fields")
        if eventful and not any(row["event_cell_evaluations"] for row in value["attempts"]):
            raise ValueError("eventful displacement payload requires recorded event work")
        try:
            rows=tuple(DisplacementAttempt(**row) for row in value["attempts"])
            return cls(AdaptiveIncrementControl(**value["limits"]),tuple(value["targets"]),value["initial"],rows)
        except (TypeError,KeyError) as error:
            raise ValueError("invalid displacement controller fields") from error
