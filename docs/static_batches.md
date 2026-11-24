# Independent static load histories

`iter_linear_batches` solves arbitrary ordered `StaticCase` records against one
unchanged model. Each case has its own `LoadCase`; `time` is an optional label,
not an interpolation parameter or a proportional-load assumption. The iterator
defaults to blocks of at most 32 cases. Existing `solve_linear` and
`solve_linear_many` return types remain unchanged.

```python
from contextlib import closing
from anysolver import (AnalysisSession, EnvelopeReducer, FixedSupport, LoadCase,
    RecoveryConfig, ResourceConfig, StaticCase, iter_linear_batches)
from anysolver.elements import BeamElement
from anysolver.fe_core import FEModel

model = FEModel("cantilever history")
model.add_material("steel", 210e9, 0.3, density=7850.0)
model.add_node(1, 0.0, 0.0, 0.0)
model.add_node(2, 1.0, 0.0, 0.0)
model.add_element(1, BeamElement(1, [1, 2], "steel",
    {"area": 0.01, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}))
model.add_boundary_condition(FixedSupport("clamp", [1]))
selected_dofs = [model.mesh.nodes[2].dofs[i] for i in (0, 2)]

def history():
    for i, (time, axial, transverse) in enumerate([
            (0.0, 1000.0, 0.0), (0.17, -250.0, 500.0),
            (3.2, 400.0, -150.0)]):
        load = LoadCase(str(i))
        load.add_nodal_load(2, [axial, 0.0, transverse, 0.0, 0.0, 0.0])
        yield StaticCase(str(i), load, time)

envelope = EnvelopeReducer()
with AnalysisSession(model) as session:
    with closing(iter_linear_batches(model, history(), session=session,
            batch_size=2, output_dofs=selected_dofs,
            recovery_config=RecoveryConfig(node_ids=[1, 2], element_ids=[1],
                components=["von_mises"], include_displacements=False,
                include_reactions=True),
            resource_config=ResourceConfig(solver_threads=1, assembly_threads=1,
                recovery_threads=1, memory_limit_bytes=64 * 1024 * 1024))) as stream:
        for block in stream:
            envelope.update_batch(block)
        envelope.finish()  # Certify exhaustion, never call after failure/break.
print(envelope.extrema[("displacement", selected_dofs[1])])
```

The complete scripts [supported history](../examples/static_batch_supported.py)
and [free-free history](../examples/static_batch_free_free.py) print selected
responses, per-case gauge diagnostics and governing envelope records. From a
source checkout in PowerShell, run:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python examples/static_batch_supported.py
python examples/static_batch_free_free.py
python examples/static_batch_free_free.py --include-failure
python examples/static_batch_free_free.py --disconnected-control
```

The free-free fixture combines independent balanced axial and torsional patterns
at irregular times. Its failure controls retain previously certified envelopes
with `complete=False`. A disconnected free-free model requires balance on every
component; global resultant balance alone is insufficient. The disconnected
control applies equal/opposite forces to different components and demonstrates
that refusal. The default physical route does not introduce balancing reactions
to make an incompatible load admissible.

Each `StaticBatch` carries input indices, case IDs, times, output DOF order,
a displacement matrix with one column per case, per-case recovery dictionaries,
and diagnostics. Cases are assembled independently. Exact plain builtin loads
are copied when observed, so a generator may reuse and update one mutable load
object. Other load providers are assembled immediately before requesting the
next input. They are never evaluated again to identify a failure.

The session retains a prepared scaled supported operator or a distinct augmented
free-free operator using its bounded factorization cache. Selected displacement
rows use the existing affine output plan. Stress recovery uses the existing
public recovery path, including qualified batching and scalar fallbacks. Stress
or reaction requests currently reconstruct a full displacement block internally;
selected returned rows do not imply that all internal vectors are selected.

Reactions use the actual assembled case load in `K u - F`. Per-case
`constraint_forces` separates support reactions, MPC slave forces, master
equivalent forces and nullspace diagnostics. Selected reaction requests omit the
full residual vector and filter physical node buckets; `reaction_node_ids`
declares their stable requested scope, including nodes with zero compact-map
entries. `EnvelopeReducer.update_batch` consumes these records and preserves
governing case identities.

`constraint_mode="auto"` preserves physical free-free load compatibility.
Incompatible loads raise `StaticBatchError`. Explicit
`allow_unbalanced_free_free=True` returns the existing gauged solution with
balancing generalized reactions and warnings. Multipliers, gauge residuals and
per-case backward errors are reported. Session-backed single-case free-free
solves reuse the augmented factor. The collecting API also accepts the explicit
`solve_linear_many(..., constraint_mode="nullspace")` route; its existing
supported/auto behavior remains available. An explicit `factorization_cache`
remains caller-owned in this route. If a session is also supplied, the cache
must be that session's cache; conflicting cache arguments raise `ValueError`.
`AnalysisSession(model, factorization_cache=cache)` can share that same cache
explicitly, and closing the session does not release caller-owned handles.

Input iteration, callbacks and yields hold no solver authority lease. Structural
model changes terminate the stream; normal load registration and load updates
remain permitted. Internally created sessions and source iterators close on
exhaustion, failure, cancellation or generator close. A supplied session remains
caller-owned. Cleanup preserves a primary failure when iterator close also fails.

Stress and reaction recovery inside one block share one package-owned
qualified recovery lease scope. The lease, the exact built-in beam authority
and the qualified trusted operation scopes are captured once per block; every
per-case recovery operation then borrows that lease and still runs the complete
class-authority, preflight and output guards under its own operation context,
so per-case failure identities, unchanged scans and guard behavior are
identical to the ordinary per-operation capture path. The scope closes before
progress callbacks, cancellation points and yields, and revokes its borrowed
lease on every exit path, including failure, cancellation and scope entry
failures. A context copied while the scope was live cannot retain a usable
borrowed lease after the scope has closed; recovery dispatched in such a
copied context takes an ordinary fresh lease capture with full guards, and a
borrowing operation must present the model the lease was captured for.

Models whose every element is an admitted exact built-in `BeamElement`
(`anyfesolver.exact_beam_batch`) assemble linear stiffness and recover
stresses in one batch.  One complete lease guard runs before the batch reads
anything and one after its last output, in place of the two per-element
observation guards of the scalar loops; the two bracket guards carry the
context of the first and last per-element guard they replace.  This relies
on the existing contract: changing a model during a request, including from
another thread, is outside the supported contract.  Single-threaded calls
accept and reject exactly the inputs of the scalar loops.  Every dispatch
the scalar path performs for an admitted element is first verified with pure
reads (class namespaces along the MRO, module globals, instance and mapping
namespaces free of shadowing names, the stiffness classification predicates)
or performed as the identical call (`_beam_material_properties` once per
element in element order; the sparsity kernel before any element side
effect).  Subclasses, replaced classes, methods or module globals, shadowing
instance attributes, DOFs outside the matrix or displacement vector, offsets,
generalized or fibre sections, non-exact materials, element activity, other
stiffness getters and small models take the unchanged scalar loop.  Results
are bit-identical to the scalar element methods because the batch performs
the identical floating-point operations: it reproduces each BLAS call with
stacked `np.matmul`, keeps elementwise operation order and per-element scalar
powers, and falls back on any floating-point exception.  A probe against the
scalar methods once per process and a check of the first and last element of
every batch disable the batch for the process on a mismatch; they detect a
broken kernel or platform and do not by themselves prove parity for the
interior elements of a batch.

`ResourceConfig.memory_limit_bytes` limits a conservative storage estimate and
adapts the requested block size; it refuses a request that cannot fit one case.
This is an allocation estimate, not an operating-system RSS cap. Native sparse
factor storage and arbitrary provider allocations remain estimates. Diagnostics
include effective block size, phase timings and process-wide monotonic backend
operation counts/deltas. Concurrent unrelated backend work affects those deltas.
Solver native pools, assembly Numba workers and recovery workers use the separate
`ResourceConfig` limits. Each operation restores its thread policy before input
iteration, callbacks and yields, including failure and early close paths.

Only certified blocks are yielded. A later failed block does not invalidate
already yielded responses. Callers should consume blocks incrementally and close
the iterator when stopping early; collecting all blocks retains all collected
outputs. Exact component superposition, nonlinear derived stress recomputation
and response-basis APIs are separate contracts and are not inferred from time
labels or implemented by this iterator.
