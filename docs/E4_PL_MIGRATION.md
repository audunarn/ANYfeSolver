# E4-PL Q4 migration

E4-PL is the production default for four-node shells. Existing callers using
`create_shell_element(...)`, `create_element("shell", ...)`, generated panel
meshes, runtime geometry, cylinder Q4 meshes, or SESAM Q4 imports select it
without an API change.

ANYfem integrations must construct solver shells through ANYfeSolver's public
`create_element("shell", ...)` selector. Direct `ShellElement(...)`
construction bypasses the Q4 default and is deprecated for four-node shells.

## Identify the selected formulation

```python
from anyfesolver import shell_formulation_diagnostics

print(shell_formulation_diagnostics(node_count=4))
```

The diagnostic is mechanics-free and reports the normalized request, selected
formulation, topology policy, rollback state, and removal target.

Serialized E4-PL elements carry `formulation_id` and reconstruct with
`QualifiedE4PLShellElement.from_dict(...)`. Preserve that field in restart and
interchange records; do not down-cast the record to `ShellElement`.

## Temporary rollback

For a diagnosed compatibility incident only:

```python
from anyfesolver import create_shell_element

element = create_shell_element(
    1,
    [1, 2, 3, 4],
    "steel",
    formulation="legacy",
)
```

The explicit aliases `legacy`, `legacy-shell`, and `legacy-s4`, plus direct Q4
`ShellElement` construction, emit `LegacyQ4DeprecationWarning`. Record the
model, reason, warning, solver version, and E4-PL comparison whenever rollback
is used. The route remains available through 0.4.x and is scheduled for
removal no earlier than 0.5.0 after two clean release gates.

TRI3, TRI6, Q8, and Q8R continue to use `ShellElement`; they do not emit the
legacy-Q4 warning and are not part of the retirement.

## Removal gates

Legacy Q4 can be removed only when two consecutive release gates pass the
complete functional suite, representative real-model comparisons, maintained
parity matrix, serialization/restart checks, and serialized performance lanes,
with no unresolved rollback incident. Removal must retain a fail-closed error
for the old Q4 selectors and document the last version containing rollback.
