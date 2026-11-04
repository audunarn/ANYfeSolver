# Principal display axes

`anysolver_results.principals.principal_axes` is the existing NumPy-only display
candidate recovered byte-for-byte from shipping wheel SHA-256
`3c35d2fc8aee1a31e8fb8994fd8d20e91c84b8b8c3810af169c2475c40cebf52`,
followed only by the owner-adopted metadata admission repairs recorded in ANYpost's
living `docs/IMPLEMENTATION_TASK.md` task record. It introduces no physical solver
formula, derived-operation registry entry or recovery capability.

Values use unchanged canonical `principal_stresses`; axes are rank-major
unoriented directions, optionally mapped with proper orthonormal basis columns
from local to global coordinates. The existing 64 float64-eps display uniqueness
convention marks unresolved directions false and NaN. Arrays publish read-only
only after successful completion. Cancellation returns no partial result.

The unchanged working estimate is `count * 512 + 4096` bytes. Shape multiplication
uses Python integers; admitted shape dimensions must be plain nonnegative ints.
For ndarrays, the base NumPy descriptor supplies the actual shape; subclass or
instance shape overrides are ignored without being called.
Tensor and basis metadata are checked before conversion, and basis metadata must
broadcast to the tensor shape. The working limit is an admission estimate, not
an operating-system RAM reservation or measured peak-memory guarantee.

Evidence and exact source origins are recorded under the sibling owner-worktree
`evidence` directory. Focused algebraic/facade checks do not broaden scientific,
installed-artifact, rendering or solver qualification; parent owns acceptance.
