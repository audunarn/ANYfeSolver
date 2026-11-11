"""Typed runtime errors for the anyfesolver-results capsule runtime."""
from __future__ import annotations


class CapsuleError(Exception):
    """Base class of every anyfesolver-results runtime error."""


class CapsuleValidationError(CapsuleError):
    """A recovery capsule failed a lightweight structural validation."""


class ReconstructionError(CapsuleError):
    """A response reconstruction could not be performed as requested."""


class DerivationError(CapsuleError):
    """A derived field specification could not be resolved or computed."""
