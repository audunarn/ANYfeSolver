"""Compatibility facade for SESAM SIF results owned by ANYfileio."""

from anyfileio.sesam.sif import (
    SesamStressResult,
    read_sesam_sif_stress,
    read_sesam_sif_summary,
)

__all__ = ["SesamStressResult", "read_sesam_sif_stress", "read_sesam_sif_summary"]
