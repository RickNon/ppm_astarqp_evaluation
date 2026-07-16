from __future__ import annotations

from dataclasses import dataclass

from .ppm_base import _PPMBuilderBase


@dataclass
class PPMBuilder3D(_PPMBuilderBase):
    """3D PPM builder (polyhedral)."""

    dim: int = 3
