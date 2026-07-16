from __future__ import annotations

from dataclasses import dataclass

from .ppm_base import _PPMBuilderBase


@dataclass
class PPMBuilder2D(_PPMBuilderBase):
    """2D PPM builder (polygonal)."""

    dim: int = 2
