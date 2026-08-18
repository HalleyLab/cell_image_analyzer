"""Shared data models for CZI inspection and analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChannelInfo:
    """Metadata describing one image channel."""

    index: int
    name: str
    pixel_type: str | None = None
    color: str | None = None


@dataclass(frozen=True)
class SceneInfo:
    """Pixel-space bounding rectangle for one scene."""

    index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class CziInfo:
    """Normalized metadata required by the analyzer."""

    path: str
    dimensions: dict[str, tuple[int, int]]
    channels: list[ChannelInfo]
    scenes: list[SceneInfo]
    pixel_size_um_x: float | None = None
    pixel_size_um_y: float | None = None
    objective_name: str | None = None
    acquisition_datetime: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedChannel:
    """Raw channel data and the optional Gaussian-smoothed analysis image."""

    raw: Any
    analysis_image: Any
