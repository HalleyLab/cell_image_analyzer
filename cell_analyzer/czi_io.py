"""CZI metadata inspection and calibrated multi-channel image loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pylibCZIrw import czi as pyczi

from .models import ChannelInfo, CziInfo, SceneInfo


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _extract_pixel_sizes_um(metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    distances = _as_list(
        _nested(
            metadata,
            "ImageDocument",
            "Metadata",
            "Scaling",
            "Items",
            "Distance",
            default=[],
        )
    )
    values: dict[str, float] = {}
    for item in distances:
        if not isinstance(item, dict):
            continue
        axis = str(item.get("@Id", "")).upper()
        try:
            # CZI stores these distance values in meters.
            values[axis] = float(item["Value"]) * 1_000_000.0
        except (KeyError, TypeError, ValueError):
            continue
    return values.get("X"), values.get("Y")


def _extract_channels(
    reader: pyczi.CziReader,
    metadata: dict[str, Any],
    dimensions: dict[str, tuple[int, int]],
) -> list[ChannelInfo]:
    nodes = _as_list(
        _nested(
            metadata,
            "ImageDocument",
            "Metadata",
            "Information",
            "Image",
            "Dimensions",
            "Channels",
            "Channel",
            default=[],
        )
    )
    node_by_index: dict[int, dict[str, Any]] = {}
    for fallback_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        raw_id = str(node.get("@Id", ""))
        try:
            index = int(raw_id.rsplit(":", 1)[-1])
        except ValueError:
            index = fallback_index
        node_by_index[index] = node

    start, count = dimensions.get("C", (0, max(len(nodes), 1)))
    channels: list[ChannelInfo] = []
    for index in range(start, start + count):
        node = node_by_index.get(index, {})
        name = str(node.get("@Name") or node.get("ShortName") or f"Channel_{index}")
        color = node.get("Color")
        try:
            pixel_type = reader.get_channel_pixel_type(index)
        except Exception:
            pixel_type = node.get("PixelType")
        channels.append(
            ChannelInfo(index=index, name=name, pixel_type=pixel_type, color=color)
        )
    return channels


def inspect_czi(path: str | Path) -> CziInfo:
    """Read normalized dimensions, channels, scenes, and calibration metadata."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CZI file not found: {source}")
    if source.suffix.lower() != ".czi":
        raise ValueError(f"Expected a .czi file, received: {source.name}")

    warnings: list[str] = []
    with pyczi.open_czi(str(source)) as reader:
        dimensions = {
            str(key): (int(value[0]), int(value[1]))
            for key, value in reader.total_bounding_box.items()
        }
        metadata = reader.metadata
        channels = _extract_channels(reader, metadata, dimensions)
        scenes = [
            SceneInfo(
                index=int(index),
                x=int(rectangle.x),
                y=int(rectangle.y),
                width=int(rectangle.w),
                height=int(rectangle.h),
            )
            for index, rectangle in sorted(reader.scenes_bounding_rectangle.items())
        ]
        pixel_size_x, pixel_size_y = _extract_pixel_sizes_um(metadata)
        image_info = _nested(
            metadata,
            "ImageDocument",
            "Metadata",
            "Information",
            "Image",
            default={},
        )
        objective_name = _nested(
            metadata,
            "ImageDocument",
            "Metadata",
            "Scaling",
            "AutoScaling",
            "ObjectiveName",
        )
        acquisition_datetime = image_info.get("AcquisitionDateAndTime")

    if pixel_size_x is None or pixel_size_y is None:
        warnings.append("Physical pixel size was not found; calibrated areas will be empty.")
    if not scenes:
        width = dimensions.get("X", (0, 0))[1]
        height = dimensions.get("Y", (0, 0))[1]
        scenes = [SceneInfo(index=0, x=0, y=0, width=width, height=height)]
        warnings.append("Scene metadata was absent; a single full-image scene was assumed.")

    return CziInfo(
        path=str(source),
        dimensions=dimensions,
        channels=channels,
        scenes=scenes,
        pixel_size_um_x=pixel_size_x,
        pixel_size_um_y=pixel_size_y,
        objective_name=objective_name,
        acquisition_datetime=acquisition_datetime,
        warnings=warnings,
    )


def _dimension_indices(
    dimensions: dict[str, tuple[int, int]],
    key: str,
    requested_index: int,
) -> int:
    start, count = dimensions.get(key, (0, 1))
    value = start + requested_index
    if requested_index < 0 or value >= start + count:
        raise IndexError(
            f"{key} index {requested_index} is outside the available range 0..{count - 1}."
        )
    return value


def _to_grayscale_plane(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    image = np.squeeze(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        rgb = image[..., :3].astype(np.float32, copy=False)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    raise ValueError(f"Unsupported CZI plane shape: {image.shape}")


def read_czi_channels(
    path: str | Path,
    info: CziInfo,
    *,
    scene: int,
    time_index: int,
    z_projection: str,
    z_index: int,
    zoom: float,
    channel_indices: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Load one scene and time point for all requested channels."""

    available_scenes = {item.index for item in info.scenes}
    if scene not in available_scenes:
        raise IndexError(
            f"Scene {scene} is unavailable. Available scenes: {sorted(available_scenes)}"
        )
    channels = channel_indices or [item.index for item in info.channels]
    available_channels = {item.index for item in info.channels}
    missing = set(channels) - available_channels
    if missing:
        raise IndexError(f"Unavailable channel indices: {sorted(missing)}")

    t_value = _dimension_indices(info.dimensions, "T", time_index)
    z_start, z_count = info.dimensions.get("Z", (0, 1))
    if z_projection == "single":
        z_values = [_dimension_indices(info.dimensions, "Z", z_index)]
    else:
        z_values = list(range(z_start, z_start + z_count))

    output: dict[int, np.ndarray] = {}
    with pyczi.open_czi(str(Path(path).expanduser().resolve())) as reader:
        has_explicit_scenes = bool(reader.scenes_bounding_rectangle)
        for channel in channels:
            accumulator: np.ndarray | None = None
            for position, z_value in enumerate(z_values):
                plane = {"C": int(channel), "T": int(t_value), "Z": int(z_value)}
                read_arguments: dict[str, Any] = {
                    "plane": plane,
                    "zoom": float(zoom),
                }
                # Scene-less CZI files reject even scene=0. Only pass a scene
                # argument when the document contains explicit scene metadata.
                if has_explicit_scenes:
                    read_arguments["scene"] = int(scene)
                image = _to_grayscale_plane(
                    reader.read(**read_arguments)
                )
                if z_projection == "max":
                    if accumulator is None:
                        accumulator = image.copy()
                    else:
                        np.maximum(accumulator, image, out=accumulator)
                elif z_projection == "mean":
                    if accumulator is None:
                        accumulator = image.astype(np.float32)
                    else:
                        accumulator += image.astype(np.float32, copy=False)
                else:
                    accumulator = image
                del image
            if accumulator is None:
                raise RuntimeError(f"No image plane was read for channel {channel}.")
            if z_projection == "mean" and len(z_values) > 1:
                accumulator /= float(len(z_values))
            output[channel] = accumulator
    return output
