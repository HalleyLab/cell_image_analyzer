"""Unified metadata inspection and channel loading for microscopy images."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.transform import resize

from .czi_io import inspect_czi, read_czi_channels
from .models import ChannelInfo, CziInfo, SceneInfo


# CZI keeps its fast native reader. Other proprietary formats use Bio-Formats.
BIOFORMATS_IMAGE_SUFFIXES = {
    ".dv",
    ".ims",
    ".lei",
    ".lif",
    ".lof",
    ".lsm",
    ".nd2",
    ".oib",
    ".oif",
    ".oir",
    ".r3d",
    ".scn",
    ".tif",
    ".tiff",
    ".vsi",
    ".xlef",
    ".zvi",
}
SUPPORTED_IMAGE_SUFFIXES = {".czi", ".png", *BIOFORMATS_IMAGE_SUFFIXES}
MICROSCOPY_FILE_PATTERN = " ".join(
    f"*{suffix}" for suffix in sorted(SUPPORTED_IMAGE_SUFFIXES)
)
CACHE_DIRECTORY = Path(
    os.environ.get(
        "CELL_ANALYZER_CACHE_DIR",
        Path(__file__).resolve().parent.parent / ".cache",
    )
).expanduser().resolve()


def configure_cache_directory(path: str | Path) -> Path:
    """Route application and image-reader caches to one user-selected folder."""

    global CACHE_DIRECTORY
    directory = Path(path).expanduser().resolve()
    temporary_directory = directory / "tmp"
    for item in (directory, temporary_directory, directory / "cjdk", directory / "matplotlib"):
        item.mkdir(parents=True, exist_ok=True)
    for name, item in {
        "CELL_ANALYZER_CACHE_DIR": directory,
        "CJDK_CACHE_DIR": directory / "cjdk",
        "MPLCONFIGDIR": directory / "matplotlib",
        "TEMP": temporary_directory,
        "TMP": temporary_directory,
        "TMPDIR": temporary_directory,
        "XDG_CACHE_HOME": directory,
    }.items():
        os.environ[name] = str(item)
    tempfile.tempdir = str(temporary_directory)
    CACHE_DIRECTORY = directory
    return directory


def _validated_pixel_size(value: Any, axis: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"Pixel size {axis} must be greater than 0.")
    return result


def _pixel_size_from_total(
    total_um: float | None, pixel_count: int, axis: str
) -> float | None:
    if total_um is None:
        return None
    total = float(total_um)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Total {axis} must be greater than 0 micrometers.")
    return total / float(pixel_count)


def _input_suffix(path: str | Path) -> tuple[Path, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image file not found: {source}")
    suffix = source.suffix.casefold()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f"Unsupported image format {source.suffix!r}; expected {supported}.")
    return source, suffix


def _open_bio_image(path: Path):
    temporary_directory = configure_cache_directory(CACHE_DIRECTORY) / "tmp"
    os.environ.setdefault("BIOFORMATS_VERSION", "8.4.0")
    os.environ.setdefault("BFF_JAVA_VENDOR", "zulu")
    os.environ.setdefault("BFF_JAVA_VERSION", "11")
    os.environ.setdefault("BFF_JAVA_FETCH", "always")
    try:
        from scyjava import config as java_config

        java_config.set_cache_dir(CACHE_DIRECTORY / "jgo")
        java_config.set_m2_repo(CACHE_DIRECTORY / "m2" / "repository")
        java_temporary_option = f"-Djava.io.tmpdir={temporary_directory}"
        if java_temporary_option not in java_config.get_options():
            java_config.add_option(java_temporary_option)
        from bioio import BioImage
        from bioio_bioformats import Reader
    except ImportError as error:
        raise RuntimeError(
            "This file needs Bio-Formats. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from error
    try:
        return BioImage(str(path), reader=Reader)
    except Exception as error:
        raise RuntimeError(
            f"Bio-Formats could not open {path.name}. Keep companion data next to "
            "the main file (for example the .vsi data folder), then try again."
        ) from error


def _dimension_size(image: Any, axis: str) -> int:
    return max(1, int(getattr(image.dims, axis, 1)))


def _channel_color(channel: Any) -> str | None:
    color = getattr(channel, "color", None)
    if color is None:
        return None
    try:
        red, green, blue = color.as_rgb_tuple()
        return f"#{int(red):02X}{int(green):02X}{int(blue):02X}"
    except Exception:
        return None


def _inspect_bioformats(
    source: Path,
    *,
    pixel_size_um_x: float | None,
    pixel_size_um_y: float | None,
    image_width_um: float | None,
    image_height_um: float | None,
) -> CziInfo:
    image = _open_bio_image(source)
    scenes: list[SceneInfo] = []
    for index, _ in enumerate(image.scenes):
        image.set_scene(index)
        scenes.append(
            SceneInfo(
                index=index,
                x=0,
                y=0,
                width=_dimension_size(image, "X"),
                height=_dimension_size(image, "Y"),
            )
        )
    if not scenes:
        scenes.append(
            SceneInfo(
                index=0,
                x=0,
                y=0,
                width=_dimension_size(image, "X"),
                height=_dimension_size(image, "Y"),
            )
        )
    image.set_scene(0)
    dimensions = {axis: (0, _dimension_size(image, axis)) for axis in "TCZYX"}

    ome_channels: list[Any] = []
    try:
        ome_channels = list(image.metadata.images[0].pixels.channels)
    except Exception:
        pass
    names = list(image.channel_names or [])
    channels = [
        ChannelInfo(
            index=index,
            name=str(names[index]) if index < len(names) and names[index] else f"Channel_{index}",
            pixel_type=str(image.dtype),
            color=_channel_color(ome_channels[index]) if index < len(ome_channels) else None,
        )
        for index in range(dimensions["C"][1])
    ]

    physical_sizes = image.physical_pixel_sizes
    size_x = _validated_pixel_size(getattr(physical_sizes, "X", None), "X")
    size_y = _validated_pixel_size(getattr(physical_sizes, "Y", None), "Y")
    if size_x is None:
        size_x = _pixel_size_from_total(image_width_um, dimensions["X"][1], "width")
    if size_y is None:
        size_y = _pixel_size_from_total(image_height_um, dimensions["Y"][1], "height")
    if size_x is None:
        size_x = _validated_pixel_size(pixel_size_um_x, "X")
    if size_y is None:
        size_y = _validated_pixel_size(pixel_size_um_y, "Y")
    warnings: list[str] = []
    if size_x is None or size_y is None:
        warnings.append(
            "The image metadata has no reliable X/Y calibration. Set input.pixel_size_um_x "
            "and input.pixel_size_um_y before quantitative analysis."
        )

    acquisition_datetime = None
    try:
        value = image.metadata.images[0].acquisition_date
        acquisition_datetime = str(value) if value is not None else None
    except Exception:
        pass
    return CziInfo(
        path=str(source),
        dimensions=dimensions,
        channels=channels,
        scenes=scenes,
        pixel_size_um_x=size_x,
        pixel_size_um_y=size_y,
        acquisition_datetime=acquisition_datetime,
        warnings=warnings,
    )


def inspect_image(
    path: str | Path,
    *,
    pixel_size_um_x: float | None = None,
    pixel_size_um_y: float | None = None,
    image_width_um: float | None = None,
    image_height_um: float | None = None,
) -> CziInfo:
    """Return normalized analyzer metadata for a supported image."""

    source, suffix = _input_suffix(path)
    if suffix == ".czi":
        return inspect_czi(source)
    if suffix in BIOFORMATS_IMAGE_SUFFIXES:
        return _inspect_bioformats(
            source,
            pixel_size_um_x=pixel_size_um_x,
            pixel_size_um_y=pixel_size_um_y,
            image_width_um=image_width_um,
            image_height_um=image_height_um,
        )

    with Image.open(source) as image:
        width, height = image.size
        mode = image.mode
    size_x = _pixel_size_from_total(image_width_um, width, "width")
    size_y = _pixel_size_from_total(image_height_um, height, "height")
    if size_x is None:
        size_x = _validated_pixel_size(pixel_size_um_x, "X")
    if size_y is None:
        size_y = _validated_pixel_size(pixel_size_um_y, "Y")
    warnings: list[str] = []
    if size_x is None or size_y is None:
        warnings.append(
            "PNG files do not provide reliable microscopy calibration. Set "
            "input.image_width_um and input.image_height_um before analysis."
        )
    return CziInfo(
        path=str(source),
        dimensions={
            "C": (0, 1),
            "T": (0, 1),
            "Z": (0, 1),
            "X": (0, int(width)),
            "Y": (0, int(height)),
        },
        channels=[ChannelInfo(index=0, name="Intensity", pixel_type=mode)],
        scenes=[SceneInfo(index=0, x=0, y=0, width=int(width), height=int(height))],
        pixel_size_um_x=size_x,
        pixel_size_um_y=size_y,
        warnings=warnings,
    )


def _read_png_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in {"P", "CMYK", "YCbCr", "HSV"}:
            image = image.convert("RGB")
        array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[..., 0]
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        rgb = array[..., :3].astype(np.float32, copy=False)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    raise ValueError(f"Unsupported PNG array shape: {array.shape}")


def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    target_shape = tuple(max(1, int(round(length * scale))) for length in image.shape)
    return resize(
        image,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32, copy=False)


def _read_bioformats_channels(
    source: Path,
    info: CziInfo,
    *,
    scene: int,
    time_index: int,
    z_projection: str,
    z_index: int,
    zoom: float,
    channel_indices: list[int] | None,
) -> dict[int, np.ndarray]:
    image = _open_bio_image(source)
    if scene < 0 or scene >= len(image.scenes):
        raise IndexError(f"Scene {scene} is outside the available range 0..{len(image.scenes) - 1}.")
    image.set_scene(scene)
    channel_count = _dimension_size(image, "C")
    time_count = _dimension_size(image, "T")
    z_count = _dimension_size(image, "Z")
    channels = list(range(channel_count)) if channel_indices is None else list(channel_indices)
    missing = [index for index in channels if index < 0 or index >= channel_count]
    if missing:
        raise IndexError(f"Unavailable channel indices: {sorted(set(missing))}")
    if time_index < 0 or time_index >= time_count:
        raise IndexError(f"T index {time_index} is outside the available range 0..{time_count - 1}.")
    if z_projection == "single":
        if z_index < 0 or z_index >= z_count:
            raise IndexError(f"Z index {z_index} is outside the available range 0..{z_count - 1}.")
        z_values = [z_index]
    else:
        z_values = list(range(z_count))

    output: dict[int, np.ndarray] = {}
    for channel in channels:
        accumulator: np.ndarray | None = None
        for z_value in z_values:
            plane = np.squeeze(
                np.asarray(
                    image.get_image_data(
                        "YX", C=channel, T=time_index, Z=z_value
                    )
                )
            )
            if plane.ndim != 2:
                raise ValueError(f"Unsupported Bio-Formats plane shape: {plane.shape}")
            if z_projection == "max":
                if accumulator is None:
                    accumulator = plane.copy()
                else:
                    np.maximum(accumulator, plane, out=accumulator)
            elif z_projection == "mean":
                if accumulator is None:
                    accumulator = plane.astype(np.float32)
                else:
                    accumulator += plane.astype(np.float32, copy=False)
            else:
                accumulator = plane
        if accumulator is None:
            raise RuntimeError(f"No image plane was read for channel {channel}.")
        if z_projection == "mean" and len(z_values) > 1:
            accumulator /= float(len(z_values))
        output[channel] = _resize_image(accumulator, zoom)
    return output


def read_image_channels(
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
    """Load requested channels and apply the selected Z projection."""

    source, suffix = _input_suffix(path)
    if z_projection not in {"single", "max", "mean"}:
        raise ValueError("z_projection must be 'single', 'max', or 'mean'.")
    scale = float(zoom)
    if not 0.01 <= scale <= 1.0:
        raise ValueError("zoom must be between 0.01 and 1.0.")
    if suffix == ".czi":
        return read_czi_channels(
            source,
            info,
            scene=scene,
            time_index=time_index,
            z_projection=z_projection,
            z_index=z_index,
            zoom=scale,
            channel_indices=channel_indices,
        )
    if suffix in BIOFORMATS_IMAGE_SUFFIXES:
        return _read_bioformats_channels(
            source,
            info,
            scene=scene,
            time_index=time_index,
            z_projection=z_projection,
            z_index=z_index,
            zoom=scale,
            channel_indices=channel_indices,
        )

    requested = [0] if channel_indices is None else list(channel_indices)
    if requested != [0]:
        raise IndexError("PNG input contains one fixed intensity channel (channel 0).")
    if scene != 0 or time_index != 0 or z_index != 0:
        raise IndexError("PNG input supports only scene=0, time_index=0, and z_index=0.")
    return {0: _resize_image(_read_png_grayscale(source), scale)}
