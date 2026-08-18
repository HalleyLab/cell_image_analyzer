# CZI Cell Analyzer

This standalone project detects cell boundaries in a selected CZI channel, creates reusable ROIs, and measures every image channel inside each ROI. The original `morphology.ipynb` and `feature_functions.py` files are read-only references and are not modified.

The implementation uses raw CZI intensities, optional Gaussian denoising, thresholding, morphological cleanup, contour-quality filtering, and quantitative export. It does not require OpenCV (`cv2`).

## Main features

- Reads native `.czi` files with the ZEISS `pylibCZIrw` reader.
- Selects multiple CZI files, previews them one at a time, and runs them as one batch.
- Detects channel count, channel names, pixel type, scenes, Z/T dimensions, and physical pixel size.
- Supports a single Z plane, maximum projection, or mean projection.
- Lets each channel use independent Gaussian smoothing and signal-area threshold settings.
- Uses one user-selected channel to define cell boundaries for all measurements.
- Optionally separates touching cells with distance-transform watershed.
- Filters candidates by area, circularity, border contact, and local contrast.
- Reports ROI area, shape, and raw mean intensity for every channel.
- Reports signal-positive area and positive fraction for every channel inside every ROI.
- Exports ImageJ ROI ZIP, GeoJSON polygons, a 32-bit label TIFF, CSV, Excel, and QC previews.

## Installation

Use Python 3.10 through 3.14. Python 3.12 is recommended.

```powershell
cd path\to\cell_image_analyzer
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Graphical workflow

```powershell
.\.venv\Scripts\python.exe launch_gui.py
```

1. Select a CZI file and click **Inspect**.
2. Confirm the scene, time point, Z projection, and processing zoom.
3. Edit the independent parameter row for each discovered channel.
4. Select the channel that defines cell boundaries.
5. Edit segmentation settings if needed.
6. Save the configuration for reproducibility or run the analysis directly.

The GUI remains responsive while analysis runs and reports the final ROI count and result directory.

## Command-line workflow

Inspect a file:

```powershell
python run_analysis.py inspect "D:\data\image.czi"
```

Create a complete editable configuration with all detected channels:

```powershell
python run_analysis.py init-config "D:\data\image.czi" --output config.yaml
```

Run the analysis:

```powershell
python run_analysis.py run --config config.yaml
```

`config.example.yaml` is a ready-to-edit example. `cell_analyzer_batch.ipynb` is the clean multi-file notebook; `cell_analyzer.ipynb` remains available for the original single-file workflow and prior results.

### Live notebook tuning

After creating `config`, launch the multi-file interactive preview panel:

```python
from cell_analyzer.interactive import launch_batch_tuning_widget

tuner = launch_batch_tuning_widget(config)
tuner
```

Click **Select CZI files** to choose multiple images, or paste one absolute path per line and click **Apply file list**. Use **Preview file**, **Previous**, and **Next** to inspect every image. Each image remembers its own tuned settings. Use **Apply current settings to all** when the entire batch should share the current parameters. Channels are matched by channel name first and channel index second. Click **Load preview** below the batch controls to read the selected CZI plane.

The **Cell segmentation and size filters** section opens by default. Edit **Minimum cell area** and **Maximum cell area** to reject cells outside the required pixel-area range; the accepted-boundary preview and the `rejected by area` count update immediately. The eight preview panels show raw intensity, Gaussian smoothing, segmentation input, initial threshold, morphology cleanup, watershed candidates, accepted numbered cell boundaries, and signal-positive pixels. When the settings are satisfactory, choose an output root and click **Run all files**. A failed or incompatible file is recorded in the batch summary while the remaining files continue.

The original single-file interface remains available through `launch_tuning_widget(config)`. Batch analysis can also be started without the widget:

```python
from cell_analyzer.batch import run_batch_analysis

batch_result = run_batch_analysis(CZI_PATHS, config, OUTPUT_ROOT)
```

## Important parameters

### Input

- `scene`: zero-based CZI scene index.
- `time_index`: zero-based time index.
- `z_projection`: `single`, `max`, or `mean`.
- `z_index`: used when `z_projection` is `single`.
- `zoom`: CZI read scale from `0.01` to `1.0`. Full resolution is `1.0`.
- `segmentation_channel`: zero-based channel used to create cell ROIs.

When `zoom` is below `1.0`, areas in source pixels and calibrated square micrometers are corrected for the scale. Measurements are still made from the image data read at that zoom, so use `1.0` for final quantitative work when memory allows.

### Per-channel analysis

- `gaussian_sigma_px`: optional denoising scale applied directly to raw CZI intensities. Use `0` to disable smoothing.
- `measurement_threshold.method`: `none`, `otsu`, `yen`, `triangle`, or `percentile`.
- `measurement_threshold.percentile`: used by `percentile`.

The measurement threshold defines the signal-positive area for that channel. Mean intensity is always calculated across the complete cell ROI.

### Segmentation

- `threshold_method`: global or adaptive threshold used on the selected segmentation channel.
- `threshold_scale`: multiplier for automatic global thresholds. Values below 1 expand the detected mask; values above 1 make it stricter.
- `min_area_px` and `max_area_px`: accepted cell area at the analysis zoom.
- `min_circularity`: rejects highly irregular debris. A permissive default is used.
- `min_local_contrast_ratio`: compares each candidate with a surrounding ring.
- `fill_all_holes`: fills every enclosed hole in a detected cell mask.
- `border_exclusion_margin_px`: rejects any ROI entering this many pixels from an image edge when border clearing is enabled.
- `split_touching`: enables watershed splitting. Leave it disabled when cells are already separated.
- `min_peak_distance_px`: controls how close watershed cell centers may be. Increase it when one cell is split into several ROIs.
- `watershed_min_peak_height_px`: ignores distance-transform peaks below this absolute height.
- `watershed_min_peak_prominence_px`: suppresses weak secondary peaks inside a cell; increase it to reduce over-segmentation.
- `watershed_compactness`: increases the preference for compact watershed regions.

## Output files

Each CZI receives its own subdirectory below the selected batch output root. Every subdirectory contains the normal per-file outputs listed below.

- `roi_measurements.csv`: one row per ROI.
- `roi_measurements.xlsx`: measurements, summary, channels, and all parameters.
- `roi_labels.tiff`: 32-bit ROI label image; background is zero.
- `imagej_rois.zip`: polygon ROIs for Fiji/ImageJ.
- `rois.geojson`: polygon ROIs in analysis-pixel coordinates.
- `roi_overlay.png`: selected-channel preview with red ROI boundaries.
- `channel_previews.png`: display-scaled preview of every channel; analysis still uses raw intensity values.
- `config_used.yaml`: exact validated configuration used for the run.
- `czi_metadata.json`: standardized input metadata.
- `analysis_summary.json`: ROI count, diagnostics, warnings, and output paths.

The batch output root also contains:

- `batch_summary.csv`, `batch_summary.xlsx`, and `batch_analysis_summary.json`: completion state, ROI count, output directory, and error message for every input file.
- `combined_roi_measurements.csv` and `combined_roi_measurements.xlsx`: all ROI measurements combined across successfully analyzed files, with source-file columns.

## Parameter tuning strategy

Start at `zoom: 0.25` for fast tuning. Inspect `roi_overlay.png`, then adjust the segmentation threshold. Use minimum area to suppress noise, peak distance to control watershed splitting, and local contrast to reject weak objects. Switch to `zoom: 1.0` for the final run and scale pixel-based segmentation parameters proportionally.

## Tests

The test suite generates a true three-channel CZI, runs the complete pipeline, and validates its measurement and ROI outputs.

```powershell
python -m unittest discover -s tests -v
```

## CZI reader references

- [ZEISS pylibCZIrw API](https://zeiss.github.io/pylibczirw/)
- [ZEISS CZI image format](https://www.zeiss.com/microscopy/en/products/software/zeiss-zen/czi-image-file-format.html)
