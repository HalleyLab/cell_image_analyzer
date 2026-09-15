# Cell Fluorescence Analyzer

This is a separate extension beside `cell_analyzer`. It does not replace or modify the original cell-analysis code. It reuses the established CZI reader, raw-intensity preprocessing, and threshold functions, then adds cellular Aβ/Iba1/CD68 measurements.

## What it measures

- Anatomical/tissue ROI area.
- Aβ area fraction, plaque count, plaque density, plaque area, perimeter, circularity, solidity, and eccentricity.
- Whole-ROI Iba1 and CD68 positive area fractions.
- CD68-positive area inside the Iba1-positive mask (`CD68 ∩ Iba1 / Iba1`).
- Aβ–CD68 and Aβ–Iba1 two-dimensional overlap fractions.
- Iba1, CD68, overlap, and intensity metrics in configurable cumulative
  plaque neighborhoods; every radius includes the plaque itself.
- Per-image, per-plaque, batch-combined, and animal-level tables.

The 20× two-dimensional Aβ/CD68 overlap is an association metric, not proof of intracellular uptake. Uptake requires higher-magnification confocal Z-stacks and 3-D containment analysis.

## Start the GUI

From the project root:

```powershell
cd E:\projects\cell_analyzer
"C:\Program Files\Python312\python.exe" brain_section_analyzer\launch_gui.py
```

The launcher adds the project root to `sys.path`, so it does not have the relative-import problem produced by running a package module directly.

GUI workflow:

1. Click **Add images**. The button can be used repeatedly to add files from different folders.
2. Inspect the first file and verify the channel indices. CZI channel names are not assumed to equal fluorophore wavelengths.
3. Tune a representative file, then click **Apply current settings to all** to copy the complete parameter set to the batch.
4. Use `full_image` only when every image is already cropped to one anatomical ROI.
5. For irregular cortex/hippocampus ROIs, choose `mask_directory` and provide one binary mask per image.
6. Set the plaque-area filter and distance-ring edges.
7. Optionally select a metadata CSV so sections can be aggregated by animal.
8. Save the YAML parameters and run all images.

## Use the Jupyter Notebook

Open the project-level notebook:

```text
E:\projects\cell_analyzer\brain_section_analyzer.ipynb
```

Run the cells from top to bottom. The notebook panel provides:

- **Add images** that can be clicked repeatedly for files in different folders.
- Previous/next navigation and a separate preview for every selected CZI.
- Explicit Aβ, Iba1, and CD68 channel selectors.
- Manual or automatic thresholds and Gaussian smoothing for Aβ, Iba1, and CD68.
- Independent Iba1 and CD68 connected-object filters for opening/closing, hole filling,
  minimum/maximum area, circularity, solidity, and eccentricity.
- Optional DAPI-nucleus segmentation plus local Iba1 confirmation for microglia counts
  in the tissue ROI, each plaque-distance ring, and each plaque.
- Optional final plaque gate requiring a configurable number of DAPI+/Iba1+
  microglia within a configurable distance from each plaque edge.
- Plaque opening/closing, complete or size-limited hole filling, area, circularity, solidity, and eccentricity filters.
- A soma-exclusion stage that uses a lower Aβ threshold to recover the complete cell outline, then combines physical diameter, filled outer circularity, filled solidity, nuclear-hole fraction, and center/shell intensity ratio.
- Optional watershed splitting for touching plaques, with peak distance, peak height, and compactness controls.
- Boundary exclusion, peri-plaque distance rings, ROI-mask inversion, QC/mask output switches, and preview/final zoom.
- Full-image or mask-directory anatomical ROIs.
- Low-resolution preview zoom and an independent final-analysis zoom.
- Parameter YAML saving, batch execution, QC images, and animal-level tables.

Each file remembers its own settings while you move with **Previous/Next**. Use
**Apply current settings to all** after tuning a representative image to standardize the
complete parameter set. A later change affects only the current file until the button is
clicked again. The batch records the template, every per-file override, and every effective
file configuration in `batch_parameters.yaml`.

**Save session** writes a YAML containing all selected image paths, the output root,
the template, and per-file parameters, plus a companion `.files.txt` path list.
**Load session** restores either that file or a previous batch's
`batch_parameters.yaml`.

For biological comparisons, do not tune WT and PAC-cKO independently merely to make the
masks look similar. Use the same acquisition and threshold rules within one staining batch;
keep a per-file override only for a documented technical reason such as a different ROI mask.

### Why CD68 does not define plaques

Plaque candidates and soma exclusion are defined from Aβ only. CD68 is never used to accept
a plaque. The optional nearby-microglia gate is applied last using DAPI+/Iba1+ cells; disable
it for analyses intended to test microglial recruitment itself, because that gate can bias
genotype comparisons when recruitment differs. The summary always records plaque counts
before and after this gate. In the preview, accepted plaques have cyan boundaries,
excluded soma-like objects have magenta boundaries, and plaques rejected for lacking nearby
microglia have orange boundaries.
The Iba1 and CD68 preview panels outline accepted components in green and red. These
component filters clean the marker-positive masks used for area and overlap measurements;
they do not alter the Aβ plaque mask.

The default `shape_and_dark_center` mode is conservative: an object must match the configured
soma diameter and filled outer shape and must also contain a threshold-mask hole or a dark
center relative to its shell. `shape` is more aggressive and `off` disables this exclusion.
Always validate the magenta objects on representative WT and PAC-cKO images before applying
one parameter set to the complete staining batch.

## Tissue ROI masks

For an image named:

```text
WT01_cortex_S01.czi
```

the default mask suffix expects:

```text
WT01_cortex_S01_mask.png
```

White/nonzero pixels are analyzed and black pixels are excluded. PNG or TIFF masks are supported in a single-image YAML configuration. Batch mask lookup uses the filename suffix selected in the GUI.

## Metadata CSV

Copy `metadata_template.csv` and replace its example rows. Matching can use `source_name` or an absolute `source_file`. Recommended columns are:

```text
source_name,mouse_id,genotype,region,section_id,sex
```

`animal_summary.csv` is produced only when `mouse_id` is available. Images and plaques are not treated as independent biological replicates.

## Main outputs

Each image folder contains:

- `image_summary.csv`: one row per image/section ROI.
- `plaque_measurements.csv`: one row per accepted interior plaque.
- `abeta_candidate_qc.csv`: every initial Aβ candidate, its soma/plaque features, exclusion decision, and reason.
- `marker_component_qc.csv`: every threshold-positive Iba1/CD68 component, morphology,
  accepted/excluded decision, and exclusion reason.
- `microglia_cells.csv`: every DAPI nucleus, morphology/QC decision, local Iba1 fraction,
  nearest plaque, distance to plaque, and assigned distance ring.
- `plaque_ring_metrics.csv`: a long-format table containing the five requested Iba1/CD68
  ring metrics for each plaque and ring.
- `cell_analysis_measurements.xlsx`: image, plaque, Aβ-candidate QC, marker-component QC, and threshold sheets.
- `cell_analysis_qc.png`: composite, plaque mask, distance-ring boundary, and tissue-ROI boundary.
- `processing_images/`: raw channels in their original CZI display colors plus every
  segmentation overlay, final mask preview, distance-ring mask, and ROI view.
- `tissue_roi_mask.tiff`, `plaque_labels.tiff`, `abeta_neuron_like_excluded.tiff`,
  `marker_component_labels.tiff`, `positive_masks.tiff`.
- `config_used.yaml` and `analysis_summary.json`.

The batch root contains:

- `selected_image_files.txt` and `batch_parameters.yaml`.
- `combined_image_summary.csv`.
- `combined_plaque_measurements.csv`.
- `combined_abeta_candidate_qc.csv`.
- `combined_marker_component_qc.csv`.
- `combined_microglia_cells.csv`.
- `combined_plaque_ring_metrics.csv`; the same values are in the dedicated
  `Plaque Ring Metrics` sheet of the batch Excel workbook.
- `animal_summary.csv`.
- `cell_analysis_batch_results.xlsx`.

## Interpretation of key columns

- `abeta_positive_fraction`: Aβ-positive area / tissue ROI area.
- `plaque_density_all_per_mm2`: all detected connected plaques / ROI mm².
- `plaque_count_interior`: plaques not touching the tissue/image boundary.
- `abeta_neuron_like_excluded_count`: initial Aβ components removed by the soma-like rule.
- `plaque_count_before_nearby_microglia_filter`: plaques before the optional
  DAPI+/Iba1+ proximity gate.
- `plaque_without_nearby_microglia_excluded_count`: plaques removed by that gate.
- `nearby_microglia_count`: accepted DAPI+/Iba1+ cells within the configured
  distance from each plaque edge.
- `iba1_component_accepted_count` / `cd68_component_accepted_count`: connected
  marker components retained after their independent morphology filters.
- `*_component_excluded_count`: marker components rejected by area or shape rules.
- `cd68_in_iba1_fraction_of_iba1`: CD68∩Iba1 area / Iba1-positive area.
- `ring_0_30um_iba1_positive_fraction`: Iba1-positive area in plaque pixels plus
  the region extending 30 µm outward from plaque edges / total combined area.
  With radii `0,15,30`, outputs are cumulative `ring_0_15um` and
  `ring_0_30um`; both include plaque pixels.
- `*_thresholded_mean_intensity`: sum of above-threshold raw intensity / total region pixels; sub-threshold pixels contribute zero.
- `*_positive_mean_intensity`: mean raw intensity among positive pixels only.

For final quantification, use `zoom: 1.0`, identical acquisition settings, identical Z range, and fixed thresholds within each staining batch. Always inspect the QC PNG before accepting a result.

## Command line

```powershell
python -m brain_section_analyzer inspect "I:\path\image.czi"
python -m brain_section_analyzer init-config "I:\path\image.czi" --output brain_config.yaml
python -m brain_section_analyzer run --config brain_config.yaml
python -m brain_section_analyzer batch --config brain_config.yaml --output-root "I:\results" "I:\a.czi" "I:\b.czi"
```

## Tests

From `E:\projects\cell_analyzer`:

```powershell
"C:\Program Files\Python312\python.exe" -m unittest discover -s brain_section_analyzer\tests -v
```
