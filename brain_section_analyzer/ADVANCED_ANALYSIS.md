# Advanced analysis and processing previews

The desktop app is still a cell-analysis app. Channel names, wavelengths and
colors are assigned by the user; no cell type or staining marker is assumed.

## Where to configure analyses

Configure **Cell Counting** before **Advanced Analysis** when an analysis needs
counted cells. The tabs are displayed in that order. Then independently enable:

- **Neighbour analysis**: the previous reference-object, cumulative distance
  range and nearby-cell workflow. Select any enabled reference channel.
  All cumulative ranges include the reference object's interior and extend
  the specified distance from its edge. New sessions leave this option off;
  saved sessions predating advanced options preserve their original workflow.
- **Colocalization**: add one or more channel pairs in **Colocalization &
  distances**. Measures Pearson correlation, directional Manders M1/M2,
  intersection area, intersection fractions, Jaccard and Dice.
- **Object distances (A -> B)**: uses the same channel pairs and exports
  each source object's nearest target-centroid distance, minimum distance
  to the target mask, overlap fraction and proximity classification.
- **Per-cell measurements**: requires Cell Counting. Each confirmed nucleus and
  a configurable physical expansion form a mutually exclusive nearest-nucleus
  region. All enabled channels are exported in long format with raw
  background-corrected mean/integrated intensity, positive fraction/area and
  positive-pixel mean. These are nuclear/perinuclear measurement regions, not
  segmented cell bodies.
- **Radial profiles**: selects any enabled channel as a reference. Bin 0 is the
  complete filtered reference-object interior; external `(inner, outer]` bins
  extend from object edges. Each external pixel is assigned to its nearest
  reference object, so neighboring profiles do not double-count pixels.
- **Spatial distribution**: for selected channels, exports object centroids,
  same-channel nearest-centroid distance, neighbors within a configurable
  radius, ROI-edge truncation flags and density. It does not apply statistical
  edge correction or claim a clustering significance test.
- **Skeleton analysis**: skeletonizes each selected filtered object separately
  and exports calibrated 8-neighbor pixel-graph length, endpoints, connected
  junction-pixel clusters, junction pixels and isolated pixels. This is a 2-D
  pixel-graph estimate, not fitted or subpixel branch tracing.

Scope is the analysis ROI, the union of accepted cell masks, or the union of
accepted neighbour-reference objects. Cell scope requires cell counting;
reference-object scope requires neighbour analysis. These scopes are unions,
not individual-cell colocalization measurements.

## Measurement definitions

Intensity measurements use finite raw image pixels with the optional
per-channel background value subtracted and negative values clipped to zero.
Gaussian smoothing is used for segmentation, not for intensity correlation.
Each channel's own threshold and object filters define its positive mask.
Neighbour-specific exclusions do not alter the masks used for relationships.

- `pearson_r`: correlation of both channel intensities over every finite pixel
  in the selected scope. Empty or constant-intensity data give NaN, not zero.
- `manders_m1 = sum(A at positive B pixels) / sum(A)`;
  `manders_m2 = sum(B at positive A pixels) / sum(B)`.
  Denominators include all background-corrected intensities in the scope.
  These are mask-defined Manders coefficients, not Coloc 2's Costes-derived
  thresholded coefficients. Zero denominators give NaN.
- `overlap_fraction_of_a/b`: intersection divided by the respective positive
  mask area; `jaccard`: intersection/union; `dice`: twice intersection divided
  by the sum of positive mask areas.
- `nearest_centroid_distance_um`: calibrated centroid-to-centroid distance
  from A to the nearest centroid in B. Its exported B ID belongs to this
  centroid criterion, not necessarily to the target giving minimum mask distance.
- `minimum_mask_distance_um`: smallest physical pixel-center distance from
  any pixel of object A to any pixel in the B mask. Overlap gives zero.
  This is not a subpixel surface-to-surface measurement. X/Y calibration is
  applied independently. Missing targets give NaN and `target_available=False`.
- `within_proximity`: true only when a target exists and minimum mask distance
  is no greater than the selected proximity threshold.

All analyses are **2-D**, using the selected slice or projection. Projection
overlap does not establish 3-D colocalization, physical interaction or
internalization. No Costes automatic threshold, randomization significance
test, image registration or spectral bleed-through correction is performed.
Use acquisition controls, appropriate ROI/background selection and single
optical sections when required; do not treat pixels/objects as independent
biological replicates.

## Outputs and previews

In **Outputs**, enable **Intermediate pipeline images** and/or **Advanced
analysis images**. Open **Choose processing images...** to choose the exact
stages saved as PNGs and offered in the preview selector. Image-category
switches still apply. Unused stages, such as watershed seeds when splitting
is disabled, are not fabricated. Processing images include:

- raw and Gaussian images, initial thresholds, morphology masks, candidate
  labels, accepted and rejected masks for every enabled channel;
- reference morphology/exclusion, watershed distance/seeds/candidates,
  final references and physical distance maps;
- nucleus morphology, watershed distance/seeds/candidates, accepted nuclei
  and confirmed cells;
- composites, channel-object overlays, excluded-object masks, ROI boundaries,
  individual cumulative-range masks/overlays;
- pairwise overlap maps, intensity-density plots using all valid pixels,
  target-distance heatmaps and source-object distance histograms.

**Overview QC panels...** independently chooses panels for the overview figure.
The **Displayed result** selector and A/D keys navigate generated previews.

Open **Plot settings...** to change each channel/object boundary's color and
line width, common opacity, display percentiles, gamma, brightness gain,
object IDs, font size, DPI, figure background, heatmap colors and histogram
bins. Channel image colors remain in **Channels**. Boundary widths are in the
rendered PNG's pixels, not the source-image pixels. **Max image dimension px**
sets processing-image size; overview figure size is controlled by DPI/layout.

For final validation, set **Preview zoom** to the final **Analysis zoom**
(usually 1.0). A downsampled preview can change threshold masks, watershed
seeds and object counts; it is a speed aid, not a numerical guarantee of the
full-resolution run. **Max image dimension px** only changes rendering size.

After creating a preview, **Apply display settings to preview** redraws cached
analysis results without changing segmentation or measurements. After changing
thresholds, object filters, channels, ROI or advanced analyses, rerun **Preview
selected image**. A new preview selector lists only the current generated
files; previous images may remain on disk but are not reused as current results.

New tables are `advanced_metrics.csv` and `object_distances.csv`; batch outputs
also include `cell_measurements.csv`, `radial_profiles.csv`,
`spatial_objects.csv` and `skeleton_objects.csv`. Batch output adds a
`combined_*.csv` for every enabled table and matching **Advanced Metrics**,
**Object Distances**, **Cell Measurements**, **Radial Profiles**,
**Spatial Objects** and **Skeleton Objects** Excel sheets. After preview,
**Choose output parameters...** can select their exact columns, just like the
existing tables. Session/parameter YAML stores analyses, channel pairs,
backgrounds, processing-stage selection and drawing styles. Running all files
applies the same settings to the batch.

## Method references and code provenance

This feature uses already-installed SciPy/scikit-image functions. No ImageJ
plugin source was copied, and installing ImageJ/Fiji is not required.

- [ImageJ Coloc 2](https://imagej.net/plugins/coloc-2) and
  [its open-source implementation](https://github.com/fiji/Colocalisation_Analysis):
  intensity correlation, Manders definitions, scatter plots and interpretation.
- [ImageJ DiAna / Distance Analysis](https://imagej.net/plugins/distance-analysis):
  object-based overlap and directed nearest-object distances. The app does not
  claim to reproduce DiAna's 3-D surface-distance implementation.
- [scikit-image measurement API](https://scikit-image.org/docs/stable/api/skimage.measure.html):
  `pearson_corr_coeff`, `manders_coloc_coeff` and connected-object measurements.
- [SciPy distance_transform_edt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)
  and [cKDTree.query](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html):
  physical X/Y pixel sampling and nearest centroid queries.
- [scikit-image skeletonize](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize)
  and [ImageJ AnalyzeSkeleton](https://imagej.net/plugins/analyze-skeleton):
  conceptual references for skeleton pixels, endpoints and junctions. The app
  reports its documented 2-D pixel graph and does not reproduce every
  AnalyzeSkeleton pruning or branch-classification option.

The existing native-only ImageJ macro is unchanged. It does not implement
these new desktop options. The Python-bridge macro can run the same saved
session and Python engine.
