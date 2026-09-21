Cell Analyzer

1. Open CellAnalyzer\CellAnalyzer.exe.
2. Select a Cache folder before preview or analysis. The output folder is required only for Run all images, not preview.
3. Add CZI, LIF, OIR, VSI, ND2, OME-TIFF, or other supported microscopy images. Images may come from multiple folders.
4. Inspect one image, then assign image channels to Channel 1-4. Set each channel's name, wavelength, and display color.
5. Adjust the parameters, open the Outputs tab, and select "Preview selected image". The processed preview is shown in the same tab. Use an analysis zoom of 1.0 for final measurements.
6. After preview, select "Choose output parameters..." to choose the exact columns included in each CSV and Excel sheet. Select the images and masks to save, then select "Run all images". The current settings apply to every selected file.
7. "Save session" stores the image paths, output folder, cache folder, parameters, and selected table columns for later reuse.

No cache is intentionally written to the system drive. The application checks the cache selection only when preview or analysis starts.

Fiji/ImageJ Python bridge

In Fiji, select Plugins > Macros > Run and open ImageJ_Macro\CellAnalyzer.ijm.
The macro asks for the project folder and a session YAML saved by the desktop application, then runs the same Python analysis pipeline.

Fiji/ImageJ native-only macro

For a workflow that never starts Python, choose Plugins > Macros > Install and open ImageJ_Macro\CellAnalyzer_Native_Fiji.ijm.
Run "Cell Analyzer Native Self-Test" first, then run "Cell Analyzer Native" on the active 2D multichannel image.
See ImageJ_Macro\CellAnalyzer_Native_Fiji_README.txt for the workflow, outputs, and scope.


Advanced analyses and drawing controls

Cell Counting now appears before Advanced Analysis. Open Advanced Analysis and
independently enable Neighbour analysis, Colocalization, Object distances,
Per-cell measurements, Radial profiles, Spatial distribution, or Skeleton
analysis. Add channel pairs and select the measurement
scope in Colocalization & distances. New sessions leave advanced analyses off.
Old saved sessions retain the original neighbour workflow.

In Outputs, enable Intermediate pipeline images or Advanced analysis images.
Choose processing images... selects the stages saved and listed in the preview.
Overview QC panels... selects panels for the overview figure independently.
Plot settings... controls each boundary color/width, common opacity, display
percentiles, gamma, brightness, object IDs, font size, DPI, figure background,
heatmap colors and histogram bins. Apply display settings to preview redraws
cached results without re-running segmentation or changing measurements.
Use A/D keys to navigate generated previews.

Advanced Metrics, Object Distances, Cell Measurements, Radial Profiles,
Spatial Objects and Skeleton Objects are available in CSV and Excel; their
columns can be chosen after preview. Parameters and sessions save all advanced
and drawing settings for application to every file.

See ADVANCED_ANALYSIS.md for definitions, ImageJ/open-source references and
limitations. These are 2-D analyses, not proof of internalization. No Costes
automatic threshold or randomization significance test is implemented.
The existing native-only ImageJ macro is unchanged.
