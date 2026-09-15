CELL ANALYZER NATIVE FIJI MACRO
================================

This is a separate, native Fiji/ImageJ workflow. It does not start Python and
does not use the Cell Analyzer desktop application.

INSTALL
-------
1. Start Fiji.
2. Choose Plugins > Macros > Install...
3. Select CellAnalyzer_Native_Fiji.ijm.
4. Two commands appear under Plugins > Macros:
   - Cell Analyzer Native
   - Cell Analyzer Native Self-Test

FIRST CHECK
-----------
Run Cell Analyzer Native Self-Test. A passing installation reports that two
synthetic objects were detected.

ANALYZE AN IMAGE
----------------
1. Open the microscopy file in Fiji. Bio-Formats can be used to open Leica,
   Olympus, Zeiss, Nikon, and other proprietary formats.
2. The active image must be a 2D multichannel image with one Z plane and one
   time point. For a Z stack, first choose Image > Stacks > Z Project.
3. Run Plugins > Macros > Cell Analyzer Native.
4. Choose an output folder. Nothing is cached by this macro.
5. Map up to four image-channel indices. Set a name, wavelength, and built-in
   Fiji display color for each enabled channel.
6. Choose a manual or automatic threshold and Gaussian blur for each channel.
7. Set independent area and circularity filters. Mask cleanup settings are
   shared by all enabled channels.
8. Keep work images open on the first run to inspect masks and boundaries.
   Adjust the values and run again when needed.

OUTPUTS
-------
For every enabled channel, the macro saves:
- optional grayscale raw-channel TIFF
- color PNG using the selected Fiji LUT
- binary-mask TIFF
- CSV object measurements
- ROI Manager ZIP when objects are found
- color PNG with adjustable yellow object boundaries

It also saves one run-summary CSV and one parameter TXT file beside the image
outputs. All files are written only to the output folder selected by the user.

SCOPE
-----
The native macro intentionally uses ImageJ's built-in thresholding,
morphology, ROI Manager, and Analyze Particles commands. It supports object
area, circularity, intensity measurements, channel colors, and counts.

The desktop Cell Analyzer remains the version for advanced cross-channel
logic such as ring measurements, distance-to-object cell counts, solidity,
multi-condition gating, spreadsheet workbooks, session reload, and batch
processing across many files and folders.
