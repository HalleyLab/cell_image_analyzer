# Cell Analyzer — cell-ROI workflow

This workflow detects cell boundaries in one selected microscopy channel, creates reusable ROIs, and measures every image channel inside each ROI. It supports CZI, grayscale PNG, and Bio-Formats microscopy inputs. The original `morphology.ipynb` and `feature_functions.py` files are read-only references and are not modified.

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
- Filters every channel independently and sets sub-threshold pixels to zero for intensity measurements.
- Reports ROI area, shape, raw mean, thresholded whole-ROI mean, and positive-only mean intensity for every channel.
- Reports integrated fluorescence intensity for every channel and ROI.
- Reports signal-positive area and positive fraction for every channel inside every ROI.
- Exports ImageJ ROI ZIP, GeoJSON polygons, a 32-bit label TIFF, CSV, Excel, and QC previews.

## Installation

Use Python 3.10 through 3.14. Python 3.12 is recommended.

```powershell
cd E:\projects\cell_analyzer
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

Click **Add images** to choose multiple PNG files, or paste one absolute `.png` path per line and click **Apply file list**. Use **Preview file**, **Previous**, and **Next** to inspect every image. PNG input is treated as one fixed intensity channel, so the parameter-channel and segmentation-channel selectors are hidden. Each image remembers its own tuned settings. Use **Apply current settings to all** when the entire batch should share the current parameters, then click **Load preview** to read the selected image.

The **Cell segmentation and size filters** section opens by default. Edit **Minimum cell area** and **Maximum cell area** in square micrometers to reject cells outside the required physical-area range; the accepted-boundary preview and the `rejected by area` count update immediately. The eight preview panels show raw intensity, Gaussian smoothing, segmentation input, initial threshold, morphology cleanup, watershed candidates, accepted numbered cell boundaries, and signal-positive pixels. When the settings are satisfactory, choose an output root and click **Run all files**. A failed or incompatible file is recorded in the batch summary while the remaining files continue.

The original single-file interface remains available through `launch_tuning_widget(config)`. Batch analysis can also be started without the widget:

```python
from cell_analyzer.batch import run_batch_analysis

batch_result = run_batch_analysis(IMAGE_PATHS, config, OUTPUT_ROOT)
```

The live panel shows only controls used by the selected method. Watershed controls appear when splitting is enabled; manual/percentile/adaptive threshold fields appear only for their respective methods. Minimum peak height, compactness, small-hole filtering, and local contrast remain available in the collapsed **Advanced filters (optional)** section. Existing configurations retain their values.

The **Active advanced filters** line reports which folded settings are still active. Preview diagnostics include candidate rejection counts for area, circularity, local contrast, and border position. Small components removed during morphology cleanup are not included in those later candidate-rejection counts.

Batch templates now retain each channel's original metadata name (`source_name`), independently of its editable output alias. Unique, informative channel names are matched across files, including the segmentation channel. Generic names such as `Channel_0`, duplicate names, or unmatched names require **Confirm same channel order (ambiguous metadata)** before applying parameters by index. Enable this only after checking the images. A different channel count requires a separate per-file configuration; the program never silently substitutes the first channel. Existing per-file configurations and single-channel images remain supported.

At the same image resolution, preview and analysis retain the same floating-point watershed height/prominence values. A downsampled preview is still approximate: resampling can change intensities, automatic thresholds, morphology, and counts. To inspect the analysis resolution, pass `max_preview_dimension` at least as large as the source image's longest side to the notebook launcher. Use `input.zoom=1.0` for final raw-intensity quantification when memory permits.

### Native Fiji version of the cell-ROI workflow

Install `CellAnalyzer_App/ImageJ_Macro/CellAnalyzer_ROI_Fiji.ijm` through **Plugins > Macros > Install**. It uses one selected channel to define ROIs and measures all detected channels in those same ROIs. It supports an active-image preview, a file-list batch across folders, reusable parameter TXT files, calibrated areas, zero-signal rows, and processing images. It does not launch Python. See [the Fiji guide](CellAnalyzer_App/ImageJ_Macro/CellAnalyzer_ROI_Fiji_README.md) for usage, self-test, and differences from the Python algorithms.

## Important parameters

### Input

- `image_path`: PNG or CZI input path. Legacy CZI configurations may keep using `czi_path`.
- `image_width_um` and `image_height_um`: total physical width and height of the entire PNG in micrometers. The program divides these by the PNG pixel dimensions to calculate X/Y `µm/pixel`.
- `scene`, `time_index`, `z_projection`, and `z_index`: CZI plane settings; PNG uses fixed zero indices.
- `zoom`: image read scale from `0.01` to `1.0`. Full resolution is `1.0`.
- `segmentation_channel`: CZI ROI channel; PNG is automatically fixed to channel `0`.

When `zoom` is below `1.0`, areas in source pixels and calibrated square micrometers are corrected for the scale. Measurements are still made from the image data read at that zoom, so use `1.0` for final quantitative work when memory allows.

PNG files do not reliably preserve microscopy calibration. Enter the total physical width and height before interpreting square-micrometer areas. All PNGs in one batch are assumed to share those physical dimensions; CZI files continue to use their own metadata.

### Per-channel analysis

- `gaussian_sigma_px`: optional denoising scale applied directly to raw image intensities. Use `0` to disable smoothing.
- `measurement_threshold.method`: `none`, `manual`, `otsu`, `yen`, `triangle`, or `percentile`.
- `measurement_threshold.value`: exact channel intensity cutoff used by `manual`.
- `measurement_threshold.percentile`: used by `percentile`.
- `measurement_threshold.scale`: independent automatic-threshold multiplier for the channel. Values below 1 retain more pixels; values above 1 retain fewer pixels.

For an exact cutoff, choose `manual` and enter `value`; the automatic multiplier is ignored in that mode. The threshold is applied to the Gaussian-smoothed image in the original intensity scale. Pixels below it are assigned intensity zero before the raw-image mean and integrated intensity are calculated. Selecting `none` disables filtering. Thresholding never removes an ROI row from the output, even when all measured values in that row are zero.

### Segmentation

- `threshold_method`: global or adaptive threshold used on the selected segmentation channel.
- `threshold_scale`: multiplier for automatic global thresholds. Values below 1 expand the detected mask; values above 1 make it stricter.
- `min_area_um2` and `max_area_um2`: accepted physical cell area in square micrometers. CZI metadata or explicit PNG X/Y calibration converts these limits to analysis pixels at each zoom.
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

Each input image receives its own subdirectory below the selected batch output root. Every subdirectory contains the normal per-file outputs listed below.

- `roi_measurements.csv`: one row per ROI.
- `roi_measurements.xlsx`: measurements, summary, channels, and all parameters.
- Per-channel columns include raw mean, thresholded whole-ROI mean (`mean_intensity`, the unchanged legacy column), positive-only mean, integrated intensity, positive area, and positive fraction. When no pixels are positive, `positive_mean_intensity` is undefined (`NaN` / an empty Excel cell); existing thresholded mean, sum, area, and fraction remain zero. Raw mean still includes the original background.
- Per-channel positive-area pixel columns are intentionally omitted. ROI rows containing zero values are retained.
- `roi_labels.tiff`: 32-bit ROI label image; background is zero.
- `imagej_rois.zip`: polygon ROIs for Fiji/ImageJ.
- `rois.geojson`: polygon ROIs in analysis-pixel coordinates.
- `roi_overlay.png`: selected-channel preview with red ROI boundaries.
- `channel_previews.png`: display-scaled preview of every channel; analysis still uses raw intensity values.
- `config_used.yaml`: exact validated configuration used for the run.
- `image_metadata.json` for PNG or `czi_metadata.json` for CZI: standardized input metadata.
- `analysis_summary.json`: ROI count, diagnostics, warnings, and output paths.

The batch output root also contains:

- `selected_image_files.txt`: absolute path of every selected image, in processing order.
- `batch_parameters.yaml`: shared template, per-file overrides, and the effective validated configuration used for every prepared file.
- `batch_summary.csv`, `batch_summary.xlsx`, and `batch_analysis_summary.json`: completion state, ROI count, output directory, and error message for every input file.
- `combined_roi_measurements.csv` and `combined_roi_measurements.xlsx`: all ROI measurements combined across successfully analyzed files, with source-file columns.

## Parameter tuning strategy

Start at `zoom: 0.25` for fast tuning. Inspect `roi_overlay.png`, then adjust the segmentation threshold. Use minimum area to suppress noise, peak distance to control watershed splitting, and local contrast to reject weak objects. Physical area limits remain unchanged when zoom changes. Switch to `zoom: 1.0` for final intensity quantification when memory allows, and scale the remaining pixel-based segmentation parameters proportionally.

## 参数、公式、包与函数（cell-ROI 流程）

本节对应 `cell_analyzer/*.py`、两个细胞 ROI Notebook 和 `CellAnalyzer_ROI_Fiji.ijm`，不是 `brain_section_analyzer` 独立 EXE 的高级分析参数。软件和宏对话框仍使用英文。以下列出所有公开 ROI 配置键；路径、文件开关等不参与数值计算的选项明确标为“无数学公式”，不为它们虚构算法。

### 符号、单位与计算顺序

- `R`：某个细胞 ROI；`N=|R|`：ROI 在**分析分辨率**下的像素数。
- `I(p)`：原始读入强度；`S(p)`：Gaussian 平滑后的强度；`B(p)`：根据 `S` 判断的阳性掩膜（0 或 1）。
- `N+=Σ_R B(p)`；`J=Σ_R I(p)B(p)`；`s_x,s_y`：原图 µm/pixel；`z`：`input.zoom`。
- 一个分析像素的面积 `a=(s_x/z)(s_y/z)` µm²；物理 ROI 面积 `A=N·a`。不同缩放下，像素强度总和并不因为面积校正而变成完全相同。
- 顺序：读图/投影 → 各通道平滑 → 指定通道分割 → 清理掩膜 → 可选分水岭 → 候选过滤 → 同一套 ROI 测量所有通道。显示拉伸不进入这些计算。

### 输入、标定与批量设置

| 参数 | 单位 / 规则或公式 | 包与具体函数 / 本地入口 |
|---|---|---|
| `input.image_path`；旧键 `czi_path` | 文件路径，无数学公式；优先使用 `image_path` | `pathlib.Path`；`image_io.inspect_image/read_image_channels`；CZI：`pylibCZIrw.czi.open_czi`；PNG：`PIL.Image.open`；其它厂商：`bioio.BioImage(reader=bioio_bioformats.Reader)`、`BioImage.get_image_data`；Java 桥：`scyjava` |
| `input.output_dir`；批量 `output_root` | 输出目录，无数学公式 | `Path.mkdir`；`pipeline.run_analysis`、`batch.run_batch_analysis` |
| `input.scene` | 0 起始场景编号；选择场景而非平均所有场景 | `czi_io.read_czi_channels`；`image_io.read_image_channels` |
| `input.time_index` | 0 起始时间点；不平均多个 T | 同上；CZI reader 的 `plane['T']` |
| `input.z_projection` | `single`: `I=I_k`；`max`: `I(p)=max_k I_k(p)`；`mean`: `I(p)=Σ_k I_k(p)/K` | `numpy.maximum`、浮点累加/除法；`czi_io.read_czi_channels` / Bio-Formats read 路径 |
| `input.z_index` | 0 起始 Z 编号；只在 `single` 时使用 | `czi_io._dimension_indices/read_czi_channels`；`image_io.read_image_channels` |
| `input.zoom` | 0.01–1；读入空间缩放；面积换算见上文 | CZI：reader `read(zoom=z)`；PNG/Bio-Formats：`skimage.transform.resize`；`config.effective_pixel_area_um2` |
| `input.pixel_size_um_x`；`input.pixel_size_um_y` | µm/pixel；无可靠元数据时显式标定。CZI 始终使用自身元数据 | `image_io._validated_pixel_size`、`czi_io.inspect_czi` |
| `input.image_width_um`；`input.image_height_um` | 整幅图的物理宽/高；`s_x=W_um/W_px`、`s_y=H_um/H_px`；PNG 优先采用它们；Bio-Formats 仅在缺失标定时使用 | `image_io._pixel_size_from_total`、`_inspect_bioformats` |
| `input.segmentation_channel` | 0 起始通道编号；决定 ROI，**不**改变其它测量通道 | `pipeline.run_analysis`；`interactive.LiveTuningPanel` |
| `input.confirm_channel_order` | 布尔值；仅在身份无法唯一匹配时，授权按相同编号复用；无数学公式 | `batch.prepare_batch_config`；Notebook 同名确认框 |
| `channels.<index>.source_name` | 只读原始通道名称；与可编辑别名分离；去首尾空白、忽略大小写作唯一身份匹配 | `config.create_default_config/normalize_config`；`batch.prepare_batch_config` |
| `channels.<index>.alias` | 输出列名前缀，无强度变换；需生成唯一列名 | `config.slugify`（标准库 `re`）；`measurements.measure_rois` |
| `image_paths` / `czi_paths`；`per_file_configs` | 批量路径及逐文件覆盖；无数学公式；相同路径去重，错误逐文件记录 | `batch._normalize_paths/run_batch_analysis`；`interactive.BatchTuningPanel` |
| `continue_on_error`；`progress` | 失败时是否继续、进度回调；无数学公式 | `batch.run_batch_analysis` / `pipeline.run_analysis` |
| `CELL_ANALYZER_CACHE_DIR` | 环境变量，不是 YAML 参数；控制 ROI Python 缓存根目录；临时、CJDK、Matplotlib 子目录随之设置。选 E/I 盘，不选 C 盘 | `cell_analyzer.__init__`；`os.environ`、`tempfile.tempdir`、`Path.mkdir` |

### 每通道平滑与阳性标准

| 参数 | 公式 / 生效条件 | 包与函数 / 本地入口 |
|---|---|---|
| `gaussian_sigma_px` | `S=G_σ*I`；`G_σ(x,y)∝exp[-(x²+y²)/(2σ²)]`，归一化后卷积；0 不平滑。σ 是分析图像像素单位 | `scipy.ndimage.gaussian_filter`（默认 reflect 边界、truncate=4）；`preprocessing.preprocess_channel_steps` |
| `measurement_threshold.method` | `manual/none/otsu/yen/triangle/percentile`；阈值算法详见下表 | `segmentation.threshold_image`；`measurements._measurement_mask` |
| `measurement_threshold.value` | Manual：`B=[S≥T]`，`T=value`；保留等于阈值的像素；此时不使用 scale | `numpy` 比较；`segmentation.threshold_image` |
| `measurement_threshold.scale` | 自动全局阈值：`T=scale·T₀`；一般 `B=[S>T]`；不用于 manual/none | `skimage.filters` / `numpy.percentile`；`segmentation.threshold_image` |
| `measurement_threshold.percentile` | 0–100；仅 percentile：`T₀=Q_q(V)`；分位数线性插值 | `numpy.percentile`；`segmentation.threshold_image` |

`None` 将整个 ROI 纳入测量，`B=1`，不是“把所有信号判为阴性”。原始图转换为 float32 用于平滑，并将平滑输入的 NaN/Inf 置零；定量强度仍读取原始图。输入应为有限值的原始显微图像。

### 自动阈值算法的实际定义

Python 的自动全局阈值先取有限像素集合；若正值像素至少 32 个，则用正值集合 `V`，否则用全部有限像素。不是直接把整幅含大量零背景的直方图交给库。自动直方图函数使用库默认分箱；当前平滑输入为 float32，默认 256 bins。Adaptive 使用二维图像，而不是这个一维集合。

| 方法 | 公式 / 判定 | 包与函数 |
|---|---|---|
| Otsu | `T₀=argmax_t ω₀(t)ω₁(t)[μ₀(t)-μ₁(t)]²`，最大化两类之间的方差 | `skimage.filters.threshold_otsu(V)` |
| Yen | 对归一化直方图 `p_i`，令 `P(t)=Σ_{i≤t}p_i`、`Q₀(t)=Σ_{i≤t}p_i²`、`Q₁(t)=Σ_{i>t}p_i²`；最大化 `log{[P(t)(1-P(t))]²/[Q₀(t)Q₁(t)]}`，取相应 bin 中心 | `skimage.filters.threshold_yen(V)` |
| Triangle | 使用直方图峰到较长尾端的基线；选离基线最远的 bin。库实现先按尾方向翻转，再最大化归一化的 `h_peak·x-width·h(x)` | `skimage.filters.threshold_triangle(V)` |
| Percentile | `T₀=Q_q(V)`；q 为所选百分位 | `numpy.percentile(V,q)` |
| Adaptive | `T(x,y)=局部 Gaussian 加权均值-offset`；`B=[S>T(x,y)]`；block size 转成至少 3 的奇数；global scale 不生效 | `skimage.filters.threshold_local(S, block_size, offset=...)`（默认 method='gaussian'） |
| 空图 / 常数图 | 集合为空或全相等时不调用上述全局库算法；自动方法将常数乘 scale，`T>0` 时用 `S≥T`，否则用 `S>T`，避免零背景全部阳性 | `segmentation.threshold_image` 的提前分支 |

函数依据：[scikit-image 阈值 API](https://scikit-image.org/docs/stable/api/skimage.filters.html)；Gaussian 实现依据：[SciPy gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)。程序自有的正值集合筛选、scale 和边界比较规则以上述本地实现为准。

### 分割、清理、分水岭和候选过滤

| 参数 | 单位 / 公式 / 判定 | 包与具体函数 / 本地入口 |
|---|---|---|
| `threshold_method` | `otsu/yen/triangle/percentile/adaptive`；ROI Python 分割不提供 Manual；Fiji 提供 Manual | `segmentation.threshold_image`；对应上表各函数 |
| `threshold_scale` | `T=scale·T₀`；仅自动全局方法；adaptive 不使用 | 同上 |
| `threshold_percentile` | `T₀=Q_q(V)`；只在 percentile 生效 | `numpy.percentile` |
| `adaptive_block_size_px` | Gaussian 局部窗口边长；至少 3，偶数加 1 变奇数 | `skimage.filters.threshold_local` |
| `adaptive_offset` | 原始强度单位；`T=local_mean-offset`，增大 offset 一般扩大阳性掩膜 | 同上 |
| `invert` | `B←¬B`；只改变分割前景方向，不反转测量通道 | `numpy` 布尔取反；`segmentation.threshold_image` |
| `opening_radius_px` | 整数像素；圆盘结构元 `D_r`；开运算 `B∘D_r=(B⊖D_r)⊕D_r`；0 关闭 | `skimage.morphology.disk/opening`；`_morphological_cleanup` |
| `closing_radius_px` | 整数像素；闭运算 `B•D_r=(B⊕D_r)⊖D_r`；0 关闭 | `skimage.morphology.disk/closing`；`_morphological_cleanup` |
| `fill_all_holes` | true 填充所有不连通图像边界的内部背景洞；此时忽略 min_hole_area | `scipy.ndimage.binary_fill_holes` |
| `min_hole_area_px` | px²；仅 fill_all_holes=false：填充面积**小于**此值且不连接边界的背景连通域；0 关闭 | `scipy.ndimage.label`、`numpy.bincount/setdiff1d/isin`；`_morphological_cleanup` |
| `min_area_um2` | µm²；`N_min=max(1,ceil(A_min/a))`；清理时先移除过小连通域，分水岭后再过滤候选面积 | `config.segmentation_config_for_zoom`；`scipy.ndimage.label`、`skimage.measure.regionprops` |
| `max_area_um2` | µm²；`N_max=floor(A_max/a)`；null 不限；分水岭后要求 `N≤N_max` | 同上；`segmentation._filter_regions` |
| `min_circularity` | `C=4πN/P_px²`；保留 `C≥C_min`；周长为 0 时按 0 处理。栅格周长估计可导致 C>1，不能当作精确几何圆度 | `skimage.measure.regionprops(...).perimeter`；`_filter_regions` |
| `min_local_contrast_ratio` | `contrast=(mean_inside+10⁻⁸)/(mean_ring+10⁻⁸)`；保留 ratio≥参数值；0 不限制。这里使用平滑图，不是背景扣除后的原图 | `numpy.mean`；`segmentation._local_contrast_ratio/_filter_regions` |
| `local_contrast_ring_px` | 整数像素，内部至少 1；`ring=dilate(ROI,D_r)\ROI`；环超出图像时裁剪；环中其它细胞不自动排除 | `skimage.morphology.disk/dilation`；`_local_contrast_ratio` |
| `clear_border` | true 才启用边界及边距排除 | `segmentation._filter_regions`；基于 regionprops bbox 比较，不调用 clear_border 函数 |
| `border_exclusion_margin_px` | 像素 m；若 bbox 的 min_row/min_col≤m 或 max_row≥H-m 或 max_col≥W-m 则排除；max 坐标为半开边界 | `skimage.measure.regionprops(...).bbox`；`_filter_regions` |
| `split_touching` | false 用连通域；true 才计算距离图、种子和分水岭 | `scipy.ndimage.label/distance_transform_edt`；`_watershed_labels` |
| `min_peak_distance_px` | 整数像素，至少 1；限制峰的邻近程度；库默认按 Chebyshev 距离（p_norm=∞）抑制邻近峰，不是两细胞边缘距离 | `skimage.feature.peak_local_max(min_distance=...)` |
| `watershed_min_peak_height_px` | 浮点像素；`D(p)=min_{q∈背景}||p-q||₂`，只接受 D 高于绝对阈值的候选峰；预览缩放保留小数 | `scipy.ndimage.distance_transform_edt`；`peak_local_max(threshold_abs=...)` |
| `watershed_min_peak_prominence_px` | 浮点像素；h>0 时，只保留距离图中高度至少为 h 的 h-maxima；这是相对突出度，不是荧光阈值 | `skimage.morphology.h_maxima(D,h)`；`_watershed_labels` |
| `watershed_compactness` | 非负 compact-watershed 系数；0 标准分水岭，增大偏向紧凑形状；不是面积/圆度过滤，也不是直接最小化本文某个圆度公式 | `skimage.segmentation.watershed(-D,markers,mask=B,compactness=...)` |

Python 连通域默认使用 SciPy 的二维 4-连通。峰过滤后，程序仍为每个前景连通域保证至少一个种子：因此很大的 minimum height/prominence 不会自动删除整个细胞，它们主要控制一个连通对象拆成多少个 ROI。候选按**边界 → 面积 → 圆度 → 对比度**顺序过滤，每个候选只记录第一个失败原因。

分水岭库入口：[scikit-image watershed](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed)。本流程没有用 solidity 过滤，不能把其它软件页面上的 solidity 当成这里的 circularity。

### 表格指标：三种 mean 不能混用

| 输出列（每通道前缀省略） | 公式 / 单位 | 包与函数 |
|---|---|---|
| `raw_mean_intensity` | `Σ_R I/N`；原始强度单位；包含阈值下的像素和背景 | `scipy.ndimage.mean(raw,labels,index)`；`measurements.measure_rois` |
| `mean_intensity`（保留旧列名） | `J/N`；阈值下置零后，仍除以整个 ROI 像素数 | `numpy.where`、`scipy.ndimage.mean` |
| `positive_mean_intensity` | `J/N+`；仅阳性像素的原始平均强度；N+=0 时 NaN，不伪造为 0 | `numpy.divide(...,where=N+>0)` |
| `integrated_intensity` | `J=Σ_R I·B`；强度×像素；不是 µm² 校正积分 | `scipy.ndimage.sum` |
| `positive_area_um2` | `N+·a`；µm² | `numpy.bincount`、面积标定乘法 |
| `positive_fraction` | `N+/N`；0–1，无单位；不是百分数 | `numpy.bincount/divide` |
| `roi_area_analysis_px` | `N`；分析像素数量（面积） | `skimage.measure.regionprops.area` |
| `roi_area_source_px` | `N/z²`；原图等效像素面积 | `measurements.measure_rois` |
| `roi_area_um2` | `N·a`；µm² | 同上 |
| `perimeter_analysis_px`；`circularity` | 栅格周长 `P_px`；`4πN/P_px²` | `skimage.measure.regionprops` |
| `roi_id`；ROI count | 接受后的整数标签；count 为标签最大值（过滤后重新连续编号） | `numpy`；`segmentation._filter_regions` |

例：100 像素的细胞中，50 像素强度 1000、另 50 像素强度 100，Manual 阈值 500：原始均值=550，旧 mean=500，阳性区域均值=1000，positive_fraction=0.5。细胞面积变化会影响旧 mean，即使阳性区域亮度不变。若没有阳性像素，旧 mean/sum/positive area/fraction 仍为 0；`positive_mean` 为 NaN；ROI 行不会删除。

### 显示、导出与复用参数

| 参数 | 作用 / 公式 | 包与具体函数 |
|---|---|---|
| Notebook `max_preview_dimension` | 预览最大边长；`preview_zoom=max(0.01,min(analysis_zoom,L/max(W,H),1))` | `interactive.LiveTuningPanel.__init__`；原图读取函数 |
| 预览像素参数缩放 | `r=preview_zoom/analysis_zoom`；长度乘 r；孔洞面积乘 r²；整数核半径取整；峰高/突出度保留浮点；物理面积过滤重新换算 | `interactive._preview_channel_config/_preview_segmentation_config`；`config.segmentation_config_for_zoom` |
| 显示强度拉伸（固定规则） | `(I-Q₁)/(Q₉₉.₈-Q₁)` 后裁剪到 [0,1]；足够多正像素时用正值分位数；仅 Notebook 显示，不用于计算 | `numpy.percentile/clip`；`interactive._display_scale` |
| `output.preview_max_dimension_px` | 仅保存的 QC 图尺寸上限；不改变 ROI 或测量；overlay 线性重采样强度、最近邻重采样标签 | `rois.save_overlay`：`skimage.transform.resize(order=1/0)`；`pipeline._save_channel_panel`：步长抽样 |
| `output.roi_simplify_tolerance_px` | 导出矢量轮廓的像素容差；0 不简化；不重新测量或修改标签表 | `skimage.measure.find_contours/approximate_polygon`；`rois._largest_contour` |
| `output.save_label_image` | 保存 32-bit 标签 TIFF，背景 0；无数学公式 | `tifffile.imwrite`；`rois.export_rois` |
| `output.save_imagej_rois` | 保存 ImageJ ROI ZIP；无数学公式 | `roifile.ImagejRoi.frompoints/roiwrite` |
| `output.save_geojson` | 保存分析像素坐标的闭合多边形；无数学公式 | 标准库 `json.dumps`、`Path.write_text` |
| CSV / XLSX / YAML（固定输出） | 导出数据、参数、元数据；不改变定量 | `pandas.DataFrame.to_csv/to_excel`、`openpyxl`、`yaml.safe_dump`；`pipeline` / `batch` |

### Fiji 参数对应与实现差异

Fiji 原生宏不启动 Python、不需要缓存。全部通道使用同一套细胞 ROI；下表覆盖宏参数 TXT 的每个分析/显示键。模式、载入参数、保存预览参数和路径文件选项均为文件/交互操作，没有数学公式。宏入口为 `runWorkflow/configure/analyzeImage`；对话框使用 ImageJ `Dialog.*`、`getNumber`，路径使用 `File.*`。

| Fiji 参数键 | 公式 / Python 对应 | ImageJ 内置函数 / 命令 |
|---|---|---|
| `workflow`；`channel_count` | 流程版本标记及实际通道数量，只读；不参与强度计算 | `getDimensions`；`configure` |
| `fallback_pixel_x_um`；`fallback_pixel_y_um`；逐文件 `pixel_size_x_um`；`pixel_size_y_um` | µm/pixel；优先每文件标定，缺失才用 fallback；`a=s_xs_y`。混合不同倍率的未标定图不能共用一个 fallback | `getPixelSize`、`setVoxelSize`；`readCalibration` |
| `source_file`；路径 TXT | 输入标识及文件清单，无公式；批量必须显式确认相同通道顺序，宏不宣称能自动匹配厂商身份 | `open` 或 `run('Bio-Formats Importer',...)`；`openSource` |
| `segmentation_channel` | 1 起始通道；与 Python 的 0 起始编号不同 | `Duplicate... channels=...`；`analyzeImage` |
| `cell_method`；`cell_value`；`cell_scale` | Manual：`S≥value`；自动 cutoff=库阈值×scale；一般严格 `S>cutoff`；自动常数图及零背景作特殊处理 | `setAutoThreshold(method+' dark')`、`getThreshold`、`setThreshold`、`Convert to Mask`；`makeMask` |
| `min_area_um2`；`max_area_um2` | µm²；0 最大面积表示不限；最小/最大范围由粒子分析校准面积过滤 | `Analyze Particles... size=min-max` |
| `min_circularity` | ImageJ 圆度 `4πA/P²`；数值由 ImageJ 的周长和校准实现决定，不保证等于 Python 栅格周长结果 | `Analyze Particles... circularity=min-1.0`；`List.setMeasurements` |
| `open_radius_px`；`close_radius_px` | 与开/闭运算相同次序；核由 ImageJ Minimum/Maximum 实现，不能假设与 skimage.disk 像素完全相同 | `Minimum...` → `Maximum...` / 反序 |
| `fill_holes` | 填充内部孔洞；宏没有 Python 的小孔面积分级选项 | `Fill Holes` |
| `split_cells`；`watershed_prominence_px` | 距离图 h-maxima 分割；h 增大通常减少分割；不是 Python min_peak_distance | `Distance Map`；`Find Maxima... prominence=h output=[Segmented Particles]`；`splitMask` |
| `exclude_edges`；`border_margin_px` | exclude 开关排除触边粒子；额外 m 像素再按 bbox 判定 | `Analyze Particles... exclude`、`getSelectionBounds`；`filterCandidates` |
| `min_local_contrast`；`contrast_ring_px` | `(innerMean+10⁻⁸)/(ringMean+10⁻⁸)`；`ringMean=(N_outer·mean_outer-N_inner·mean_inner)/(N_outer-N_inner)`；0 关闭；没有可用环时跳过该判定 | `Enlarge...`、`getRawStatistics`；`filterCandidates` |
| `z_projection`；`z_plane`；`time_point` | Max/Mean/Single 对应 Python 公式；Z/T 编号从 1 开始 | `Duplicate... slices=... frames=...`；`Z Project...` |
| `C<n>_name`；`C<n>_color` | 名称只影响表头；LUT 色只影响显示，不改变灰度定量 | `run(LUT)`、`RGB Color`；`configure/saveBoundaries` |
| `C<n>_sigma` | Gaussian σ，像素；0 关闭；核与边界处理按 ImageJ 实现，不保证与 SciPy 完全相同 | `Gaussian Blur... sigma=...` |
| `C<n>_signal_method`；`C<n>_signal_value`；`C<n>_signal_scale` | None：B=1；Manual/自动阈值见 cell_method；宏没有 Python percentile/adaptive 选项 | `makeMask` 中的 `setAutoThreshold/setThreshold/Convert to Mask` |
| `boundary_color`；`boundary_width_px` | 名称或 #RRGGBB、绘制前像素线宽；仅 QC，montage 缩放后显示线宽也会缩放 | `setColor/setLineWidth`、`roiManager('select')`、`Draw` |
| `save_stages` | 是否保存原始、平滑、阳性及分割阶段 TIFF；不改变分析 | `saveAs('Tiff',...)`；`saveImage` |

Fiji 三种 mean 和阳性面积/比例的数学定义与上方表格相同；实现使用 `getRawStatistics`，掩膜 255 转 1 后 `imageCalculator('Multiply create 32-bit',...)`；正像素数由 `N·maskMean/255` 得到。`positive_mean_intensity` 在无阳性时为 NaN。Fiji 另输出校准周长 `perimeter_um`，Python 原有周长列仍是分析像素。

Fiji 预览先显示分割 Overview；点击 OK 后选择任一测量通道，可看 Raw、Smoothed、Positive mask 和带细胞边界的 ROI 内阳性信号；通过 `makeSignalOverview` 使用 `imageCalculator`、LUT、`Draw` 和 `Make Montage...` 生成。只在选择时生成该通道视图。完整原生函数参考：[ImageJ 宏函数](https://imagej.net/ij/developer/macro/functions.html)。

### 可复现性与验证限制

本次核对的 Python 环境：NumPy 2.4.6、SciPy 1.18.0、scikit-image 0.26.0；这不是强制依赖锁定。保存 `config_used.yaml`、批量路径、实际阈值、库版本与原图；跨版本或跨 Python/Fiji 实现，自动阈值、连通性、周长和分水岭边界可能不同。组间比较固定同一个实现，不能仅因公式相同就混用两种实现的 ROI。

缓存验证：完整测试使用本机现有的 E 盘缓存。另测 I 盘新建 Bio-Formats 缓存时，Java 依赖环境建立硬链接失败（WinError 1）；此为依赖缓存初始化限制，本次没有修改读取模块。当前建议使用已验证的 E 盘缓存，不要把此失败误判为图像或 ROI 算法不支持。原生 Fiji 宏不经过这套 Python Java 缓存。

回归测试覆盖浮点预览参数、通道换序/自定义别名、模糊身份确认、三种 mean、零阳性、已有 CSV/Excel 和批量参数复用。Fiji 自测还覆盖实际五通道 TIFF、标定、Z/T 投影、粘连分割和逐通道预览。它们证明计算分支可运行，不证明真实细胞识别准确；先人工核对代表性视野，再冻结参数批量分析。

## Tests

The test suite covers the three-channel CZI workflow and the fixed-single-channel PNG batch workflow, including combined measurements and hidden channel selectors.

```powershell
python -m unittest discover -s tests -v
```

## CZI reader references

- [ZEISS pylibCZIrw API](https://zeiss.github.io/pylibczirw/)
- [ZEISS CZI image format](https://www.zeiss.com/microscopy/en/products/software/zeiss-zen/czi-image-file-format.html)
