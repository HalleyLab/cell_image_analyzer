# 细胞 ROI 分析：Fiji 宏

使用 `CellAnalyzer_ROI_Fiji.ijm`。它对应 `cell_analyzer_batch.ipynb` / `cell_analyzer.ipynb` 的流程：**一个用户选择的通道定义细胞边界，所有通道在同一套 ROI 中测量**。旧的 `CellAnalyzer_Native_Fiji.ijm` 为每个通道分别识别对象，适用目的不同。

## 第一次使用

1. 打开 Fiji（ImageJ 1.54f 或更新版本）。
2. 点击 **Plugins > Macros > Install...**，选择 `CellAnalyzer_ROI_Fiji.ijm`。
3. 先运行 **Cell Analyzer ROI - Self-Test**，选择 E/I 盘测试目录。成功会生成 `self_test_passed.txt`。
4. 打开原始图片。CZI/LIF/ND2/OIR 等可通过 Fiji 自带的 Bio-Formats 打开；多个场景先明确选择要分析的场景。通道数由当前图片读取。
5. 运行 **Cell Analyzer ROI - Preview or Batch**，先选 **Preview active image**。
6. 选择分割通道（Fiji 从 **1** 编号；Python 从 **0** 编号），设置面积、圆度、阈值方法等。每个通道分别设置名称、颜色、Gaussian sigma 和信号阈值。
7. 检查六格 Overview：原图、平滑图、初始阈值、形态学清理、分水岭候选、最终边界。对应的逐细胞测量显示在 Results 中。

预览窗口点击 OK 后会出现 **Choose preview view**：可选分割 Overview 或任意测量通道。各通道视图展示 Raw、Smoothed、Positive mask，以及带细胞边界的 ROI 内阳性信号；只有选择该通道时才生成。查看完后再点击 OK，选择 **Finish preview**。预览不需要输出目录。

预览本身不需要输出目录。如需保存此次选用参数，起始窗口勾选 **Save parameters for a preview run**，选择参数 TXT 保存位置。后续运行勾选 **Load saved macro parameters**，打开该文件；所有参数仍可在对话框中调整。

## 核心参数

- **Segmentation channel**：用哪一通道确定细胞 ROI。其它通道不重新定义细胞。
- **Cell threshold**：细胞边界阈值。Manual 输入原始强度值；Otsu/Yen/Triangle 可调自动阈值倍率。
- **Minimum / Maximum cell area**：µm²；最大面积 0 表示不限。宏优先读取每个文件的物理像素大小。无标定图片会要求输入 µm/pixel，计算方法为整幅图实际宽度 µm / 像素宽度。
- **Opening / Closing radius**：去除小噪声 / 连接小间隙；0 关闭。
- **Split touching cells**：启用后才询问分水岭突出度，单位为距离图的像素。调大通常减少切分。Fiji 使用 `Distance Map` + `Find Maxima`。
- **Signal threshold**：各通道的阳性标准；Manual 的值直接作用于平滑后的原始强度。None 让整个 ROI 都纳入测量。
- **Advanced filters**：额外边缘排除宽度和局部对比度过滤。默认局部对比度为 0（关闭）。

所有对话框使用英文。边界颜色支持 `yellow` 等颜色名称或 `#RRGGBB`，线宽可调。

## 批量运行

准备一个 TXT，每行一个完整文件路径，允许来自不同文件夹。例如：

```text
I:\experiment\WT\image01.tif
I:\experiment\KO\image02.tif
E:\other_folder\image03.czi
```

也可以直接使用 Python 分析导出的 `selected_image_files.txt`。

运行宏，选择 **Batch from file-list TXT**，选路径文件。宏会要求勾选 **I verified the channel order for all files**：必须先核对所有文件的通道身份和顺序一致，不能仅凭通道数量相同判断。原生宏不自动匹配厂商通道名称；不勾选就不会开始批量。

用第一张图配置通道和参数，然后选择输出目录；同一套参数用于所有文件。通道数和所选 Z/T 不兼容的文件会在批次汇总中标记并跳过。损坏文件或 Bio-Formats 导入失败可能中断宏；先确认这些文件能单独在 Fiji 中打开。

原始图为 Z-stack 时，可选择 Max Intensity、Average Intensity 或 Single plane，并选择一个时间点。Fiji 的 Z/T 编号也从 1 开始。

每次保存都会新建带时间戳的输出子目录，包含：

- `parameters.txt`、`selected_image_files.txt`：可复用宏参数和文件路径。
- `combined_roi_measurements.csv`、`batch_summary.csv`：合并逐细胞表、批次状态。
- 每图子目录：`roi_measurements.csv`、`thresholds.csv`、`parameters.txt`、`roi_labels.tif`、`imagej_rois.zip`（有细胞时）、`overview_qc.png` 和每通道的彩色边界图。
- 勾选保存阶段图时：每通道原始 TIFF、平滑 TIFF、阳性掩膜 TIFF，以及初始阈值、清理掩膜、分水岭候选 TIFF。

不创建应用缓存。所有文件只保存到用户选定的目录。宏需要空的 ROI Manager，运行前先保存已有 ROI；旧 Results 表会改名保留。

## 表格数值如何理解

每个 ROI 每个通道同时给出：

- `raw_mean_intensity`：整个 ROI 的未阈值过滤原始均值。
- `mean_intensity`：阈值以下像素置零后，仍以整个 ROI 面积为分母的均值；对应 Python 的同名指标。
- `positive_mean_intensity`：仅阳性像素的原始强度平均值；没有阳性像素时为 NaN，不能解释为亮度 0。
- `integrated_intensity`：阳性位置的原始像素强度总和。
- `positive_area_um2`：阳性像素数 × 单像素物理面积。
- `positive_fraction`：阳性像素数 / ROI 像素数，范围 0–1。

阳性标准从平滑图计算，强度从原始图读取。没有阳性信号的 ROI 仍输出一行：阈值过滤均值、积分、阳性面积和比例为 0；阳性区域均值为 NaN；原始均值仍包含背景。`thresholds.csv` 的阈值 `-1` 表示该通道选择 None。

例：100 像素 ROI，一半原始强度 1000，另一半 100，手动阈值 500：raw mean=550、mean=500、positive mean=1000、positive fraction=0.5。

全部参数的单位、公式、Python 包/函数与 ImageJ 命令对照见根目录 [README 的参数参考](../../README.md)。

## 与 Python 的差异

这是原生 Fiji 实现，不能保证边界、ROI 数和自动阈值与 Python 数值完全一致。ImageJ 和 scikit-image 的直方图、圆度、形态学核、连通性、分水岭边界各有差异；Fiji 的核心分水岭以突出度控制，不提供 Python 的最小峰间距、最小峰高或 compactness。宏保留 Manual/Otsu/Yen/Triangle；Python 的 percentile 和 adaptive 阈值不在本宏中。Python YAML 和 Fiji TXT 也不是同一种配置文件，不能直接互相加载。

建议固定一批代表性图片，人工核对 20–30 个细胞的边界和粘连拆分后再批量分析；同一组比较统一使用一个实现和同一套参数。

实现参考：[ImageJ 内置宏函数](https://imagej.net/ij/developer/macro/functions.html)、[MaximumFinder 源码](https://imagej.net/ij/developer/source/ij/plugin/filter/MaximumFinder.java.html)。
