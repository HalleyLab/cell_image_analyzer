// Cell Analyzer - native Fiji/ImageJ macro set.
// Uses only commands included with Fiji/ImageJ. No Python process is started.
// Install with Plugins > Macros > Install, then use the two commands below.

macro "Cell Analyzer Native" {
    runAnalyzer();
}

macro "Cell Analyzer Native Self-Test" {
    runSelfTest();
}

function runAnalyzer() {
    requires("1.53t");
    if (nImages == 0) exit("Open a multichannel image first.");

    sourceTitle = getTitle();
    getDimensions(imageWidth, imageHeight, channelCount, sliceCount, frameCount);
    if (channelCount < 1)
        exit("The active image has no readable channels.");
    if (sliceCount != 1 || frameCount != 1)
        exit("This native macro analyzes one 2D field at a time. Select one time point and run Image > Stacks > Z Project first.");

    getPixelSize(unit, pixelWidth, pixelHeight, voxelDepth);
    if (unit == "pixel" || unit == "pixels") {
        pixelWidth = 1;
        pixelHeight = 1;
    }

    outputDir = getDirectory("Choose output folder");
    if (outputDir == "") exit("No output folder selected.");

    channelIndex = newArray(1, 2, 3, 4);
    channelName = newArray("Channel 1", "Channel 2", "Channel 3", "Channel 4");
    wavelength = newArray(488, 647, 555, 405);
    lut = newArray("Green", "Magenta", "Red", "Blue");
    enabled = newArray(1, 1, 1, channelCount >= 4);
    lutChoices = newArray("Green", "Magenta", "Red", "Blue", "Cyan", "Yellow", "Grays");

    Dialog.create("Cell Analyzer Native - Channels");
    Dialog.addMessage("Map the image channels. Disable channels that are not needed.");
    for (i = 0; i < 4; i++) {
        Dialog.addMessage("Channel " + (i + 1));
        Dialog.addCheckbox("Enable channel " + (i + 1), enabled[i]);
        Dialog.addNumber("Image channel index " + (i + 1), channelIndex[i], 0);
        Dialog.addString("Channel name " + (i + 1), channelName[i], 20);
        Dialog.addNumber("Wavelength " + (i + 1) + " (nm; 0 = unset)", wavelength[i], 0);
        Dialog.addChoice("Display color " + (i + 1), lutChoices, lut[i]);
    }
    Dialog.show();
    for (i = 0; i < 4; i++) {
        enabled[i] = Dialog.getCheckbox();
        channelIndex[i] = round(Dialog.getNumber());
        channelName[i] = Dialog.getString();
        wavelength[i] = Dialog.getNumber();
        lut[i] = Dialog.getChoice();
    }

    validateChannels(enabled, channelIndex, channelName, channelCount);

    methods = newArray("Manual", "Default", "Otsu", "Yen", "Triangle");
    thresholdMethod = newArray("Manual", "Manual", "Manual", "Manual");
    manualThreshold = newArray(10000, 5000, 5000, 5000);
    gaussianSigma = newArray(1, 1, 1, 1);

    Dialog.create("Cell Analyzer Native - Segmentation");
    Dialog.addMessage("Bright objects on a dark background are expected.");
    for (i = 0; i < 4; i++) {
        Dialog.addChoice("Threshold method C" + (i + 1), methods, thresholdMethod[i]);
        Dialog.addNumber("Manual threshold C" + (i + 1), manualThreshold[i], 0);
        Dialog.addNumber("Gaussian sigma C" + (i + 1) + " (px)", gaussianSigma[i], 2);
    }
    Dialog.addCheckbox("Save grayscale channel TIFF files", true);
    Dialog.addNumber("Boundary line width (px)", 2, 1);
    Dialog.addCheckbox("Keep work images open after analysis", true);
    Dialog.show();
    for (i = 0; i < 4; i++) {
        thresholdMethod[i] = Dialog.getChoice();
        manualThreshold[i] = Dialog.getNumber();
        gaussianSigma[i] = Dialog.getNumber();
    }
    saveRaw = Dialog.getCheckbox();
    lineWidth = Dialog.getNumber();
    keepOpen = Dialog.getCheckbox();

    minArea = newArray(20, 5, 5, 15);
    maxArea = newArray(100000, 100000, 100000, 1000);
    minCircularity = newArray(0.05, 0.00, 0.00, 0.10);
    maxCircularity = newArray(1.00, 1.00, 1.00, 1.00);

    Dialog.create("Cell Analyzer Native - Object Filters");
    Dialog.addMessage("Area values use calibrated square micrometers.");
    for (i = 0; i < 4; i++) {
        Dialog.addMessage("Channel " + (i + 1) + " objects");
        Dialog.addNumber("Minimum area C" + (i + 1) + " (um^2)", minArea[i], 2);
        Dialog.addNumber("Maximum area C" + (i + 1) + " (um^2)", maxArea[i], 2);
        Dialog.addNumber("Minimum circularity C" + (i + 1), minCircularity[i], 2);
        Dialog.addNumber("Maximum circularity C" + (i + 1), maxCircularity[i], 2);
    }
    Dialog.addMessage("Binary-mask cleanup (applied to every enabled channel)");
    Dialog.addNumber("Open iterations (px)", 0, 0);
    Dialog.addNumber("Close iterations (px)", 0, 0);
    Dialog.addCheckbox("Fill holes", true);
    Dialog.addCheckbox("Watershed touching objects", false);
    Dialog.addCheckbox("Exclude objects touching image edges", true);
    Dialog.show();
    for (i = 0; i < 4; i++) {
        minArea[i] = Dialog.getNumber();
        maxArea[i] = Dialog.getNumber();
        minCircularity[i] = Dialog.getNumber();
        maxCircularity[i] = Dialog.getNumber();
    }
    openIterations = round(Dialog.getNumber());
    closeIterations = round(Dialog.getNumber());
    fillHoles = Dialog.getCheckbox();
    watershed = Dialog.getCheckbox();
    excludeEdges = Dialog.getCheckbox();

    validateSettings(enabled, manualThreshold, gaussianSigma, minArea, maxArea,
        minCircularity, maxCircularity, openIterations, closeIterations, lineWidth,
        pixelWidth, pixelHeight);

    base = fileStem(sourceTitle);
    rawTitles = newArray(4);
    maskTitles = newArray(4);
    outputStems = newArray(4);
    counts = newArray(4);

    setBatchMode(true);
    for (i = 0; i < 4; i++) {
        if (enabled[i]) {
            outputStems[i] = base + "_C" + (i + 1) + "_" + safeName(channelName[i]);
            rawTitles[i] = "CA_RAW_C" + (i + 1) + "_" + base;
            maskTitles[i] = "CA_MASK_C" + (i + 1) + "_" + base;
            makeChannelImages(sourceTitle, channelIndex[i], rawTitles[i], maskTitles[i],
                outputStems[i], outputDir, pixelWidth, pixelHeight, lut[i],
                thresholdMethod[i], manualThreshold[i], gaussianSigma[i], saveRaw,
                openIterations, closeIterations, fillHoles, watershed);
        }
    }

    for (i = 0; i < 4; i++) {
        if (enabled[i]) {
            counts[i] = analyzeObjects(maskTitles[i], rawTitles[i], outputStems[i],
                outputDir, lut[i], minArea[i], maxArea[i], minCircularity[i],
                maxCircularity[i], excludeEdges, lineWidth);
        }
    }

    saveRunFiles(outputDir, base, sourceTitle, enabled, channelIndex, channelName,
        wavelength, lut, thresholdMethod, manualThreshold, gaussianSigma, minArea,
        maxArea, minCircularity, maxCircularity, counts, pixelWidth, pixelHeight,
        openIterations, closeIterations, fillHoles, watershed, excludeEdges, lineWidth);

    setBatchMode(false);
    if (!keepOpen) {
        for (i = 0; i < 4; i++) {
            if (enabled[i]) {
                closeIfOpen(rawTitles[i]);
                closeIfOpen(maskTitles[i]);
            }
        }
    }
    selectWindow(sourceTitle);
    showMessage("Cell Analyzer Native", "Analysis finished.\n\nOutput folder:\n" + outputDir);
}

function makeChannelImages(sourceTitle, channelIndex, rawTitle, maskTitle,
        outputStem, outputDir, pixelWidth, pixelHeight, lutName, thresholdMethod,
        manualThreshold, sigma, saveRaw, openIterations, closeIterations,
        fillHoles, watershed) {
    selectWindow(sourceTitle);
    Stack.setChannel(channelIndex);
    run("Duplicate...", "title=[" + rawTitle + "]");
    setVoxelSize(pixelWidth, pixelHeight, 1, "um");

    if (saveRaw)
        saveAs("Tiff", outputDir + outputStem + "_raw.tif");

    run("Duplicate...", "title=[" + outputStem + "_color]");
    run(lutName);
    run("RGB Color");
    saveAs("PNG", outputDir + outputStem + "_color.png");
    close();

    selectWindow(rawTitle);
    run("Duplicate...", "title=[" + maskTitle + "]");
    if (sigma > 0)
        run("Gaussian Blur...", "sigma=" + sigma);
    if (thresholdMethod == "Manual") {
        getStatistics(area, mean, minimum, maximum);
        if (manualThreshold > maximum) maximum = manualThreshold;
        setThreshold(manualThreshold, maximum);
    } else {
        setAutoThreshold(thresholdMethod + " dark");
    }
    setOption("BlackBackground", true);
    run("Convert to Mask");
    for (j = 0; j < openIterations; j++) run("Open");
    for (j = 0; j < closeIterations; j++) run("Close");
    if (fillHoles) run("Fill Holes");
    if (watershed) run("Watershed");
    saveAs("Tiff", outputDir + outputStem + "_mask.tif");
}

function analyzeObjects(maskTitle, rawTitle, outputStem, outputDir, lutName,
        minimumArea, maximumArea, minimumCircularity, maximumCircularity,
        excludeEdges, lineWidth) {
    selectWindow(maskTitle);
    roiManager("reset");
    run("Clear Results");
    run("Set Measurements...", "area mean min max centroid perimeter fit shape feret's area_fraction redirect=[" + rawTitle + "] decimal=3");

    options = "size=" + minimumArea + "-" + maximumArea +
        " circularity=" + minimumCircularity + "-" + maximumCircularity +
        " show=Nothing display clear add";
    if (excludeEdges) options = options + " exclude";
    run("Analyze Particles...", options);
    objectCount = nResults;

    if (objectCount > 0) {
        saveAs("Results", outputDir + outputStem + "_objects.csv");
        roiManager("Save", outputDir + outputStem + "_rois.zip");
        for (j = 0; j < roiManager("count"); j++) {
            roiManager("select", j);
            Roi.setStrokeColor("yellow");
            Roi.setStrokeWidth(lineWidth);
            roiManager("update");
        }
    } else {
        File.saveString("No objects detected\n", outputDir + outputStem + "_objects.csv");
    }

    selectWindow(rawTitle);
    run("Duplicate...", "title=[" + outputStem + "_boundaries]");
    run(lutName);
    run("RGB Color");
    if (objectCount > 0) {
        roiManager("Show All without labels");
        run("Flatten");
        closePreviousDuplicate(outputStem + "_boundaries");
    }
    rename(outputStem + "_boundaries_final");
    saveAs("PNG", outputDir + outputStem + "_boundaries.png");
    close();
    roiManager("reset");
    return objectCount;
}

function closePreviousDuplicate(title) {
    if (isOpen(title)) {
        current = getTitle();
        selectWindow(title);
        close();
        selectWindow(current);
    }
}

function saveRunFiles(outputDir, base, sourceTitle, enabled, channelIndex,
        channelName, wavelength, lut, thresholdMethod, manualThreshold,
        gaussianSigma, minArea, maxArea, minCircularity, maxCircularity,
        counts, pixelWidth, pixelHeight, openIterations, closeIterations,
        fillHoles, watershed, excludeEdges, lineWidth) {
    summary = "source_image,channel_slot,image_channel_index,channel_name,wavelength_nm,object_count\n";
    parameters = "Cell Analyzer Native parameters\n";
    parameters = parameters + "Source image: " + sourceTitle + "\n";
    parameters = parameters + "Pixel width (um): " + pixelWidth + "\n";
    parameters = parameters + "Pixel height (um): " + pixelHeight + "\n";
    parameters = parameters + "Open iterations: " + openIterations + "\n";
    parameters = parameters + "Close iterations: " + closeIterations + "\n";
    parameters = parameters + "Fill holes: " + fillHoles + "\n";
    parameters = parameters + "Watershed: " + watershed + "\n";
    parameters = parameters + "Exclude edges: " + excludeEdges + "\n";
    parameters = parameters + "Boundary line width (px): " + lineWidth + "\n";

    for (i = 0; i < 4; i++) {
        if (enabled[i]) {
            cleanName = csvSafe(channelName[i]);
            summary = summary + csvSafe(sourceTitle) + "," + (i + 1) + "," +
                channelIndex[i] + "," + cleanName + "," + wavelength[i] + "," + counts[i] + "\n";
            parameters = parameters + "\nChannel " + (i + 1) + "\n";
            parameters = parameters + "Enabled: true\n";
            parameters = parameters + "Image channel index: " + channelIndex[i] + "\n";
            parameters = parameters + "Name: " + channelName[i] + "\n";
            parameters = parameters + "Wavelength (nm): " + wavelength[i] + "\n";
            parameters = parameters + "Display color: " + lut[i] + "\n";
            parameters = parameters + "Threshold method: " + thresholdMethod[i] + "\n";
            parameters = parameters + "Manual threshold: " + manualThreshold[i] + "\n";
            parameters = parameters + "Gaussian sigma (px): " + gaussianSigma[i] + "\n";
            parameters = parameters + "Minimum area (um^2): " + minArea[i] + "\n";
            parameters = parameters + "Maximum area (um^2): " + maxArea[i] + "\n";
            parameters = parameters + "Minimum circularity: " + minCircularity[i] + "\n";
            parameters = parameters + "Maximum circularity: " + maxCircularity[i] + "\n";
        }
    }
    File.saveString(summary, outputDir + base + "_native_summary.csv");
    File.saveString(parameters, outputDir + base + "_native_parameters.txt");
}

function validateChannels(enabled, channelIndex, channelName, channelCount) {
    enabledCount = 0;
    for (i = 0; i < 4; i++) {
        if (enabled[i]) {
            enabledCount++;
            if (channelIndex[i] < 1 || channelIndex[i] > channelCount)
                exit("Channel " + (i + 1) + " index must be between 1 and " + channelCount + ".");
            if (lengthOf(channelName[i]) == 0)
                exit("Channel " + (i + 1) + " needs a name.");
            for (j = i + 1; j < 4; j++) {
                if (enabled[j] && channelIndex[i] == channelIndex[j])
                    exit("Two enabled channel slots use image channel " + channelIndex[i] + ".");
            }
        }
    }
    if (enabledCount == 0) exit("Enable at least one channel.");
}

function validateSettings(enabled, manualThreshold, gaussianSigma, minArea,
        maxArea, minCircularity, maxCircularity, openIterations,
        closeIterations, lineWidth, pixelWidth, pixelHeight) {
    if (pixelWidth <= 0 || pixelHeight <= 0)
        exit("Pixel size must be greater than zero.");
    if (openIterations < 0 || closeIterations < 0)
        exit("Open and close iterations cannot be negative.");
    if (lineWidth <= 0)
        exit("Boundary line width must be greater than zero.");
    for (i = 0; i < 4; i++) {
        if (enabled[i]) {
            if (manualThreshold[i] < 0 || gaussianSigma[i] < 0)
                exit("Threshold and Gaussian sigma cannot be negative.");
            if (minArea[i] < 0 || maxArea[i] <= minArea[i])
                exit("Channel " + (i + 1) + " area limits are invalid.");
            if (minCircularity[i] < 0 || maxCircularity[i] > 1 ||
                    maxCircularity[i] < minCircularity[i])
                exit("Channel " + (i + 1) + " circularity limits must be between 0 and 1.");
        }
    }
}

function runSelfTest() {
    requires("1.53t");
    newImage("CellAnalyzer_Native_SelfTest", "8-bit black", 64, 64, 1);
    setForegroundColor(255, 255, 255);
    makeOval(8, 8, 12, 12);
    run("Fill");
    makeOval(40, 40, 10, 10);
    run("Fill");
    run("Select None");
    setThreshold(1, 255);
    run("Clear Results");
    run("Analyze Particles...", "size=20-Infinity circularity=0.00-1.00 show=Nothing display clear");
    detected = nResults;
    close();
    run("Clear Results");
    if (detected != 2)
        exit("Self-test failed. Expected 2 objects but found " + detected + ".");
    showMessage("Cell Analyzer Native", "Self-test passed: 2 objects detected.");
}

function fileStem(name) {
    dot = lastIndexOf(name, ".");
    if (dot > 0) name = substring(name, 0, dot);
    return safeName(name);
}

function safeName(text) {
    text = replace(text, " ", "_");
    text = replace(text, "/", "_");
    text = replace(text, "\\", "_");
    text = replace(text, ":", "_");
    text = replace(text, "*", "_");
    text = replace(text, "?", "_");
    text = replace(text, "\"", "_");
    text = replace(text, "<", "_");
    text = replace(text, ">", "_");
    text = replace(text, "|", "_");
    return text;
}

function csvSafe(text) {
    text = replace(text, ",", "_");
    text = replace(text, "\n", " ");
    text = replace(text, "\r", " ");
    return text;
}

function closeIfOpen(title) {
    if (isOpen(title)) {
        selectWindow(title);
        close();
    }
}
