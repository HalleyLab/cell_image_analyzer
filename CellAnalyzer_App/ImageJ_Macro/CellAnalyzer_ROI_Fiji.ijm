// One segmentation channel defines cell ROIs; every channel uses the SAME ROIs.
// Fiji/ImageJ commands only. Bio-Formats is used only to import vendor formats.
// Install via Plugins > Macros > Install. No cache or Python is used.

var loadedSettings="", settings="", channelCount=0, names, colors, sigmas;
var signalMethods, signalValues, signalScales, rawIDs, smoothIDs, positiveIDs;
var referenceChannel=1, cellMethod="Otsu", cellValue=0, cellScale=0.8;
var minArea=5, maxArea=0, minCircularity=0.05, openRadius=1, closeRadius=2;
var fillHoles=true, splitCells=false, prominence=1, excludeEdges=true, borderMargin=0;
var minContrast=0, contrastRadius=4, projection="Max Intensity", zPlane=1, timePoint=1;
var fallbackX=0, fallbackY=0, pixelX=0, pixelY=0, lineWidth=2, lineColor="yellow";
var saveStages=true, cellCSV="", roiCount=0, actualCellThreshold=0;
var prefix="", initialID=0, cleanID=0, candidateID=0, labelsID=0, overviewID=0;
var signalOverviewID=0;

macro "Cell Analyzer ROI - Preview or Batch" {
    argument=getArgument();
    if (startsWith(argument, "self-test="))
        selfTest(substring(argument, 10));
    else
        runWorkflow();
}

macro "Cell Analyzer ROI - Self-Test" {
    selfTest(getDirectory("Choose self-test output folder"));
}

function runWorkflow() {
    requires("1.54f");
    if (roiManager("count")>0)
        exit("Save your existing ROI Manager entries and clear them before running this macro.");
    if (nResults>0) Table.rename("Results", "Results before Cell Analyzer "+getTime());
    modes=newArray("Preview active image", "Analyze active image", "Batch from file-list TXT");
    Dialog.create("Cell Analyzer ROI");
    Dialog.addChoice("Mode", modes, modes[0]);
    Dialog.addCheckbox("Load saved macro parameters", false);
    Dialog.addCheckbox("Save parameters for a preview run", false);
    Dialog.show();
    mode=Dialog.getChoice();
    reload=Dialog.getCheckbox();
    savePreviewParameters=Dialog.getCheckbox();
    if (reload) loadedSettings=File.openAsString(File.openDialog("Open parameters.txt"));
    paths=newArray(0);
    openedFirst=false;
    if (mode==modes[2]) {
        lines=split(replace(File.openAsString(File.openDialog("Open selected_image_files.txt")), "\r", ""), "\n");
        for (i=0; i<lines.length; i++) {
            path=String.trim(lines[i]);
            if (lengthOf(path)>0) {
                if (!File.exists(path)) exit("Image does not exist:\n"+path);
                paths=Array.concat(paths, newArray(path));
            }
        }
        if (paths.length==0) exit("The file list is empty.");
        Dialog.create("Confirm batch channel order");
        Dialog.addMessage("This native macro cannot reliably match vendor channel identities.\nVerify that every file has the same channels in the same order before proceeding.");
        Dialog.addCheckbox("I verified the channel order for all files", false);
        Dialog.show();
        if (!Dialog.getCheckbox()) exit("Inspect the channel order before starting a batch.");
        sourceID=openSource(paths[0]);
        openedFirst=true;
    } else {
        if (nImages==0) exit("Open an image in Fiji first, or select Batch from file-list TXT.");
        sourceID=getImageID();
        sourcePath=getInfo("image.directory")+getInfo("image.filename");
        if (sourcePath=="") sourcePath="Active image: "+getTitle();
        paths=newArray(sourcePath);
    }
    selectImage(sourceID);
    if (bitDepth()==24) exit("Use grayscale channel data or a multichannel hyperstack; RGB screenshots are not supported.");
    getDimensions(w, h, channelCount, nz, nt);
    configure(sourceID, nz, nt);
    preview=mode==modes[0];
    if (preview && savePreviewParameters) {
        parameterDir=getDirectory("Choose folder for preview parameters");
        File.saveString(settings, parameterDir+"CellAnalyzer_ROI_parameters_"+d2s(getTime(),0)+".txt");
    }
    output="";
    if (!preview) {
        root=getDirectory("Choose output folder");
        output=root+"CellAnalyzer_ROI_"+d2s(getTime(), 0)+File.separator;
        File.makeDirectory(output);
        File.saveString(settings, output+"parameters.txt");
        File.saveString(joinLines(paths), output+"selected_image_files.txt");
    }
    combined="";
    summary="source_file,status,roi_count,pixel_size_x_um,pixel_size_y_um,cell_threshold\n";
    setBatchMode(true);
    for (fileIndex=0; fileIndex<paths.length; fileIndex++) {
        if (fileIndex>0) sourceID=openSource(paths[fileIndex]);
        imageOutput="";
        if (!preview) {
            imageOutput=output+d2s(fileIndex+1, 0)+"_"+safeName(File.getName(paths[fileIndex]))+File.separator;
            File.makeDirectory(imageOutput);
        }
        success=analyzeImage(sourceID, paths[fileIndex], imageOutput);
        if (success) {
            if (combined=="") combined=cellCSV;
            else combined=combined+substring(cellCSV, indexOf(cellCSV, "\n")+1);
            summary=summary+csv(paths[fileIndex])+",complete,"+roiCount+","+pixelX+","+pixelY+","+actualCellThreshold+"\n";
            if (preview) {
                showPreview();
            }
            closeWork();
        } else summary=summary+csv(paths[fileIndex])+",incompatible_channels_or_plane,0,,,\n";
        if (mode==modes[2]) { selectImage(sourceID); close(); }
    }
    if (!preview) {
        File.saveString(combined, output+"combined_roi_measurements.csv");
        File.saveString(summary, output+"batch_summary.csv");
    }
    roiManager("reset");
    setBatchMode(false);
    if (!openedFirst) selectImage(sourceID);
    if (!preview) showMessage("Cell Analyzer ROI", "Finished.\n"+output);
}

function configure(sourceID, nz, nt) {
    settings="workflow=CellAnalyzer_ROI_Fiji_v1\n";
    selectImage(sourceID);
    readCalibration();
    fallbackX=numberParameter("fallback_pixel_x_um", pixelX);
    fallbackY=numberParameter("fallback_pixel_y_um", pixelY);
    if (pixelX<=0 || pixelY<=0) {
        Dialog.create("Image calibration");
        Dialog.addMessage("This image has no micrometer calibration.\nEnter micrometers per pixel (field width in um / image width in pixels).");
        Dialog.addNumber("Pixel width (um)", fallbackX, 6);
        Dialog.addNumber("Pixel height (um)", fallbackY, 6);
        Dialog.show();
        fallbackX=Dialog.getNumber(); fallbackY=Dialog.getNumber();
        if (fallbackX<=0 || fallbackY<=0) exit("Valid physical pixel sizes are required.");
    }
    put("fallback_pixel_x_um", fallbackX); put("fallback_pixel_y_um", fallbackY);
    channelChoices=newArray(channelCount);
    for (i=0; i<channelCount; i++) channelChoices[i]="Channel "+(i+1);
    referenceChannel=round(numberParameter("segmentation_channel", 1));
    if (referenceChannel<1 || referenceChannel>channelCount) referenceChannel=1;
    cellMethod=parameter("cell_method", "Otsu");
    Dialog.create("Cell ROIs - essential settings");
    Dialog.addChoice("Segmentation channel", channelChoices, channelChoices[referenceChannel-1]);
    Dialog.addChoice("Cell threshold method", newArray("Manual", "Otsu", "Yen", "Triangle"), cellMethod);
    Dialog.addNumber("Minimum cell area (um^2)", numberParameter("min_area_um2", 5), 2);
    Dialog.addNumber("Maximum cell area (um^2; 0 = no limit)", numberParameter("max_area_um2", 0), 2);
    Dialog.addNumber("Minimum circularity (0-1)", numberParameter("min_circularity", 0.05), 2);
    Dialog.addNumber("Opening radius (px; 0 = off)", numberParameter("open_radius_px", 1), 0);
    Dialog.addNumber("Closing radius (px; 0 = off)", numberParameter("close_radius_px", 2), 0);
    Dialog.addCheckbox("Fill enclosed holes", numberParameter("fill_holes", 1));
    Dialog.addCheckbox("Split touching cells", numberParameter("split_cells", 0));
    Dialog.addCheckbox("Exclude image-edge cells", numberParameter("exclude_edges", 1));
    Dialog.addCheckbox("Adjust advanced filters", false);
    Dialog.show();
    selectedChannel=Dialog.getChoice();
    for (i=0; i<channelCount; i++) if (channelChoices[i]==selectedChannel) referenceChannel=i+1;
    cellMethod=Dialog.getChoice();
    minArea=Dialog.getNumber(); maxArea=Dialog.getNumber(); minCircularity=Dialog.getNumber();
    openRadius=round(Dialog.getNumber()); closeRadius=round(Dialog.getNumber());
    fillHoles=Dialog.getCheckbox(); splitCells=Dialog.getCheckbox(); excludeEdges=Dialog.getCheckbox();
    advanced=Dialog.getCheckbox();
    if (minArea<=0 || maxArea<0 || (maxArea>0 && maxArea<=minArea) || minCircularity<0 || minCircularity>1 || openRadius<0 || closeRadius<0)
        exit("Invalid area, circularity, or cleanup settings.");
    if (cellMethod=="Manual") cellValue=getNumber("Cell intensity threshold", numberParameter("cell_value", 1000));
    else cellScale=getNumber("Cell automatic threshold multiplier", numberParameter("cell_scale", 0.8));
    if (cellValue<0 || cellScale<=0) exit("Threshold must be nonnegative and multiplier must be positive.");
    prominence=numberParameter("watershed_prominence_px", 1);
    if (splitCells) prominence=getNumber("Watershed prominence (px; larger = fewer splits)", prominence);
    if (prominence<0) exit("Watershed prominence cannot be negative.");
    borderMargin=numberParameter("border_margin_px", 0);
    minContrast=numberParameter("min_local_contrast", 0);
    contrastRadius=numberParameter("contrast_ring_px", 4);
    if (advanced) {
        Dialog.create("Optional cell filters");
        if (excludeEdges) Dialog.addNumber("Additional border margin (px)", borderMargin, 0);
        Dialog.addNumber("Minimum local contrast ratio (0 = off)", minContrast, 2);
        Dialog.addNumber("Contrast ring radius (px)", contrastRadius, 0);
        Dialog.show();
        if (excludeEdges) borderMargin=round(Dialog.getNumber());
        minContrast=Dialog.getNumber(); contrastRadius=round(Dialog.getNumber());
    }
    if (borderMargin<0 || minContrast<0 || contrastRadius<1) exit("Invalid optional filter settings.");
    projection=parameter("z_projection", "Max Intensity");
    zPlane=round(numberParameter("z_plane", 1)); timePoint=round(numberParameter("time_point", 1));
    if (nz>1 || nt>1) {
        Dialog.create("Image plane");
        if (nz>1) {
            Dialog.addChoice("Z projection", newArray("Max Intensity", "Average Intensity", "Single plane"), projection);
            Dialog.addNumber("Z plane (used by Single plane)", zPlane, 0);
        }
        if (nt>1) Dialog.addNumber("Time point (1-based)", timePoint, 0);
        Dialog.show();
        if (nz>1) { projection=Dialog.getChoice(); zPlane=round(Dialog.getNumber()); }
        if (nt>1) timePoint=round(Dialog.getNumber());
    }
    if (zPlane<1 || timePoint<1 || timePoint>nt || (projection=="Single plane" && zPlane>nz)) exit("Selected plane is unavailable.");
    names=newArray(channelCount); colors=newArray(channelCount); sigmas=newArray(channelCount);
    signalMethods=newArray(channelCount); signalValues=newArray(channelCount); signalScales=newArray(channelCount);
    defaultColors=newArray("Green", "Magenta", "Red", "Blue", "Cyan", "Yellow", "Grays");
    for (i=0; i<channelCount; i++) {
        key="C"+(i+1)+"_";
        Dialog.create("Channel "+(i+1)+" - signal measurements");
        Dialog.addString("Name", parameter(key+"name", "Channel_"+(i+1)));
        Dialog.addChoice("Display color", defaultColors, parameter(key+"color", defaultColors[i%defaultColors.length]));
        Dialog.addNumber("Gaussian sigma (px; 0 = off)", numberParameter(key+"sigma", 1), 2);
        Dialog.addChoice("Signal threshold", newArray("Manual", "Otsu", "Yen", "Triangle", "None"), parameter(key+"signal_method", "Otsu"));
        Dialog.show();
        names[i]=Dialog.getString(); colors[i]=Dialog.getChoice(); sigmas[i]=Dialog.getNumber(); signalMethods[i]=Dialog.getChoice();
        signalValues[i]=numberParameter(key+"signal_value", 5000); signalScales[i]=numberParameter(key+"signal_scale", 1);
        if (signalMethods[i]=="Manual") signalValues[i]=getNumber(names[i]+": manual signal threshold", signalValues[i]);
        else if (signalMethods[i]!="None") signalScales[i]=getNumber(names[i]+": automatic signal threshold multiplier", signalScales[i]);
        if (names[i]=="" || sigmas[i]<0 || signalValues[i]<0 || signalScales[i]<=0) exit("Invalid channel parameters.");
        put(key+"name", names[i]); put(key+"color", colors[i]); put(key+"sigma", sigmas[i]);
        put(key+"signal_method", signalMethods[i]); put(key+"signal_value", signalValues[i]); put(key+"signal_scale", signalScales[i]);
    }
    Dialog.create("Preview and export");
    Dialog.addMessage("Active optional filters: local contrast = "+minContrast+" (0 = off).\nBorder margin = "+borderMargin+" px; exclude edges = "+excludeEdges+".");
    Dialog.addString("Boundary color (name or #RRGGBB)", parameter("boundary_color", "yellow"));
    Dialog.addNumber("Boundary width (px)", numberParameter("boundary_width_px", 2), 1);
    Dialog.addCheckbox("Save all processing-stage TIFF images", numberParameter("save_stages", 1));
    Dialog.show();
    lineColor=Dialog.getString(); lineWidth=Dialog.getNumber(); saveStages=Dialog.getCheckbox();
    if (lineWidth<=0) exit("Boundary width must be positive.");
    put("channel_count", channelCount); put("segmentation_channel", referenceChannel);
    put("cell_method", cellMethod); put("cell_value", cellValue); put("cell_scale", cellScale);
    put("min_area_um2", minArea); put("max_area_um2", maxArea); put("min_circularity", minCircularity);
    put("open_radius_px", openRadius); put("close_radius_px", closeRadius); put("fill_holes", fillHoles);
    put("split_cells", splitCells); put("watershed_prominence_px", prominence); put("exclude_edges", excludeEdges);
    put("border_margin_px", borderMargin); put("min_local_contrast", minContrast); put("contrast_ring_px", contrastRadius);
    put("z_projection", projection); put("z_plane", zPlane); put("time_point", timePoint);
    put("boundary_color", lineColor); put("boundary_width_px", lineWidth); put("save_stages", saveStages);
}

function analyzeImage(sourceID, sourcePath, output) {
    selectImage(sourceID);
    getDimensions(w, h, nc, nz, nt);
    if (nc!=channelCount || bitDepth()==24 || timePoint>nt || (projection=="Single plane" && zPlane>nz)) return false;
    readCalibration();
    if (pixelX<=0 || pixelY<=0) { pixelX=fallbackX; pixelY=fallbackY; }
    if (pixelX<=0 || pixelY<=0) exit("Missing physical calibration for "+sourcePath);
    prefix="CAR_"+d2s(getTime(), 0)+"_";
    rawIDs=newArray(nc); smoothIDs=newArray(nc); positiveIDs=newArray(nc);
    thresholds="channel_index,channel_name,signal_threshold\n";
    for (c=0; c<nc; c++) {
        selectImage(sourceID); run("Select None");
        range="1-"+nz;
        if (projection=="Single plane") range=""+zPlane;
        run("Duplicate...", "title=["+prefix+"raw_C"+(c+1)+"] duplicate channels="+(c+1)+" slices="+range+" frames="+timePoint);
        rawIDs[c]=getImageID();
        if (nz>1 && projection!="Single plane") {
            run("Z Project...", "projection=["+projection+"]");
            projected=getImageID(); selectImage(rawIDs[c]); close(); selectImage(projected);
            rename(prefix+"raw_C"+(c+1)); rawIDs[c]=projected;
        }
        setVoxelSize(pixelX, pixelY, 1, "um"); resetThreshold();
        run("Duplicate...", "title=["+prefix+"smoothed_C"+(c+1)+"]"); run("32-bit");
        if (sigmas[c]>0) run("Gaussian Blur...", "sigma="+sigmas[c]);
        smoothIDs[c]=getImageID();
        run("Duplicate...", "title=["+prefix+"positive_C"+(c+1)+"]");
        cutoff=makeMask(signalMethods[c], signalValues[c], signalScales[c]);
        positiveIDs[c]=getImageID();
        thresholds=thresholds+(c+1)+","+csv(names[c])+","+cutoff+"\n";
        if (output!="" && saveStages) {
            saveImage(rawIDs[c], output+"C"+(c+1)+"_raw.tif");
            saveImage(smoothIDs[c], output+"C"+(c+1)+"_smoothed.tif");
            saveImage(positiveIDs[c], output+"C"+(c+1)+"_positive_mask.tif");
        }
    }
    selectImage(smoothIDs[referenceChannel-1]);
    run("Duplicate...", "title=["+prefix+"initial_threshold]");
    actualCellThreshold=makeMask(cellMethod, cellValue, cellScale); initialID=getImageID();
    run("Duplicate...", "title=["+prefix+"cleaned]");
    if (openRadius>0) { run("Minimum...", "radius="+openRadius); run("Maximum...", "radius="+openRadius); }
    if (closeRadius>0) { run("Maximum...", "radius="+closeRadius); run("Minimum...", "radius="+closeRadius); }
    if (fillHoles) run("Fill Holes"); cleanID=getImageID();
    candidateID=splitMask(cleanID);
    selectImage(candidateID); roiManager("reset"); run("Clear Results");
    run("Set Measurements...", "area perimeter shape centroid redirect=None decimal=6");
    upperArea="Infinity"; if (maxArea>0) upperArea=""+maxArea;
    options="size="+minArea+"-"+upperArea+" circularity="+minCircularity+"-1.0 show=Nothing add";
    if (excludeEdges) options=options+" exclude";
    run("Analyze Particles...", options);
    filterCandidates(w, h);
    roiCount=roiManager("count");
    newImage(prefix+"accepted_labels", "32-bit black", w, h, 1); labelsID=getImageID();
    setVoxelSize(pixelX, pixelY, 1, "um");
    for (r=0; r<roiCount; r++) { roiManager("select", r); setColor(r+1); run("Fill", "slice"); }
    run("Select None");
    measureCells(sourcePath);
    makeOverview(w, h);
    if (output!="") {
        File.saveString(cellCSV, output+"roi_measurements.csv");
        File.saveString(thresholds+"segmentation,"+csv(cellMethod)+","+actualCellThreshold+"\n", output+"thresholds.csv");
        File.saveString("source_file="+sourcePath+"\npixel_size_x_um="+pixelX+"\npixel_size_y_um="+pixelY+"\n"+settings, output+"parameters.txt");
        if (roiCount>0) { selectImage(rawIDs[referenceChannel-1]); roiManager("Save", output+"imagej_rois.zip"); }
        saveImage(labelsID, output+"roi_labels.tif");
        selectImage(overviewID); saveAs("PNG", output+"overview_qc.png");
        if (saveStages) {
            saveImage(initialID, output+"initial_threshold.tif"); saveImage(cleanID, output+"cleaned_mask.tif"); saveImage(candidateID, output+"watershed_candidates.tif");
        }
        for (c=0; c<channelCount; c++) saveBoundaries(rawIDs[c], colors[c], output+"C"+(c+1)+"_boundaries.png");
    }
    return true;
}

function makeMask(method, manual, scale) {
    run("Select None"); resetThreshold();
    if (method=="None") { run("Multiply...", "value=0"); run("Add...", "value=255"); setMinAndMax(0, 255); run("8-bit"); return -1; }
    cutoff=manual;
    lower=cutoff;
    if (method!="Manual") {
        getRawStatistics(np, mean, minimum, maximum);
        if (minimum==maximum) {
            cutoff=minimum*scale; lower=cutoff;
            if (cutoff<=0) lower=1e-8;
        } else {
            setAutoThreshold(method+" dark"); getThreshold(low, high); cutoff=low*scale;
            // Automatic cutoffs are strict: a cutoff of zero must not include the black background.
            lower=cutoff+maxOf(1e-8, abs(cutoff)*1e-9);
        }
    }
    setThreshold(lower, 1e30); setOption("BlackBackground", true); run("Convert to Mask");
    return cutoff;
}

function splitMask(maskID) {
    selectImage(maskID); run("Duplicate...", "title=["+prefix+"candidates]");
    if (splitCells) {
        // Keep ImageJ's 'EDM of ' title: MaximumFinder uses it to recognize a float distance map.
        previousType=call("ij.plugin.filter.EDM.getOutputType");
        call("ij.plugin.filter.EDM.setOutputType", "3"); run("Distance Map");
        call("ij.plugin.filter.EDM.setOutputType", previousType);
        edmID=getImageID();
        run("Find Maxima...", "prominence="+prominence+" output=[Segmented Particles]");
        resultID=getImageID(); selectImage(edmID); close(); selectImage(resultID);
        rename(prefix+"watershed");
    }
    return getImageID();
}

function filterCandidates(w, h) {
    for (r=roiManager("count")-1; r>=0; r--) {
        selectImage(smoothIDs[referenceChannel-1]); roiManager("select", r);
        reject=false;
        getSelectionBounds(x, y, rw, rh);
        if (excludeEdges && borderMargin>0 && (x<=borderMargin || y<=borderMargin || x+rw>=w-borderMargin || y+rh>=h-borderMargin)) reject=true;
        if (!reject && minContrast>0) {
            getRawStatistics(innerPixels, innerMean);
            run("Enlarge...", "enlarge="+contrastRadius); getRawStatistics(outerPixels, outerMean);
            if (outerPixels>innerPixels) {
                ringMean=(outerPixels*outerMean-innerPixels*innerMean)/(outerPixels-innerPixels);
                if ((innerMean+1e-8)/(ringMean+1e-8)<minContrast) reject=true;
            }
        }
        if (reject) { roiManager("select", r); roiManager("delete"); }
    }
}

function measureCells(sourcePath) {
    run("Clear Results");
    header="source_file,roi_id,roi_area_um2,roi_area_px,perimeter_um,circularity";
    rows=newArray(roiCount);
    for (r=0; r<roiCount; r++) {
        selectImage(rawIDs[referenceChannel-1]); roiManager("select", r);
        getRawStatistics(np, mean); List.setMeasurements;
        area=List.getValue("Area"); perimeter=List.getValue("Perim."); circularity=List.getValue("Circ.");
        rows[r]=""+csv(sourcePath)+","+(r+1)+","+num(area)+","+np+","+num(perimeter)+","+num(circularity);
        setResult("source_file", r, sourcePath); setResult("roi_id", r, r+1);
        setResult("roi_area_um2", r, area); setResult("roi_area_px", r, np);
        setResult("perimeter_um", r, perimeter); setResult("circularity", r, circularity);
    }
    for (c=0; c<channelCount; c++) {
        key="C"+(c+1)+"_"+safeName(names[c]);
        header=header+","+key+"_raw_mean_intensity,"+key+"_mean_intensity,"+key+"_integrated_intensity,"+key+"_positive_area_um2,"+key+"_positive_fraction,"+key+"_positive_mean_intensity";
        selectImage(positiveIDs[c]); run("Select None");
        run("Duplicate...", "title=["+prefix+"weight]"); weightID=getImageID();
        run("32-bit"); run("Divide...", "value=255");
        imageCalculator("Multiply create 32-bit", rawIDs[c], weightID); filteredID=getImageID();
        for (r=0; r<roiCount; r++) {
            selectImage(rawIDs[c]); roiManager("select", r); getRawStatistics(np, rawMean);
            selectImage(filteredID); roiManager("select", r); getRawStatistics(nf, filteredMean);
            selectImage(positiveIDs[c]); roiManager("select", r); getRawStatistics(nm, maskMean);
            positive=np*maskMean/255; fraction=maskMean/255; integral=filteredMean*np;
            positiveMean=parseFloat("NaN"); if (positive>0) positiveMean=integral/positive;
            rows[r]=rows[r]+","+num(rawMean)+","+num(filteredMean)+","+num(integral)+","+num(positive*pixelX*pixelY)+","+num(fraction)+","+num(positiveMean);
            setResult(key+"_raw_mean_intensity", r, rawMean); setResult(key+"_mean_intensity", r, filteredMean);
            setResult(key+"_integrated_intensity", r, integral); setResult(key+"_positive_area_um2", r, positive*pixelX*pixelY); setResult(key+"_positive_fraction", r, fraction);
            setResult(key+"_positive_mean_intensity", r, positiveMean);
        }
        selectImage(filteredID); close(); selectImage(weightID); close();
    }
    cellCSV=header+"\n"+joinLines(rows);
    updateResults();
}

function showPreview() {
    views=newArray("Segmentation overview");
    for (c=0; c<channelCount; c++) views=Array.concat(views, newArray("C"+(c+1)+": "+names[c]));
    views=Array.concat(views, newArray("Finish preview"));
    viewID=overviewID;
    while (true) {
        selectImage(viewID); setBatchMode("show");
        waitForUser("Cell Analyzer ROI preview", roiCount+" cells detected.\nInspect this image and Results. Click OK to switch channels or finish.\nMean intensity = positive raw sum / all ROI pixels; positive mean uses positive pixels only.");
        Dialog.create("Choose preview view");
        Dialog.addChoice("View", views, "Finish preview");
        Dialog.show(); choice=Dialog.getChoice();
        if (choice=="Finish preview") return;
        if (choice==views[0]) viewID=overviewID;
        else for (c=0; c<channelCount; c++) if (choice==views[c+1]) {
            if (signalOverviewID!=0 && isOpen(signalOverviewID)) { selectImage(signalOverviewID); close(); }
            makeSignalOverview(c); viewID=signalOverviewID;
        }
    }
}

function makeSignalOverview(c) {
    selectImage(rawIDs[c]); getDimensions(w, h, nc, nz, nt);
    selectImage(labelsID); run("Duplicate...", "title=["+prefix+"accepted_mask]");
    setThreshold(1, 1e30); setOption("BlackBackground", true); run("Convert to Mask"); acceptedMask=getImageID();
    imageCalculator("AND create", acceptedMask, positiveIDs[c]); weight=getImageID();
    run("32-bit"); run("Divide...", "value=255");
    imageCalculator("Multiply create 32-bit", rawIDs[c], weight); inside=getImageID();
    panels=newArray(rawIDs[c], smoothIDs[c], positiveIDs[c], inside);
    captions=newArray("Raw", "Smoothed", "Positive mask", "Positive signal in cells");
    newImage(prefix+"signal_tiles", "RGB black", w, h, 4); tiles=getImageID();
    for (p=0; p<4; p++) {
        selectImage(panels[p]); run("Select None"); run("Duplicate...", "title=["+prefix+"signal_panel]");
        panel=getImageID(); resetThreshold();
        if (p==2) run("Grays"); else run(colors[c]);
        resetMinAndMax(); run("RGB Color");
        if (p==3) drawBoundaries();
        run("Select All"); run("Copy"); selectImage(tiles); setSlice(p+1); run("Paste"); run("Select None");
        setMetadata("Label", captions[p]); selectImage(panel); close();
    }
    selectImage(tiles); run("Make Montage...", "columns=2 rows=2 scale=0.5 first=1 last=4 increment=1 border=2 font=14 label");
    signalOverviewID=getImageID(); rename(prefix+"Signal_C"+(c+1));
    selectImage(tiles); close(); selectImage(acceptedMask); close(); selectImage(weight); close(); selectImage(inside); close();
    selectImage(signalOverviewID);
}

function makeOverview(w, h) {
    panelIDs=newArray(rawIDs[referenceChannel-1], smoothIDs[referenceChannel-1], initialID, cleanID, candidateID, rawIDs[referenceChannel-1]);
    captions=newArray("Raw", "Smoothed", "Threshold", "Cleanup", "Watershed", "Accepted cells");
    newImage(prefix+"overview_tiles", "RGB black", w, h, 6); tiles=getImageID();
    for (p=0; p<6; p++) {
        selectImage(panelIDs[p]); run("Select None"); run("Duplicate...", "title=["+prefix+"panel]");
        panel=getImageID(); resetThreshold();
        if (p<2 || p==5) { run(colors[referenceChannel-1]); resetMinAndMax(); }
        run("RGB Color");
        if (p==5) drawBoundaries();
        run("Select All"); run("Copy"); selectImage(tiles); setSlice(p+1); run("Paste"); run("Select None");
        setMetadata("Label", captions[p]); selectImage(panel); close();
    }
    selectImage(tiles); run("Make Montage...", "columns=3 rows=2 scale=0.5 first=1 last=6 increment=1 border=2 font=14 label");
    overviewID=getImageID(); rename(prefix+"Overview_QC"); selectImage(tiles); close(); selectImage(overviewID);
}

function drawBoundaries() {
    setColor(lineColor); setLineWidth(lineWidth);
    for (r=0; r<roiCount; r++) { roiManager("select", r); run("Draw"); }
    run("Select None");
}

function saveBoundaries(rawID, lut, path) {
    selectImage(rawID); run("Select None"); run("Duplicate...", "title=["+prefix+"boundaries]");
    run(lut); resetMinAndMax(); run("RGB Color"); drawBoundaries(); saveAs("PNG", path); close();
}

function readCalibration() {
    getPixelSize(unit, pixelX, pixelY);
    unit=toLowerCase(unit);
    if (unit=="nm" || unit=="nanometer") { pixelX/=1000; pixelY/=1000; }
    else if (unit=="mm" || unit=="millimeter") { pixelX*=1000; pixelY*=1000; }
    else if (unit=="m" || unit=="meter") { pixelX*=1e6; pixelY*=1e6; }
    else if (unit!="um" && unit!="µm" && unit!="micron" && unit!="micrometer" && unit!="microns") { pixelX=0; pixelY=0; }
}

function openSource(path) {
    lower=toLowerCase(path);
    if (endsWith(lower, ".tif") || endsWith(lower, ".tiff") || endsWith(lower, ".png")) open(path);
    else run("Bio-Formats Importer", "open=["+path+"] color_mode=Default view=Hyperstack stack_order=XYCZT");
    return getImageID();
}

function closeWork() {
    titles=getList("image.titles");
    for (i=0; i<titles.length; i++) if (startsWith(titles[i], prefix)) { selectWindow(titles[i]); close(); }
}
function saveImage(id, path) { selectImage(id); run("Select None"); saveAs("Tiff", path); }
function parameter(key, fallback) {
    lines=split(replace(loadedSettings, "\r", ""), "\n");
    for (i=0; i<lines.length; i++) if (startsWith(lines[i], key+"=")) return substring(lines[i], lengthOf(key)+1);
    return ""+fallback;
}
function numberParameter(key, fallback) {
    value=parameter(key, ""+fallback);
    if (value=="true") return 1; if (value=="false") return 0;
    n=parseFloat(value); if (isNaN(n)) exit("Invalid saved number for "+key+": "+value); return n;
}
function put(key, value) { settings=settings+key+"="+value+"\n"; }
function num(n) { return d2s(n, 9); }
function csv(s) { return "\""+replace(replace(replace(s, "\"", "\"\""), "\r", " "), "\n", " ")+"\""; }
function safeName(s) { return replace(s, "[^A-Za-z0-9_.-]", "_"); }
function joinLines(lines) {
    text=""; for (i=0; i<lines.length; i++) text=text+lines[i]+"\n"; return text;
}

function selfTest(output) {
    requires("1.54f");
    if (roiManager("count")>0) exit("Save and clear existing ROI Manager entries before the self-test.");
    if (nResults>0) Table.rename("Results", "Results before Cell Analyzer self-test "+getTime());
    if (output=="") exit("A self-test output folder is required.");
    if (!endsWith(output, File.separator)) output=output+File.separator;
    File.makeDirectory(output);
    setBatchMode(true);
    channelCount=5; names=newArray("Cells", "Signal", "Zero", "Fourth", "Fifth");
    colors=newArray("Green", "Magenta", "Red", "Blue", "Cyan"); sigmas=newArray(0,0,0,0,0);
    signalMethods=newArray("Manual", "Manual", "Manual", "None", "None");
    signalValues=newArray(1000,500,500,0,0); signalScales=newArray(1,1,1,1,1);
    referenceChannel=1; cellMethod="Manual"; cellValue=1000; minArea=5; maxArea=0;
    minCircularity=0; openRadius=0; closeRadius=0; fillHoles=true; splitCells=false; excludeEdges=false; minContrast=0;
    projection="Max Intensity"; timePoint=1; saveStages=true; settings="workflow=self-test\n";
    newImage("ROI_Fiji_Synthetic", "16-bit black", 64, 64, 5);
    Stack.setDimensions(5,1,1); run("Make Composite", "display=Color"); source=getImageID(); setVoxelSize(0.5,0.75,1,"um");
    Stack.setChannel(1); setColor(2000); makeRectangle(8,8,10,10); run("Fill", "slice"); makeRectangle(40,40,10,10); run("Fill", "slice");
    Stack.setChannel(2); setColor(1000); makeRectangle(8,8,10,10); run("Fill", "slice");
    Stack.setChannel(4); run("Select All"); setColor(200); run("Fill", "slice");
    Stack.setChannel(5); setColor(300); run("Fill", "slice"); run("Select None");
    saveAs("Tiff", output+"synthetic_input.tif");
    if (!analyzeImage(source,"synthetic",output)) exit("Self-test failed: image was rejected.");
    if (roiCount!=2) exit("Self-test failed: expected 2 ROIs, got "+roiCount);
    testRows=split(String.trim(cellCSV), "\n");
    if (testRows.length!=3) exit("Self-test failed: CSV must contain a header and 2 cell rows.");
    if (!startsWith(testRows[1], "\"synthetic\",1,37.500000000,100,")) exit("Self-test failed: exported CSV values.");
    if (abs(getResult("roi_area_um2",0)-37.5)>1e-6) exit("Self-test failed: physical area.");
    if (getResult("C2_Signal_mean_intensity",0)!=1000 || getResult("C2_Signal_integrated_intensity",0)!=100000) exit("Self-test failed: raw thresholded intensities.");
    if (getResult("C2_Signal_positive_fraction",0)!=1 || getResult("C2_Signal_mean_intensity",1)!=0) exit("Self-test failed: positive / zero ROI.");
    if (getResult("C2_Signal_positive_mean_intensity",0)!=1000 || !isNaN(getResult("C2_Signal_positive_mean_intensity",1))) exit("Self-test failed: positive-only mean / undefined mean.");
    if (getResult("C3_Zero_positive_fraction",0)!=0 || getResult("C3_Zero_mean_intensity",1)!=0) exit("Self-test failed: empty channel.");
    if (getResult("C5_Fifth_mean_intensity",1)!=300) exit("Self-test failed: dynamic channels or None threshold.");
    makeSignalOverview(1);
    if (!isOpen(signalOverviewID) || indexOf(getTitle(), "Signal_C2")<0) exit("Self-test failed: per-channel signal preview.");
    selectImage(signalOverviewID); saveAs("PNG", output+"signal_preview_C2.png");
    closeWork();
    selectImage(source); Stack.setChannel(2); makeRectangle(13,8,5,10); setColor(100); run("Fill", "slice"); run("Select None");
    if (!analyzeImage(source,"synthetic_partial", "")) exit("Self-test failed: partial signal image.");
    if (getResult("C2_Signal_raw_mean_intensity",0)!=550 || getResult("C2_Signal_mean_intensity",0)!=500 || getResult("C2_Signal_positive_mean_intensity",0)!=1000 || getResult("C2_Signal_positive_fraction",0)!=0.5) exit("Self-test failed: three distinct intensity means.");
    closeWork(); selectImage(source); close(); roiManager("reset"); run("Clear Results");
    reopened=openSource(output+"synthetic_input.tif"); getDimensions(rw,rh,rc,rz,rt);
    readCalibration();
    if (rc!=5 || abs(pixelX-0.5)>1e-6 || abs(pixelY-0.75)>1e-6) {
        message="Self-test failed: TIFF channels="+rc+", pixel size="+pixelX+","+pixelY;
        print(message); exit(message);
    }
    selectImage(reopened); close();
    loadedSettings="C1_name=Reloaded\nfill_holes=true\ncell_scale=0.7\n";
    testName=parameter("C1_name", "default");
    if (testName!="Reloaded" || numberParameter("fill_holes", 0)!=1 || numberParameter("cell_scale",1)!=0.7) exit("Self-test failed: saved parameters.");
    loadedSettings="";
    channelCount=2; names=newArray("Cells", "Signal"); colors=newArray("Green", "Magenta");
    sigmas=newArray(0,0); signalMethods=newArray("Manual", "Manual"); signalValues=newArray(1000,500); signalScales=newArray(1,1);
    newImage("ROI_Fiji_ZT", "16-bit black",64,64,8);
    Stack.setDimensions(2,2,2); run("Make Composite", "display=Color"); source=getImageID(); setVoxelSize(0.5,0.75,1,"um");
    for (t=1; t<=2; t++) for (z=1; z<=2; z++) for (c=1; c<=2; c++) {
        Stack.setPosition(c,z,t); makeRectangle(8,8,10,10);
        v=4500;
        if (t==2 && c==1) { v=1200; if(z==2) v=2000; }
        if (t==2 && c==2) { v=100; if(z==2) v=900; }
        setColor(v); run("Fill", "slice");
    }
    run("Select None"); timePoint=2;
    if (!analyzeImage(source,"synthetic_zt","") || roiCount!=1) exit("Self-test failed: Z/T projection ROIs.");
    if (getResult("C1_Cells_mean_intensity",0)!=2000 || getResult("C2_Signal_mean_intensity",0)!=900) exit("Self-test failed: selected time / maximum projection.");
    closeWork(); projection="Single plane"; zPlane=1;
    if (!analyzeImage(source,"synthetic_plane","") || getResult("C1_Cells_mean_intensity",0)!=1200 || getResult("C2_Signal_mean_intensity",0)!=0) exit("Self-test failed: selected Z plane.");
    closeWork(); projection="Average Intensity";
    if (!analyzeImage(source,"synthetic_mean","") || getResult("C1_Cells_mean_intensity",0)!=1600 || getResult("C2_Signal_mean_intensity",0)!=500) exit("Self-test failed: mean projection.");
    closeWork(); selectImage(source); close(); roiManager("reset"); run("Clear Results");
    channelCount=1; names=newArray("Cells"); colors=newArray("Green"); sigmas=newArray(1);
    signalMethods=newArray("Manual"); signalValues=newArray(1); signalValues[0]=1000;
    signalScales=newArray(1); signalScales[0]=1;
    timePoint=1; projection="Max Intensity";
    newImage("ROI_Fiji_Empty", "16-bit black",64,64,1); source=getImageID(); setVoxelSize(0.5,0.75,1,"um");
    if (!analyzeImage(source,"synthetic_empty","") || roiCount!=0) exit("Self-test failed: empty image.");
    testRows=split(String.trim(cellCSV), "\n");
    if (testRows.length!=1) exit("Self-test failed: empty table header.");
    closeWork(); cellMethod="Otsu"; signalMethods[0]="Otsu";
    if (!analyzeImage(source,"synthetic_empty_otsu","") || roiCount!=0) exit("Self-test failed: automatic threshold on an empty image.");
    closeWork(); selectImage(source); close(); roiManager("reset"); run("Clear Results");
    newImage("ROI_Fiji_Touching", "8-bit black", 80,64,1); mask=getImageID();
    setColor(255); makeOval(12,18,24,24); run("Fill"); makeOval(30,18,24,24); run("Fill"); run("Select None");
    prefix="CAR_test_"; splitCells=true; prominence=1;
    splitID=splitMask(mask); selectImage(splitID); run("Set Measurements...", "area redirect=None decimal=6");
    run("Analyze Particles...", "size=10-Infinity show=Nothing display clear");
    if (nResults!=2) exit("Self-test failed: touching-cell watershed.");
    closeWork(); selectImage(mask); close(); run("Clear Results");
    File.saveString("PASS: calibrated ROIs; 5 channels; raw sums; positive-only means; zero rows; None threshold; watershed; per-channel preview; saved parameters; max/mean/single Z; selected T; empty image; stage files.\n",output+"self_test_passed.txt");
    setBatchMode(false);
    print("Cell Analyzer ROI self-test passed: "+output);
}
