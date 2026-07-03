// ===== Macro2: Apply ROI from Max Projection to the time series, measure mean over time,
//                and save outputs with filenames based on the original TIFF =====
// What it does:
// - Prompts you to select the ORIGINAL time series window.
// - Uses the FIRST ROI in ROI Manager (draw it on the MaxProjection and press 't' before running).
// - Measures mean intensity in that ROI across all frames and saves CSV.
// - Saves Max Projection as <basename>_MAX.tif
// - Saves Max Projection with ROI painted in YELLOW as <basename>_MAX_withROI_yellow.png

macro "Makro2 - ROI trace + save max + save max+ROI (yellow)" {
    
    // --- Build base name and directory from the original stack ---
    dir = getDirectory("image");           // directory of the active image
    title = getTitle();                    // window title (usually the filename)
    lname = toLowerCase(title);
    extLen = 0;
    if (endsWith(lname, ".tif")) extLen = 4;
    else if (endsWith(lname, ".tiff")) extLen = 5;
    if (extLen > 0)
        base = substring(title, 0, lengthOf(title)-extLen);
    else
        base = title;

    // --- Compute and save Max Projection from the ORIGINAL stack ---
    run("Z Project...", "projection=[Max Intensity]");
    saveAs("Tiff", dir + base + "_MaxProj.tif");

    // --- Create a copy to paint the ROI (yellow) and save ---
    roiManager("Select", 0);
    setForegroundColor(255, 255, 0);
    roiManager("Set Line Width", 1);
    roiManager("Draw");
    saveAs("Tiff", dir + base + "_Roi.tif");
    
    // --- Measure mean intensity over all frames in original stack ---
    selectWindow(title);
    run("Set Measurements...", "mean median decimal=6");
    roiManager("Select", 0);
    roiManager("Multi Measure");  // does the loop for you!

    // --- Save Results as CSV ---
    selectWindow("Results");
    saveAs("Results", dir + base + "_Roi_Trace.csv");
    
    // Optional: tidy up the extra max window we created for saving
    if (isOpen("MaxProjection_forSave")) close("MaxProjection_forSave");