
inputDir = "E:\\hTauxAPP2(6m)_PFC\\apotome\\";
outputDir = "E:\\hTauxAPP2(6m)_PFC\\orth\\";

setBatchMode(true);

list = getFileList(inputDir);
for (i=0; i<list.length; i++) {
    name = list[i];
    path = inputDir + name;
    if (File.isDirectory(path)) continue;

    lower = toLowerCase(name);
    if (!(endsWith(lower, ".tif") || endsWith(lower, ".tiff") || endsWith(lower, ".lsm") || endsWith(lower, ".czi") || endsWith(lower, ".nd2"))) continue;

    open(path);

    // Z Projection: choose method: "Max Intensity", "Average Intensity", "Sum Slices", etc.
    run("Z Project...", "projection=[Max Intensity]");

    // The projection becomes the active image
    outName = outputDir + name + "_Zproj.tif";
    saveAs("Tiff", outName);

    // Close projection + original
    close(); // closes projection
    selectWindow(name);
    close();
}

setBatchMode(false);
print("Done.");
