// Makro1 — Create Max Projection from a Frames×Y×X TIFF
// Use: open your TIFF (time series) so it’s the active window, then run.

macro "Makro1 - Create Max Projection (Frames x Y x X)" {
    
    // get base name + directory of the active image
    dir  = getDirectory("image");
    name = getTitle();
    
    // strip extension (handles .tif and .tiff)
    lname = toLowerCase(name);
    extLen = 0;
    if (endsWith(lname, ".tif"))  extLen = 4;
    if (endsWith(lname, ".tiff")) extLen = 5;
    if (extLen>0)
        base = substring(name, 0, lengthOf(name)-extLen);
    else
        base = name;

    // create max-intensity projection
    run("Z Project...", "projection=[Max Intensity]");
    rename("MaxProjection");

    // save next to the original with an addendum
    saveAs("Tiff", dir + base + "_MaxProj.tif");
}
