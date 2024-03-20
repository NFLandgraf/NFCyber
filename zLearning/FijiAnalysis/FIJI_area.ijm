curPath = "C:\\Users\\landgrafn\\Desktop\\VMHvl_4peranimal\\";
file_list = getFileList(curPath);

for(i = 0;i<file_list.length;i++){
	// only choose files with "orth" in the name
	if(file_list[i].indexOf("orth") != -1) {
		// Split and choose correct image
		open(curPath + file_list[i]);
		selectImage(file_list[i]);
		run("Split Channels");
		selectImage("C3-" + file_list[i]);
		
		// Brightness/Contrast
		setMinAndMax(6, 300);
		run("Apply LUT");
		
		// Threshold
		setAutoThreshold("Default dark no-reset");
		setThreshold(24000, 65535, "raw");
		setOption("BlackBackground", true);
		run("Convert to Mask");
		
		// Measure
		run("Set Measurements...", "area mean min area_fraction redirect=None decimal=3");
		run("Measure");
		
		// save Results and Pic
		selectWindow("Results");
		saveAs("Results", curPath + "Results\\results_area\\" + file_list[i] + "_area.csv");
		
		selectImage("C3-" + file_list[i]);
		saveAs("Tiff", curPath + "Results\\results_pics\\" + file_list[i] + "_pic.tif");
		
		// close all windows
		close("*");
		close("Results");
		close("Original Metadata - " + file_list[i]);
	}
}
