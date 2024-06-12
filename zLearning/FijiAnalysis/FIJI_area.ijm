// data must be in folder called 'data'
curPath = "C:\\Users\\landgrafn\\Desktop\\PVN\\";
// C1-NET, C2-Iba1, C3-Ab, C4-DAPI
channel = "C1-";
channel_name = "NET"

// create folders and define paths
File.makeDirectory(curPath + "results_" + channel_name);
resultPath = curPath + "results_" + channel_name + "\\";
File.makeDirectory(resultPath + "results_area");
File.makeDirectory(resultPath + "results_pics");
dataPath = curPath + "data\\";


// create csv file to store results in
csvFilePath = resultPath + "area_percent.csv";
f = File.open(csvFilePath);
print(f, "file, %area");


// let's go
file_list = getFileList(dataPath);
for(i = 0;i<file_list.length;i++){
	// only choose files with "orth" in the name
	if(file_list[i].indexOf("orth") != -1) {
		// Split and choose correct image
		open(dataPath + file_list[i]);
		selectImage(file_list[i]);
		run("Split Channels");
		selectImage(channel + file_list[i]);
		
		// Brightness/Contrast
		setMinAndMax(6, 600);
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
		saveAs("Results", resultPath + "results_area\\" + file_list[i] + "_area.csv");
		
		selectImage(channel + file_list[i]);
		saveAs("Tiff", resultPath + "results_pics\\" + file_list[i] + "_pic.tif");
		
		// save file in csv to know the order
		result_area = getResult("%Area");
		print(f, file_list[i] + "," + result_area);
		
		
		// close all windows
		close("*");
		close("Results");
		close("Original Metadata - " + file_list[i]);
	}
}

File.close(f);
