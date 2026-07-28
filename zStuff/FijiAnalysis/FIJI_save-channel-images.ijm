// for each file in folder 'data', save specific channels with certain adjustments


curPath = "C:\\Users\\landgrafn\\Desktop\\PVN_hits\\";
// C1-NET, C2-Iba1, C3-Ab, C4-DAPI
channel = "C1-";
channel_name = "NET"

// create folders and define paths
File.makeDirectory(curPath + "images_" + channel_name);
resultPath = curPath + "images_" + channel_name + "\\";
File.makeDirectory(resultPath + "results_pics");
dataPath = curPath + "data\\";


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
		
		// save Pic
		selectImage(channel + file_list[i]);
		saveAs("Tiff", resultPath + "results_pics\\" + file_list[i] + "_pic.tif");
		
		// close all windows
		close("*");
		close("Original Metadata - " + file_list[i]);
	}
}

