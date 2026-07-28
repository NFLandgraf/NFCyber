curPath = "C:\\Users\\neuwardti\\Desktop/2025-12-10_hTauxAPP1(3m)_Stainings\\LC\\"; 
channel = "C2-"; 
channel_name = "cfos"; 


// create folders and define paths 
File.makeDirectory(curPath + "images_" + channel_name); 
resultPath = curPath + "images_" + channel_name + "\\"; 
File.makeDirectory(resultPath + "results_pics"); 
dataPath = curPath; 


// let's go 
file_list = getFileList(dataPath); 
for(i = 0;i<file_list.length;i++){ 
	// only choose files with "orth" in the name 
	if(file_list[i].indexOf("orth") != -1) { 
		
		// Split and choose correct image 
		open(dataPath + file_list[i]); 
		selectImage(file_list[i]); 
		
		// Split channels 
		
		run("Split Channels"); 
		selectImage(channel + file_list[i]); 
		
		// Brightness/Contrast 
		setMinAndMax(80, 125); 
		//run("Apply LUT"); 
		
		//run threshold 
		setThreshold(360, 65535, "raw"); 
		setOption("BlackBackground", true); 
		run("Convert to Mask"); 
		
		// save Pic 
		
		selectImage(channel + file_list[i]); 
		saveAs("Jpeg", resultPath+ "results_pics\\" + file_list[i] + "_highthreshold.jpg"); 

		// close all 
		close("*"); 
		close("Original Metadata - " + file_list[i]); } } 
		
		// for each file in folder 'data', save specific channels with certain adjustments
	}
}
File.close(f);
