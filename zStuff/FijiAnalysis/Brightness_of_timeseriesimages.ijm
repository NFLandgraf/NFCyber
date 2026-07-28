stack = getImageID();
meanBrightness = newArray(nSlices);

// Loop through each slice
for (i = 1; i <= nSlices; i++) {
    setSlice(i);
    getStatistics(area, mean, min, max, stdDev);
    meanBrightness[i-1] = mean;
}

path = "D:\\2pTube_rGRABDA_Airpuff\\Results_FullFrame\\brightness_means.csv";
file = File.open(path);

for (i = 0; i < nSlices; i++) 
	print(file, meanBrightness[i] + "\n");

File.close(file);
