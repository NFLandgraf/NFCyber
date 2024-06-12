stack = getImageID();
meanBrightness = newArray(nSlices);

// Loop through each slice
for (i = 1; i <= nSlices; i++) {
    setSlice(i);
    getStatistics(area, mean, min, max, stdDev);
    meanBrightness[i-1] = mean;
}


path = "C:\\Users\\landgrafn\\Desktop\\brightness_means.csv";

file = File.open(path);

for (i = 0; i < nSlices; i++) 
	print(file, meanBrightness[i] + "\n");


File.close(file);
