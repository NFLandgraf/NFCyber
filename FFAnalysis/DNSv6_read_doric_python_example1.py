"""
Example on how to read a .doric file using Python

To use this scrypt you will need to have Python3 and 3 Python library
matplotlib, h5py and numpy

Example file generated with an FPC
There is two way to do it Automatically or Manually
"""

if __name__ == "__main__":


	# import doric tools to read doric file
	import doric as dr #doric tools use numpy and h5py library 
	
	import matplotlib.pyplot as plt
	
	#Set the name of the file that you want to extract the data from in
	#this example the file is in the current folder where we have your python script
	
	filename = 'Console_Acq_0000.doric'; 
	
	
	#------------ Automatically------------------------------
	#It's possible to use a function that we codded to extract all the data
	#automatically. The function is named ExtractDataAcquisition .
	
	Data_Acquired = dr.ExtractDataAcquisition(filename); #Data_Acquired is a list of dictionary 
	for data in Data_Acquired:
		plt.figure()
		plt.title(data["Name"])
		
		Signal = data["Data"][0]["Data"]
		Time = data["Data"][1]["Data"]
		plt.plot(Time,Signal)
		plt.xlabel("Time")
		plt.ylabel("signal")
	
	
	
	
	#--------------------Manually-----------------------------
	#It's also possible to do every thing manually 
	#To help you it's possible to display what is inside a .doric file:
	dr.h5print(filename)
	
	#It will give that:
	# Console_Acq_0000.doric
	#   Configurations
	#     FPConsole
	#       AIN01
	#         GraphSettings
	#         Settings
	#       AOUT01
	#         GraphSettings
	#         Modulations
	#           Modulation1
	#         Settings
	#       GlobalSettings
	#       SavingSettings
	#   DataAcquisition
	#     FPConsole
	#       Signals
	#         Series0001
	#           AnalogIn
	#             AIN01: (60244,)
	#             Time: (60244,)
	#           AnalogOut
	#             AOUT01: (60244,)
	#             Time: (60244,)
	#         Series0002
	#           AnalogIn
	#             AIN01: (60244,)
	#             Time: (60244,)
	#           AnalogOut
	#             AOUT01: (60244,)
	#             Time: (60244,)
	#         Series0003
	#           AnalogIn
	#             AIN01: (60244,)
	#             Time: (60244,)
	#           AnalogOut
	#             AOUT01: (60244,)
	#             Time: (60244,)
	#         Series0004
	#           AnalogIn
	#             AIN01: (60244,)
	#             Time: (60244,)
	#           AnalogOut
	#             AOUT01: (60244,)
	#             Time: (60244,)
	
	#If for example I want to load data and time from the AnalogIn channel I use those commands:
	SignalIn, SignalInInfo = dr.h5read(filename,['DataAcquisition','FPConsole','Signals','Series0001','AnalogIn','AIN01']);
	TimeIn, TimeInInfo = dr.h5read(filename,['DataAcquisition','FPConsole','Signals','Series0001','AnalogIn','Time']);
	
	plt.figure()
	plt.plot(TimeIn,SignalIn)
	