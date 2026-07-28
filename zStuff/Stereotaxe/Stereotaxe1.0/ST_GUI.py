from tkinter import *
import pickle, os
dir_path = os.getcwd()
from internal_info import *

win = Tk()

win.geometry("400x230")
win.title("Stereotaxe tilt compensation")
win.columnconfigure(0, weight=1)
win.columnconfigure(1, weight=8)
win.columnconfigure(2, weight=8)
win.columnconfigure(3, weight=8)

Label(win,text='coordinate format: [AP, ML, DV]').grid(row=0,column=0,sticky=W)
Label(win,text='regular bregma: [0, 0, 0]').grid(row=1,column=0,sticky=W)
Label(win,text='regular lambda: [-3.8, 0, 0]').grid(row=2,column=0, sticky=W)
Label(win,text='').grid(row=3,column=0)

check_size_compVar = BooleanVar()#value=check_size_comp)
size_comp = Checkbutton(win,text='Size compensation',variable=check_size_compVar,onvalue=True,offvalue=False,height=0,width=0)
size_comp.grid(row=0,column=2,columnspan=2,sticky=W)
check_AP_tiltVar = BooleanVar()#value=check_AP_tilt)
AP_tilt = Checkbutton(win,text='AP tilt compensation',variable=check_AP_tiltVar,onvalue=True,offvalue=False,height=0,width=0)
AP_tilt.grid(row=1,column=2,columnspan=2,sticky=W)
check_ML_tiltVar = BooleanVar()#value=check_ML_tilt)
ML_tilt = Checkbutton(win,text='ML tilt compensation',variable=check_ML_tiltVar,onvalue=True,offvalue=False,height=0,width=0)
ML_tilt.grid(row=2,column=2,columnspan=2,sticky=W)

Label(win,text='AP').grid(row=4,column=1)
Label(win,text='ML').grid(row=4,column=2)
Label(win,text='DV').grid(row=4,column=3)

u = 5
Label(win,text='aim coordinates').grid(row=u,column=0, sticky=W)
aim_AP = StringVar()
aim_AP_entry = Entry(win, textvariable=aim_AP)
aim_AP_entry.grid(row=u, column=1)
aim_ML = StringVar()
aim_ML_entry = Entry(win, textvariable=aim_ML)
aim_ML_entry.grid(row=u, column=2)
aim_DV = StringVar()
aim_DV_entry = Entry(win, textvariable=aim_DV)
aim_DV_entry.grid(row=u, column=3)

a = 6
Label(win,text='measured lambda: ').grid(row=6,column=0, sticky=W)
lmd_AP = StringVar()
lmd_AP_entry = Entry(win, textvariable=lmd_AP)
lmd_AP_entry.grid(row=a, column=1)
lmd_ML = StringVar()
lmd_ML_entry = Entry(win, textvariable=lmd_ML)
lmd_ML_entry.grid(row=a, column=2)
lmd_DV = StringVar()
lmd_DV_entry = Entry(win, textvariable=lmd_DV)
lmd_DV_entry.grid(row=a, column=3)

#MLR
b = 7
Label(win,text='measured ML-R: ').grid(row=b,column=0, sticky=W)
lb_ML_R = Label(win,text='0')
lb_ML_R.grid(row=b,column=1)

def value_changed():
    MLL_ML.config(text=-float(MLR_ML.get()))

MLR_ML = StringVar()
MLR_ML_sb = Spinbox(win,textvariable=MLR_ML,from_=0.5,to=3,values=(0.5,1.0,1.5,2.0,2.5,3.0),command=value_changed)
MLR_ML_sb.grid(row=b,column=2)

MLR_DV = StringVar()
MLR_DV_entry = Entry(win, textvariable=MLR_DV)
MLR_DV_entry.grid(row=b, column=3)

#MLL
c = 8
Label(win,text='measured ML-L: ').grid(row=c,column=0, sticky=W)
MLL_AP = Label(win,text='0')
MLL_AP.grid(row=c,column=1)
MLL_ML = Label(win,text=-float(MLR_ML.get()))
MLL_ML.grid(row=c,column=2)
MLL_DV = StringVar()
MLL_DV_entry = Entry(win, textvariable=MLL_DV)
MLL_DV_entry.grid(row=c, column=3)


def print_internal_info():
    with open("internal_info.py", "w") as f:
        f.write("# user input:\n"
                f"ocheck_size_comp = {check_size_compVar.get()}\n"
                f"ocheck_AP_tilt = {check_AP_tiltVar.get()}\n"
                f"ocheck_ML_tilt = {check_ML_tiltVar.get()}\n"
                f"onormal_lmd = [{float(-3.8)}, {int(0)}, {int(0)}]\n"
                f"oraw_aim = [{aim_AP.get()}, {aim_ML.get()}, {aim_DV.get()}]\n"
                f"oaim = [{aim_AP.get()}, {aim_ML.get()}, {aim_DV.get()}]\n"
                f"omeasured_lmd = [{lmd_AP.get()}, {lmd_ML.get()}, {lmd_DV.get()}]\n"
                f"omeasured_ML_R = [{int(0)}, {MLR_ML.get()}, {MLR_DV.get()}]\n"
                f"omeasured_ML_L = [{int(0)}, -{MLR_ML.get()}, {MLL_DV.get()}]\n\n")

def func():
    check_size_comp = check_size_compVar.get()
    check_AP_tilt = check_AP_tiltVar.get()
    check_ML_tilt = check_ML_tiltVar.get()
    normal_lmd = [float(-3.8), float(0), float(0)]
    raw_aim = [float(aim_AP.get()), float(aim_ML.get()), float(aim_DV.get())]
    aim = [float(aim_AP.get()), float(aim_ML.get()), float(aim_DV.get())]
    measured_lmd = [float(lmd_AP.get()), float(lmd_ML.get()), float(lmd_DV.get())]
    measured_ML_R = [int(0), float(MLR_ML.get()), float(MLR_DV.get())]
    measured_ML_L = [int(0), -float(MLR_ML.get()), float(MLL_DV.get())]
    input_done = True

    data = [check_size_comp, check_AP_tilt, check_ML_tilt, normal_lmd, raw_aim, aim, measured_lmd, measured_ML_R, measured_ML_L, input_done]
    with open("internal_pickle.py", "wb") as f:
        pickle.dump(data, f)
    print_internal_info()
    print("GUI_done")
    win.destroy()


Button(win, text="GO", width=0,height=0, command=func).grid(row=10, column=3, sticky=E)


aim_AP_entry.insert(0, oraw_aim[0])
aim_ML_entry.insert(0, oraw_aim[1])
aim_DV_entry.insert(0, oraw_aim[2])

lmd_AP_entry.insert(0, omeasured_lmd[0])
lmd_ML_entry.insert(0, omeasured_lmd[1])
lmd_DV_entry.insert(0, omeasured_lmd[2])

MLR_DV_entry.insert(0, omeasured_ML_R[2])
MLL_DV_entry.insert(0, omeasured_ML_L[2])

win.mainloop()
