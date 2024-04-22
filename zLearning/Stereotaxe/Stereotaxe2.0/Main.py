#%% IMPORT
# IMPORT

import os
from tkinter import *
from tkinter import ttk
import math
from PIL import ImageTk,Image
import numpy as np
from user_input import *


dir_path = os.getcwd()
regular_lambda = [-4.3, 0, 0]
decnumb = 2


#%% SET UP GUI
# SET UP GUI

win = Tk()

# window configurations
win.geometry("400x230")
win.title("Stereotaxe tilt compensation")
win.columnconfigure(0, weight=1)
win.columnconfigure(1, weight=8)
win.columnconfigure(2, weight=8)
win.columnconfigure(3, weight=8)

# normal anatomy info
Label(win,text='coordinate format: [AP, ML, DV]').grid(row=0,column=0,sticky=W)
Label(win,text='regular bregma: [0, 0, 0]').grid(row=1,column=0,sticky=W)
Label(win,text=f'regular lambda: {regular_lambda}').grid(row=2,column=0, sticky=W)
Label(win,text='').grid(row=3,column=0)

# checkboxes
check_size_compVar = BooleanVar()
size_comp = Checkbutton(win,text='skull size compensation',variable=check_size_compVar,onvalue=True,offvalue=False,height=0,width=0).grid(row=0,column=2,columnspan=2,sticky=W)
check_pitchVar = BooleanVar()
pitch = Checkbutton(win,text='pitch compensation',variable=check_pitchVar,onvalue=True,offvalue=False,height=0,width=0).grid(row=1,column=2,columnspan=2,sticky=W)
check_rollVar = BooleanVar()
roll = Checkbutton(win,text='roll compensation',variable=check_rollVar,onvalue=True,offvalue=False,height=0,width=0).grid(row=2,column=2,columnspan=2,sticky=W)

# bottom grid
Label(win,text='AP').grid(row=4,column=1)
Label(win,text='ML').grid(row=4,column=2)
Label(win,text='DV').grid(row=4,column=3)

row = 5
Label(win,text='aim coordinates').grid(row=row,column=0, sticky=W)
aim_AP = StringVar()
aim_AP_entry = Entry(win, textvariable=aim_AP)
aim_AP_entry.grid(row=row, column=1)
aim_ML = StringVar()
aim_ML_entry = Entry(win, textvariable=aim_ML)
aim_ML_entry.grid(row=row, column=2)
aim_DV = StringVar()
aim_DV_entry = Entry(win, textvariable=aim_DV)
aim_DV_entry.grid(row=row, column=3)

row = 6
Label(win,text='measured lambda: ').grid(row=6,column=0, sticky=W)
lmd_AP = StringVar()
lmd_AP_entry = Entry(win, textvariable=lmd_AP)
lmd_AP_entry.grid(row=row, column=1)
lmd_ML = StringVar()
lmd_ML_entry = Entry(win, textvariable=lmd_ML)
lmd_ML_entry.grid(row=row, column=2)
lmd_DV = StringVar()
lmd_DV_entry = Entry(win, textvariable=lmd_DV)
lmd_DV_entry.grid(row=row, column=3)

#MLR
row = 7
Label(win,text='measured ML-R: ').grid(row=row,column=0, sticky=W)
lb_ML_R = Label(win,text='0')
lb_ML_R.grid(row=row,column=1)
MLR_ML = StringVar()

def value_changed():
    MLL_ML.config(text=-float(MLR_ML.get()))

MLR_ML_sb = Spinbox(win,textvariable=MLR_ML,from_=0.5,to=3,values=(0.5,1.0,1.5,2.0,2.5,3.0),command=value_changed).grid(row=row,column=2)
MLR_DV = StringVar()
MLR_DV_entry = Entry(win, textvariable=MLR_DV)
MLR_DV_entry.grid(row=row, column=3)

#MLL
row = 8
Label(win,text='measured ML-L: ').grid(row=row,column=0, sticky=W)
MLL_AP = Label(win,text='0').grid(row=row,column=1)
MLL_ML = Label(win,text=-float(MLR_ML.get()))
MLL_ML.grid(row=row,column=2)
MLL_DV = StringVar()
MLL_DV_entry = Entry(win, textvariable=MLL_DV)
MLL_DV_entry.grid(row=row, column=3)


# insert last values
def insert_old_values():
    aim_AP_entry.insert(0, puser_aim[0])
    aim_ML_entry.insert(0, puser_aim[1])
    aim_DV_entry.insert(0, puser_aim[2])

    lmd_AP_entry.insert(0, puser_lambda[0])
    lmd_ML_entry.insert(0, puser_lambda[1])
    lmd_DV_entry.insert(0, puser_lambda[2])

    MLR_DV_entry.insert(0, puser_ML_R[2])
    MLL_DV_entry.insert(0, puser_ML_L[2])


# print user_input to check and for inserting last values
def print_internal_info(user_check_size_comp, user_check_pitch, user_check_roll, user_aim, user_lambda, user_ML_R, user_ML_L):
    with open("user_input.py", "w") as f:
        f.write("# user input:\n"
                f"puser_check_size_comp = {user_check_size_comp}\n"
                f"puser_check_pitch = {user_check_pitch}\n"
                f"puser_check_roll = {user_check_roll}\n"
                f"pregular_lambda = {regular_lambda}\n"
                f"puser_aim = {user_aim}\n"
                f"puser_lambda = {user_lambda}\n"
                f"puser_ML_R = {user_ML_R}\n"
                f"puser_ML_L = {user_ML_L}\n\n")
    
# get input values and make them global
def get_user_input():
    global user_check_size_comp, user_check_pitch, user_check_roll, user_aim, user_lambda, user_ML_R, user_ML_L

    user_check_size_comp = check_size_compVar.get()
    user_check_pitch = check_pitchVar.get()
    user_check_roll = check_rollVar.get()
    user_aim = [float(aim_AP.get()), float(aim_ML.get()), float(aim_DV.get())]
    user_lambda = [float(lmd_AP.get()), float(lmd_ML.get()), float(lmd_DV.get())]
    user_ML_R = [int(0), float(MLR_ML.get()), float(MLR_DV.get())]
    user_ML_L = [int(0), -float(MLR_ML.get()), float(MLL_DV.get())]
    
    print_internal_info(user_check_size_comp, user_check_pitch, user_check_roll, user_aim, user_lambda, user_ML_R, user_ML_L)
    win.destroy()
    print('input done')
    

Button(win, text="GO", width=0,height=0, command=get_user_input).grid(row=10, column=3, sticky=E)
insert_old_values()
win.mainloop()


#%% CALCULATIONS
# CALCULATIONS


def raw_calc():
    # I think this is for drawing the injection points without correction
    global raw_sc, raw_ap, alpha_AP, raw_ml, delta_ML
    
    # Size compensation
    size_comp_ratio = user_lambda[0] / regular_lambda[0]
    raw_sc = [round(user_aim[0] * size_comp_ratio, decnumb),
              round(user_aim[1] * size_comp_ratio, decnumb),
              round(user_aim[2] * size_comp_ratio, decnumb)]

    # Pitch
    alpha_AP = np.degrees(np.arcsin(user_lambda[2] / -user_lambda[0]))  # Winkel bei bregma
    raw_beta = np.degrees(np.arctan(user_aim[2] / user_aim[0])) # calculates normal angle (beta) at bregma for aim
    raw_beta_hypot = np.sqrt(user_aim[0] ** 2 + user_aim[2] ** 2)
    raw_corrected_beta = -alpha_AP + raw_beta # corrects beta angle for tilt

    raw_ap = [round(-raw_beta_hypot * np.cos(np.radians(raw_corrected_beta)), decnumb), 
              round(user_aim[1], decnumb), 
              round(-raw_beta_hypot * np.sin(np.radians(raw_corrected_beta)), decnumb)]


    # Roll
    delta_gegenkathete = user_ML_L[2] - user_ML_R[2]
    delta_ankathete = -user_ML_L[1] + user_ML_R[1]
    delta_ML = np.degrees(np.arctan(delta_gegenkathete / delta_ankathete))

    raw_gamma_ankathete = -user_aim[2] # calculating ML angle of aim
    raw_gamma_gegenkathete = user_aim[1]
    raw_gamma_hyp = np.sqrt(raw_gamma_ankathete ** 2 + raw_gamma_gegenkathete ** 2)
    raw_gamma = np.degrees(np.arctan(raw_gamma_gegenkathete / raw_gamma_ankathete))

    raw_corrected_gamma = raw_gamma - delta_ML # calculate correct aim

    raw_ml = [round(user_aim[0], decnumb),
              round(np.sin(np.radians(raw_corrected_gamma)) * raw_gamma_hyp, decnumb),
              round(-np.cos(np.radians(raw_corrected_gamma)) * raw_gamma_hyp, decnumb)]

def size_compensation(aim): 
    global aim_sc, size_comp_ratio
    size_comp_ratio = round(user_lambda[0] / regular_lambda[0], decnumb)
    aim_sc = [round(aim[0] * size_comp_ratio, decnumb),
              round(aim[1] * size_comp_ratio, decnumb),
              round(aim[2] * size_comp_ratio, decnumb)]
    aim = aim_sc
    return aim

def AP_tilt(aim):
    global aim_ap, alpha_AP
    aim[0] = 0.0001 if aim[0] == 0 else aim[0]
    alpha_AP = round(np.degrees(np.arcsin(user_lambda[2] / -user_lambda[0])), decnumb) # Winkel bei bregma

    # calculates normal angle (beta) at bregma for aim
    beta = np.degrees(np.arctan(aim[2] / aim[0]))
    raw_beta = np.degrees(np.arctan(user_aim[2] / user_aim[0]))
    beta_hypot = math.sqrt(aim[0] ** 2 + aim[2] ** 2)
    raw_beta_hypot = math.sqrt(user_aim[0] ** 2 + user_aim[2] ** 2)

    # corrects beta angle for tilt
    corrected_beta = -alpha_AP + beta
    aim_ap =   [round(-beta_hypot * np.cos(np.radians(corrected_beta)), decnumb), 
                round(aim[1], decnumb), 
                round(-beta_hypot * np.sin(np.radians(corrected_beta)), decnumb)]
    aim = aim_ap

    return aim

def ML_tilt(aim):
    global aim_ml, delta_ML
    #skull ML tilt
    delta_gegenkathete = user_ML_L[2] - user_ML_R[2]
    delta_ankathete = -user_ML_L[1] + user_ML_R[1]
    delta_ML = round(np.degrees(np.arctan(delta_gegenkathete / delta_ankathete)), decnumb)

    # calculating ML angle of aim
    gamma_ankathete = -aim[2]
    gamma_gegenkathete = aim[1]
    gamma_hyp = np.sqrt(gamma_ankathete**2 + gamma_gegenkathete**2)
    gamma = np.degrees(np.arctan(gamma_gegenkathete / gamma_ankathete))

    #calculate correct aim
    corrected_gamma = gamma - delta_ML
    aim_ml =  [round(aim[0], decnumb), 
               round(np.sin(np.radians(corrected_gamma)) * gamma_hyp, decnumb), 
               round(-np.cos(np.radians(corrected_gamma)) * gamma_hyp, decnumb)]
    aim = aim_ml

    return aim

def print_out_detailed(aim):

    output = "Stereotaxe_Feedback.txt"

    with open (output, "w") as f:
        f.write(f"Raw aim coordinates: {user_aim}\n"
    		f"Measured lambda: {user_lambda}\n"
    		f"Normal lambda: {regular_lambda}\n"
    		f"Measured ML_L: {user_ML_L}\n"
    		f"Measured ML_R: {user_ML_R}\n")

    # size compensation
    if user_check_size_comp:
        if size_comp_ratio != 1:
            size = "(smaller than usual)" if size_comp_ratio < 1 else "(bigger than usual)"
            with open(output, "a") as f:
                f.write(f"\nSKULL SIZE COMPENSATION:\nFactor x = {round(size_comp_ratio, decnumb)} {size}\n"
                        f"Corrected aim coordinates: {aim_sc}\n")
        else:
            with open(output, "a") as f:
                f.write("\nSKULL SIZE COMPENSATION:\n"
                        "No compensation necessary bc AP(measured lambda) = AP(normal lambda).\n")
    
    # pitch
    if user_check_pitch:
        if alpha_AP != 0:
            which_AP_tilt = "(Bregma < Lambda)" if alpha_AP > 0 else "(Bregma > Lambda)"
            with open(output, "a") as f:
                f.write(f"\nAP TILT\n"
                        f"Skull AP tilt alpha: {round(alpha_AP, 2)} degree {which_AP_tilt}\n"
                        f"Corrected aim: {aim_ap}\n")
        else:
            with open(output, "a") as f:
                f.write("\nAP TILT:\n"
                        "No compensation necessary bc DV(bregma) = DV(lambda).\n")

    # roll
    if user_check_roll:
        if delta_ML != 0:
            which_ML_tilt = "(left > right)" if delta_ML > 0 else "(right > left)"
            with open (output, "a") as f:
                f.write(f"\nML TILT\n"
                        f"Skull ML tilt delta: {round(delta_ML, decnumb)} degrees {which_ML_tilt}\n"
                        f"Corrected aim: {aim_ml}\n")
        else:
            with open(output, "a") as f:
                f.write("\nML TILT:\n"
                        "No compensation necessary bc DV(ML_L) = DV (ML_R).\n")

def calculations():

    raw_calc()
    aim = user_aim
    if user_check_size_comp:
        aim = size_compensation(aim)
    if user_check_pitch:
        aim = AP_tilt(aim)
    if user_check_roll:
        aim = ML_tilt(aim)

    print_out_detailed(aim)
    print('calculations done')
    return aim


aim = calculations()

#%% OUTPUT
# OUTPUT

ap_slices_list = [0.04, 0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84, 0.96, 1.08, 1.2, 1.32, 1.44, 1.56, 1.68, 1.8, 1.92,
                  2.04, 2.16, 2.28, 2.4, 2.52, 2.64, 2.76, 2.88, 3, 3.12, 3.25, 3.36, 3.44, 3.6, 3.72, 999999]

ml_slices_list = [4.28, 3.92, 3.56, 3.08, 2.96, 2.8, 2.68, 2.58, 2.46, 2.34, 2.22, 2.1, 1.98, 1.94, 1.78, 1.7, 1.54,
                  1.42, 1.34, 1.18, 1.1, 0.98, 0.86, 0.74, 0.62, 0.5, 0.38, 0.26, 0.14, 0.02, -0.1, -0.22, -0.34, -0.46,
                  -0.58, -0.7, -0.82, -0.94, -1.06, -1.22, -1.34, -1.46, -1.58, -1.7, -1.82, -1.94, -2.06, -2.18, -2.3,
                  -2.46, -2.54, -2.7, -2.8, -2.92, -3.08, -3.16, -3.28, -3.4, -3.52, -3.64, -3.8, -3.88, -4.04, -4.16,
                  -4.24, -4.36, -4.48, -4.6, -4.72, -4.84, -4.96, -5.02, -5.2, -5.34, -5.4, -5.52, -5.68, -5.8, -5.88,
                  -6, -6.12, -6.24, -6.36, -6.48, -6.64, -6.72, -6.84, -6.96, -7.08, -7.2, -7.32, -7.48, -7.56, -7.64,
                  -7.76, -7.92, -8, -8.12, -8.24, 999]


root = Tk()
root.geometry("1400x650")
root.title("Output")

# Tabs
tabControl = ttk.Notebook(root)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)
tab4 = ttk.Frame(tabControl)
tabControl.pack(expand=1, fill="both")

def summary(tab):
    Label(tab, text="SUMMARY").place(x=990, y=360)
    Label(tab, text=f"Orig. aim:                                          {user_aim}").place(x=990, y=380)
    if user_check_size_comp:
        Label(tab, text=f"Size comp.:   {size_comp_ratio}x -> adapt. aim: {aim_sc}").place(x=990, y=400)
    if user_check_pitch:
        Label(tab, text=f"AP tilt:          {alpha_AP}° -> adapt. aim: {aim_ap}").place(x=990, y=420)
    if user_check_roll:
        Label(tab, text=f"ML tilt:          {delta_ML}°  -> adapt. aim: {aim_ml}").place(x=990, y=440)
    Label(tab, text=f"Aim for coordinates {aim}").place(x=990, y=480)





# ----------size comparison
def show_sc_tab(tab):
    global canvas1
    tabControl.add(tab, text=" size compensation")
    sc_size = [600, 550]
    sc_bregma = [sc_size[0] / 2, 50, 30]
    sc_lambda = [sc_size[0] / 2, 300, 320]
    sc_measured_lmd = [sc_size[0] / 2, sc_lambda[1] * size_comp_ratio, sc_lambda[1] * size_comp_ratio + 20]

    canvas1 = Canvas(tab, width=sc_size[0], height=sc_size[1])
    canvas1.pack(anchor=W)
    sc_draw_normal(sc_size, sc_bregma, sc_lambda)
    sc_draw_measured(sc_measured_lmd, sc_lambda)
    sc_text(tab)
    summary(tab)

def sc_draw_normal(sc_size, sc_bregma, sc_lambda):
    canvas1.create_rectangle(2,2,sc_size[0],sc_size[1])
    canvas1.create_line(sc_bregma[0],sc_bregma[1],sc_lambda[0],sc_lambda[1])       # main line
    canvas1.create_line(sc_bregma[0],sc_bregma[1],sc_bregma[0]-100,sc_bregma[2])  # links oben
    canvas1.create_line(sc_bregma[0],sc_bregma[1],sc_bregma[0]+100,sc_bregma[2])  # rechts oben

    canvas1.create_line(sc_lambda[0],sc_lambda[1],sc_lambda[0]-100,sc_lambda[2])  # links unten
    canvas1.create_line(sc_lambda[0],sc_lambda[1],sc_lambda[0]+100,sc_lambda[2])  # rechts unten

    canvas1.create_line(sc_bregma[0],sc_bregma[1],sc_bregma[0]+200,sc_bregma[1], dash=(2, 4))
    canvas1.create_line(sc_lambda[0],sc_lambda[1],sc_lambda[0]+200,sc_lambda[1], dash=(2, 4))

def sc_draw_measured(sc_measured_lmd, sc_lambda):
    canvas1.create_line(sc_measured_lmd[0],sc_lambda[1],sc_measured_lmd[0],sc_measured_lmd[1], fill="red")       # main line
    canvas1.create_line(sc_measured_lmd[0],sc_measured_lmd[1],sc_measured_lmd[0]-100,sc_measured_lmd[2], fill="red")    # links unten
    canvas1.create_line(sc_measured_lmd[0],sc_measured_lmd[1],sc_measured_lmd[0]+100,sc_measured_lmd[2], fill="red")    # rechts unten
    canvas1.create_line(sc_measured_lmd[0],sc_measured_lmd[1],sc_measured_lmd[0]+200,sc_measured_lmd[1], dash=(2, 4), fill="red")

def sc_text(tab):
    Label(tab, text=f"AP regular lambda:       {regular_lambda[0]}").place(x=990, y=200)
    Label(tab, text=f"AP measured lambda:  {user_lambda[0]}").place(x=990, y=220)
    Label(tab, text=f"compensating factor:   {size_comp_ratio}x").place(x=990, y=240)



# -----------AP
show_AP_tilt = IntVar()
ap_ratio = 0.4947   # ratio between x and y of ml images (frame,normal)
ap_size = [1004, 0]     # canvas window size
ap_size[1] = int(ap_size[0] * ap_ratio)
ap_mm = 62
ap_bregma = [407, 71]
raw_ap_aim_px = [ap_bregma[0] - user_aim[0] * ap_mm, ap_bregma[1] - user_aim[2] * ap_mm]  # AP aim coordinates to pixels coordinates
corr_ap_aim_px = [ap_bregma[0] - raw_ap[0] * ap_mm, ap_bregma[1] - raw_ap[2] * ap_mm]   # only after AP calc

def ap_image_tilt_calc(ap_size):
    # tilt calculations
    ap_origin = [ap_size[0] / 2, ap_size[1] / 2]
    ap_ori2breg = math.sqrt((ap_origin[0] - ap_bregma[0])**2 + (ap_origin[1] - ap_bregma[1])**2)
    ap_ori2breg_angle = math.degrees(math.acos((ap_origin[0] - ap_bregma[0]) / ap_ori2breg))
    ap_ori2breg_ang_tilt = ap_ori2breg_angle - alpha_AP
    ap_bregma_tilt = [ap_origin[0] - math.cos(math.radians(ap_ori2breg_ang_tilt)) * ap_ori2breg,
                    ap_origin[1] - math.sin(math.radians(ap_ori2breg_ang_tilt)) * ap_ori2breg]
    ap_bregma_tilt_diff = [ap_bregma[0] - ap_bregma_tilt[0], ap_bregma[1] - ap_bregma_tilt[1]]

    return ap_bregma_tilt_diff

ap_bregma_tilt_diff = ap_image_tilt_calc(ap_size)

# find correct png
ap_i = 0
while abs(ap_slices_list[ap_i] - abs(user_aim[1])) > abs(ap_slices_list[ap_i+1] - abs(user_aim[1])):
    ap_i += 1

# load the frame, straight png and tilted png
ap_frame = Image.open(f"{dir_path}\\ap\\frame.PNG").resize((ap_size[0],ap_size[1]))
ap_frame = ImageTk.PhotoImage(ap_frame)
ap_img = Image.open(f"{dir_path}\\ap\\{ap_slices_list[ap_i]}.PNG").resize((ap_size[0] - 72,int(ap_size[1] - (70*ap_ratio))))
ap_img = ImageTk.PhotoImage(ap_img)
ap_img_tilt = Image.open(f"{dir_path}\\ap\\{ap_slices_list[ap_i]}.PNG").resize((ap_size[0] - 72,int(ap_size[1] - (70*ap_ratio)))).rotate(alpha_AP)
ap_img_tilt = ImageTk.PhotoImage(ap_img_tilt)

def show_ap_tab(tab):
    tabControl.add(tab, text="  AP tilt  ")

    global canvas2
    canvas2 = Canvas(tab, width=ap_size[0], height=ap_size[1])
    canvas2.pack(anchor=W)
    canvas2.create_image(ap_size[0] / 2, ap_size[1] / 2, anchor=CENTER, image=ap_img, tags="ap")
    canvas2.create_image(ap_size[0] / 2, ap_size[1] / 2, anchor=CENTER, image=ap_frame, tags="frame")

    Label(tab, text=f"AP tilt degree: {alpha_AP}°").place(x=1050, y=200)
    Label(tab, text=f"raw aim coordinates:   {user_aim}").place(x=1050, y=220)
    Label(tab, text=f"corr. aim coordinates: {raw_ap}").place(x=1050, y=240)

    ap_lines()
    ap_check = Checkbutton(tab2,text='show AP tilt compensation',command=ap_tilt,variable=show_AP_tilt,onvalue=1,offvalue=0)
    ap_check.place(x=1050, y=100)
    summary(tab)

def ap_lines():
    middle_line = canvas2.create_line(30, 71, 970, 71)
    bregma_line = canvas2.create_line(407, 0, 407, 500)
    raw_aim_x = canvas2.create_line(raw_ap_aim_px[0], 15, raw_ap_aim_px[0], raw_ap_aim_px[1], dash=(2, 4))
    raw_aim_y = canvas2.create_line(raw_ap_aim_px[0], raw_ap_aim_px[1], ap_size[0]-30, raw_ap_aim_px[1], dash=(2, 4))

def ap_tilt_lines():
    corr_middle_line = canvas2.create_line(ap_bregma[0] + math.cos(math.radians(alpha_AP)) * 800,
                                           ap_bregma[1] - math.sin(math.radians(alpha_AP)) * 800,
                                           ap_bregma[0] - math.cos(math.radians(alpha_AP)) * 800,
                                           ap_bregma[1] + math.sin(math.radians(alpha_AP)) * 800, fill="red")
    corr_aim_y = canvas2.create_line(corr_ap_aim_px[0], 15, corr_ap_aim_px[0], corr_ap_aim_px[1], dash=(2, 4), fill="red")
    corr_aim_x = canvas2.create_line(corr_ap_aim_px[0], corr_ap_aim_px[1], ap_size[0]-30, corr_ap_aim_px[1], dash=(2, 4), fill="red")

def ap_tilt():
    if show_AP_tilt.get() == 0:
        canvas2.delete("ap_tilt")
        canvas2.create_image(ap_size[0] / 2, ap_size[1] / 2, anchor=CENTER,image=ap_img, tags="ap")
        canvas2.create_image(ap_size[0] / 2, ap_size[1] / 2, anchor=CENTER, image=ap_frame, tags="ap_frame")
        ap_lines()
        ap_tilt_lines()

    elif show_AP_tilt.get() == 1:
        canvas2.delete("all")
        canvas2.create_image(ap_size[0]/2 + ap_bregma_tilt_diff[0],  # abweichung due to tilt
                            ap_size[1]/2 + ap_bregma_tilt_diff[1],
                            anchor=CENTER,image=ap_img_tilt, tags="ap_tilt")
        canvas2.create_image(ap_size[0] / 2, ap_size[1] / 2, anchor=CENTER, image=ap_frame, tags="ap_frame")
        ap_lines()
        ap_tilt_lines()




# --------------ML
show_ML_tilt = IntVar()
ml_ratio = 0.5916  # ratio between x and y of ml images (frame,normal)
ml_size = [1004, int(1004 * ml_ratio)]  # canvas window size
ml_mm = 92  # how many pixels is 1 mm

# find correct png
ml_i = 0
while abs(ml_slices_list[ml_i] - user_aim[0]) > abs(ml_slices_list[ml_i+1] - user_aim[0]) and ml_i < len(ml_slices_list)-1:
    ml_i += 1

# load the frame, straight png and tilted png
ml_frame = Image.open(f"{dir_path}\\ml\\frame.PNG").resize((ml_size[0],ml_size[1]))
ml_frame = ImageTk.PhotoImage(ml_frame)
ml_img = Image.open(f"{dir_path}\\ml\\{ml_slices_list[ml_i]}.PNG").resize((ml_size[0] - 78,int(ml_size[1] - (78*ml_ratio))))
ml_img = ImageTk.PhotoImage(ml_img)
ml_img_tilt = Image.open(f"{dir_path}\\ml\\{ml_slices_list[ml_i]}.PNG").resize((ml_size[0] - 78,int(ml_size[1] - (78*ml_ratio)))).rotate(-delta_ML)
ml_img_tilt = ImageTk.PhotoImage(ml_img_tilt)

# ML aim coordinates to pixels coordinates
raw_ml_aim_px = [ml_size[0]/2 + user_aim[1] * ml_mm, -user_aim[2] * ml_mm]
corr_ml_aim_px = [ml_size[0]/2 + raw_ml[1] * ml_mm, -raw_ml[2] * ml_mm]

def show_ml_tab(tab):
    tabControl.add(tab, text="  ML tilt  ")
    ml_canvas(tab)
    ml_text(tab)
    ml_check = Checkbutton(tab3, text='show ML tilt compensation', command=ml_tilt, variable=show_ML_tilt, onvalue=1,
                           offvalue=0)
    ml_check.place(x=1050, y=100)
    summary(tab)

def ml_canvas(tab):
    global canvas
    canvas = Canvas(tab, width=ml_size[0], height=ml_size[1])
    canvas.pack(anchor=W)
    canvas.create_image(ml_size[0]/2,ml_size[1]/2,anchor=CENTER,image=ml_img, tags="ml")
    canvas.create_image(ml_size[0]/2,ml_size[1]/2,anchor=CENTER,image=ml_frame, tags="ml_frame")
    ml_lines()

def ml_text(tab):
    Label(tab, text=f"ML tilt degree: {delta_ML}°").place(x=1050, y=200)
    Label(tab, text=f"raw aim coordinates:   {user_aim}").place(x=1050, y=220)
    Label(tab, text=f"corr. aim coordinates: {raw_ml}").place(x=1050, y=240)

def ml_lines():
    middle_line = canvas.create_line(ml_size[0] / 2, 0, ml_size[0] / 2, 550)
    raw_aim_x = canvas.create_line(raw_ml_aim_px[0], 15, raw_ml_aim_px[0], raw_ml_aim_px[1], dash=(2, 4))
    raw_aim_y = canvas.create_line(raw_ml_aim_px[0], raw_ml_aim_px[1], ml_size[0]-30, raw_ml_aim_px[1], dash=(2, 4))

def ml_tilt_lines():
    corr_middle_line = canvas.create_line(ml_size[0] / 2, 0,
                                          ml_size[0]/2 + math.sin(math.radians(-delta_ML)) * ml_size[1],
                                          math.cos(math.radians(-delta_ML)) * ml_size[1], fill="red")
    corr_aim_x = canvas.create_line(corr_ml_aim_px[0], 15, corr_ml_aim_px[0], corr_ml_aim_px[1], dash=(2, 4), fill="red")
    corr_aim_y = canvas.create_line(corr_ml_aim_px[0], corr_ml_aim_px[1], ml_size[0]-30, corr_ml_aim_px[1], dash=(2, 4), fill="red")

def ml_tilt():
    if show_ML_tilt.get() == 0:
        canvas.delete("ml_tilt")
        canvas.create_image(ml_size[0] / 2, ml_size[1] / 2, anchor=CENTER,image=ml_img, tags="ml")
        canvas.create_image(ml_size[0] / 2, ml_size[1] / 2, anchor=CENTER, image=ml_frame, tags="ml_frame")
        ml_lines()
        ml_tilt_lines()
    elif show_ML_tilt.get() == 1:
        canvas.delete("ml")
        canvas.create_image(ml_size[0]/2 + math.sin(math.radians(-delta_ML)) * ml_size[1]/2, #abweichung due to tilt
                            math.cos(math.radians(delta_ML)) * ml_size[1]/2,
                            anchor=CENTER,image=ml_img_tilt, tags="ml_tilt")
        canvas.create_image(ml_size[0] / 2, ml_size[1] / 2, anchor=CENTER, image=ml_frame, tags="ml_frame")
        ml_lines()
        ml_tilt_lines()


def output():
    if user_check_size_comp:
        show_sc_tab(tab1)
    if user_check_pitch:
        show_ap_tab(tab2)
    if user_check_roll:
        show_ml_tab(tab3)

    print('output done')
    root.mainloop()

output()
