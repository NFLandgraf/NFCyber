from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image
import math, pickle, os
a = 1
root = Tk()
root.geometry("1400x650")
root.title("Output")
dir_path = os.getcwd()


# defining all valuable using pickle
with open("internal_pickle.py", "rb") as f:
    data = pickle.load(f)
    check_size_comp = data[0]
    check_AP_tilt = data[1]
    check_ML_tilt = data[2]
    normal_lmd = data[3]
    measured_lmd = data[4]
    measured_ML_R = data[5]
    measured_ML_L = data[6]
    raw_aim = data[7]
    aim = data[8]
    raw_sc = data[9]
    size_comp_ratio = data[10]
    raw_ap = data[11]
    alpha_AP = data[12]
    raw_ml = data[13]
    delta_ML = -data[14]
    aim_sc = data[15]
    aim_ap = data[16]
    aim_ml = data[17]

def summary(tab):
    Label(tab, text="SUMMARY").place(x=990, y=360)
    Label(tab, text=f"Orig. aim:                                          {raw_aim}").place(x=990, y=380)
    if check_size_comp:
        Label(tab, text=f"Size comp.:   {size_comp_ratio}x -> adapt. aim: {aim_sc}").place(x=990, y=400)
    if check_AP_tilt:
        Label(tab, text=f"AP tilt:          {alpha_AP}° -> adapt. aim: {aim_ap}").place(x=990, y=420)
    if check_ML_tilt:
        Label(tab, text=f"ML tilt:         {delta_ML}° -> adapt. aim: {aim_ml}").place(x=990, y=440)
    Label(tab, text=f"Aim for coordinates {aim}").place(x=990, y=480)

# Tabs
tabControl = ttk.Notebook(root)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)
tab4 = ttk.Frame(tabControl)
tabControl.pack(expand=1, fill="both")


def do_sc(tab):
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
    Label(tab, text=f"AP regular lambda:       {normal_lmd[0]}").place(x=990, y=200)
    Label(tab, text=f"AP measured lambda:  {measured_lmd[0]}").place(x=990, y=220)
    Label(tab, text=f"compensating factor:   {size_comp_ratio}x").place(x=990, y=240)



# AP
show_AP_tilt = IntVar()
ap_ratio = 0.4947   # ratio between x and y of ml images (frame,normal)
ap_size = [1004, 0]     # canvas window size
ap_size[1] = int(ap_size[0] * ap_ratio)
ap_mm = 62
ap_bregma = [407, 71]
raw_ap_aim_px = [ap_bregma[0] - raw_aim[0] * ap_mm, ap_bregma[1] - raw_aim[2] * ap_mm]  # AP aim coordinates to pixels coordinates
corr_ap_aim_px = [ap_bregma[0] - raw_ap[0] * ap_mm, ap_bregma[1] - raw_ap[2] * ap_mm]   # only after AP calc

# tilt calculations
ap_origin = [ap_size[0] / 2, ap_size[1] / 2]
ap_ori2breg = math.sqrt((ap_origin[0] - ap_bregma[0])**2 + (ap_origin[1] - ap_bregma[1])**2)
ap_ori2breg_angle = math.degrees(math.acos((ap_origin[0] - ap_bregma[0]) / ap_ori2breg))
ap_ori2breg_ang_tilt = ap_ori2breg_angle - alpha_AP
ap_bregma_tilt = [ap_origin[0] - math.cos(math.radians(ap_ori2breg_ang_tilt)) * ap_ori2breg,
                  ap_origin[1] - math.sin(math.radians(ap_ori2breg_ang_tilt)) * ap_ori2breg]
ap_bregma_tilt_diff = [ap_bregma[0] - ap_bregma_tilt[0], ap_bregma[1] - ap_bregma_tilt[1]]

# AP images
ap_slices_list = [0.04, 0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84, 0.96, 1.08, 1.2, 1.32, 1.44, 1.56, 1.68, 1.8, 1.92,
                  2.04, 2.16, 2.28, 2.4, 2.52, 2.64, 2.76, 2.88, 3, 3.12, 3.25, 3.36, 3.44, 3.6, 3.72, 999999]

ap_i = 0
while abs(ap_slices_list[ap_i] - abs(raw_aim[1])) > abs(ap_slices_list[ap_i+1] - abs(raw_aim[1])):
    ap_i += 1

ap_frame = Image.open(f"{dir_path}\\ap\\frame.PNG").resize((ap_size[0],ap_size[1]))
ap_frame = ImageTk.PhotoImage(ap_frame)
ap_img = Image.open(f"{dir_path}\\ap\\{ap_slices_list[ap_i]}.PNG").resize((ap_size[0] - 72,int(ap_size[1] - (70*ap_ratio))))
ap_img = ImageTk.PhotoImage(ap_img)
ap_img_tilt = Image.open(f"{dir_path}\\ap\\{ap_slices_list[ap_i]}.PNG").resize((ap_size[0] - 72,int(ap_size[1] - (70*ap_ratio)))).rotate(alpha_AP)
ap_img_tilt = ImageTk.PhotoImage(ap_img_tilt)


def do_ap(tab):
    tabControl.add(tab, text="  AP tilt  ")

    global canvas2
    canvas2 = Canvas(tab, width=ap_size[0], height=ap_size[1])
    canvas2.pack(anchor=W)
    canvas2.create_image(ap_size[0] / 2, ap_size[1] / 2, anchor=CENTER, image=ap_img, tags="ap")
    canvas2.create_image(ap_size[0] / 2, ap_size[1] / 2, anchor=CENTER, image=ap_frame, tags="frame")

    Label(tab, text=f"AP tilt degree: {alpha_AP}°").place(x=1050, y=200)
    Label(tab, text=f"raw aim coordinates:   {raw_aim}").place(x=1050, y=220)
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



# ML
show_ML_tilt = IntVar()
ml_ratio = 0.5916  # ratio between x and y of ml images (frame,normal)
ml_size = [1004, int(1004 * ml_ratio)]  # canvas window size
ml_mm = 92  # how many pixels is 1 mm

# ML images
ml_slices_list = [4.28, 3.92, 3.56, 3.08, 2.96, 2.8, 2.68, 2.58, 2.46, 2.34, 2.22, 2.1, 1.98, 1.94, 1.78, 1.7, 1.54,
                  1.42, 1.34, 1.18, 1.1, 0.98, 0.86, 0.74, 0.62, 0.5, 0.38, 0.26, 0.14, 0.02, -0.1, -0.22, -0.34, -0.46,
                  -0.58, -0.7, -0.82, -0.94, -1.06, -1.22, -1.34, -1.46, -1.58, -1.7, -1.82, -1.94, -2.06, -2.18, -2.3,
                  -2.46, -2.54, -2.7, -2.8, -2.92, -3.08, -3.16, -3.28, -3.4, -3.52, -3.64, -3.8, -3.88, -4.04, -4.16,
                  -4.24, -4.36, -4.48, -4.6, -4.72, -4.84, -4.96, -5.02, -5.2, -5.34, -5.4, -5.52, -5.68, -5.8, -5.88,
                  -6, -6.12, -6.24, -6.36, -6.48, -6.64, -6.72, -6.84, -6.96, -7.08, -7.2, -7.32, -7.48, -7.56, -7.64,
                  -7.76, -7.92, -8, -8.12, -8.24, 999]
ml_i = 0
while abs(ml_slices_list[ml_i] - raw_aim[0]) > abs(ml_slices_list[ml_i+1] - raw_aim[0]) and ml_i < len(ml_slices_list)-1:
    ml_i += 1

ml_frame = Image.open(f"{dir_path}\\ml\\frame.PNG").resize((ml_size[0],ml_size[1]))
ml_frame = ImageTk.PhotoImage(ml_frame)
ml_img = Image.open(f"{dir_path}\\ml\\{ml_slices_list[ml_i]}.PNG").resize((ml_size[0] - 78,int(ml_size[1] - (78*ml_ratio))))
ml_img = ImageTk.PhotoImage(ml_img)
ml_img_tilt = Image.open(f"{dir_path}\\ml\\{ml_slices_list[ml_i]}.PNG").resize((ml_size[0] - 78,int(ml_size[1] - (78*ml_ratio)))).rotate(delta_ML)
ml_img_tilt = ImageTk.PhotoImage(ml_img_tilt)

# ML aim coordinates to pixels coordinates
raw_ml_aim_px = [ml_size[0]/2 + raw_aim[1] * ml_mm, -raw_aim[2] * ml_mm]
corr_ml_aim_px = [ml_size[0]/2 + raw_ml[1] * ml_mm, -raw_ml[2] * ml_mm]

def do_ml(tab):
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
    Label(tab, text=f"raw aim coordinates:   {raw_aim}").place(x=1050, y=220)
    Label(tab, text=f"corr. aim coordinates: {raw_ml}").place(x=1050, y=240)

def ml_lines():
    middle_line = canvas.create_line(ml_size[0] / 2, 0, ml_size[0] / 2, 550)
    raw_aim_x = canvas.create_line(raw_ml_aim_px[0], 15, raw_ml_aim_px[0], raw_ml_aim_px[1], dash=(2, 4))
    raw_aim_y = canvas.create_line(raw_ml_aim_px[0], raw_ml_aim_px[1], ml_size[0]-30, raw_ml_aim_px[1], dash=(2, 4))

def ml_tilt_lines():
    corr_middle_line = canvas.create_line(ml_size[0] / 2, 0,
                                          ml_size[0]/2 + math.sin(math.radians(delta_ML)) * ml_size[1],
                                          math.cos(math.radians(delta_ML)) * ml_size[1], fill="red")
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
        canvas.create_image(ml_size[0]/2 + math.sin(math.radians(delta_ML)) * ml_size[1]/2, #abweichung due to tilt
                            math.cos(math.radians(delta_ML)) * ml_size[1]/2,
                            anchor=CENTER,image=ml_img_tilt, tags="ml_tilt")
        canvas.create_image(ml_size[0] / 2, ml_size[1] / 2, anchor=CENTER, image=ml_frame, tags="ml_frame")
        ml_lines()
        ml_tilt_lines()


if check_size_comp:
    do_sc(tab1)
if check_AP_tilt:
    do_ap(tab2)
if check_ML_tilt:
    do_ml(tab3)

print("output_done")
root.mainloop()
