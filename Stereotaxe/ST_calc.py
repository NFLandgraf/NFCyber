import math, os
dir_path = os.getcwd()


decnumb = 2
raw_sc = size_comp_ratio = raw_ap = alpha = raw_ml = delta = aim = aim_sc = aim_ap = aim_ml = 999


def raw_calc(normal_lmd, measured_lmd, raw_aim, measured_ML_R, measured_ML_L):
    global raw_sc, size_comp_ratio, raw_ap, alpha, raw_ml, delta
    #size_comp
    raw_sc = ["", "", ""]
    size_comp_ratio = round(measured_lmd[0] / normal_lmd[0], decnumb)
    for i in range(3):
        raw_sc[i] = round(raw_aim[i] * size_comp_ratio, decnumb)

    #AP_tilt
    raw_ap = ["", "", ""]
    alpha = round(math.degrees(math.asin(measured_lmd[2] / -measured_lmd[0])), decnumb)  # Winkel bei bregma
    raw_beta = round(math.degrees(math.atan(raw_aim[2] / raw_aim[0])), decnumb)# calculates normal angle (beta) at bregma for aim
    raw_beta_hypot = math.sqrt(raw_aim[0] ** 2 + raw_aim[2] ** 2)
    raw_corrected_beta = round(-alpha + raw_beta, decnumb) # corrects beta angle for tilt

    raw_ap[0] = round(-raw_beta_hypot * math.cos(math.radians(raw_corrected_beta)), decnumb)
    raw_ap[1] = round(raw_aim[1], decnumb)
    raw_ap[2] = round(-raw_beta_hypot * math.sin(math.radians(raw_corrected_beta)), decnumb)

    #ML_tilt
    raw_ml = ["", "", ""]
    delta_gegenkathete = measured_ML_L[2] - measured_ML_R[2]
    delta_ankathete = -measured_ML_L[1] + measured_ML_R[1]
    delta = round(math.degrees(math.atan(delta_gegenkathete / delta_ankathete)), decnumb)

    raw_gamma_ankathete = -raw_aim[2] # calculating ML angle of aim
    raw_gamma_gegenkathete = raw_aim[1]
    raw_gamma_hyp = math.sqrt(raw_gamma_ankathete ** 2 + raw_gamma_gegenkathete ** 2)
    raw_gamma = round(math.degrees(math.atan(raw_gamma_gegenkathete / raw_gamma_ankathete)), decnumb)

    raw_corrected_gamma = round(raw_gamma - delta, decnumb) # calculate correct aim
    raw_ml[0] = round(raw_aim[0], decnumb)
    raw_ml[1] = round(math.sin(math.radians(raw_corrected_gamma)) * raw_gamma_hyp, decnumb)
    raw_ml[2] = round(-math.cos(math.radians(raw_corrected_gamma)) * raw_gamma_hyp, decnumb)

def size_comp(aim, normal_lmd, measured_lmd): #based on AP position of lambda
    global aim_sc
    size_comp_ratio = round(measured_lmd[0] / normal_lmd[0], decnumb)
    # fake_aim just to prevent weird bug that didn`t allow to directly manipulate aim[x]
    fake_aim = ["","",""]
    for i in range(3):
        fake_aim[i] = round(aim[i] * size_comp_ratio, decnumb)
    aim_sc = fake_aim
    aim = fake_aim

    return aim

def AP_tilt(aim, raw_aim, measured_lmd): #calculates angle of AP tilt (alpha) at bregma
    global aim_ap
    aim[0] = 0.0001 if aim[0] == 0 else aim[0]
    alpha = round(math.degrees(math.asin(measured_lmd[2] / -measured_lmd[0])), decnumb) # Winkel bei bregma

    # calculates normal angle (beta) at bregma for aim
    beta = round(math.degrees(math.atan(aim[2] / aim[0])), decnumb)
    raw_beta = round(math.degrees(math.atan(raw_aim[2] / raw_aim[0])), decnumb)
    beta_hypot = math.sqrt(aim[0] ** 2 + aim[2] ** 2)
    raw_beta_hypot = math.sqrt(raw_aim[0] ** 2 + raw_aim[2] ** 2)

    # corrects beta angle for tilt
    corrected_beta = round(-alpha + beta, decnumb)
    fake_aim = ["","",""]
    fake_aim[0] = round(-beta_hypot * math.cos(math.radians(corrected_beta)), decnumb)
    fake_aim[1] = round(aim[1], decnumb)
    fake_aim[2] = round(-beta_hypot * math.sin(math.radians(corrected_beta)), decnumb)
    aim_ap = fake_aim
    aim = fake_aim

    return aim

def ML_tilt(aim, measured_ML_R, measured_ML_L):
    global aim_ml
    #skull ML tilt
    delta_gegenkathete = measured_ML_L[2] - measured_ML_R[2]
    delta_ankathete = -measured_ML_L[1] + measured_ML_R[1]
    delta = round(math.degrees(math.atan(delta_gegenkathete / delta_ankathete)), decnumb)

    # calculating ML angle of aim
    gamma_ankathete = -aim[2]
    gamma_gegenkathete = aim[1]
    gamma_hyp = math.sqrt(gamma_ankathete**2 + gamma_gegenkathete**2)
    gamma = round(math.degrees(math.atan(gamma_gegenkathete / gamma_ankathete)), decnumb)

    #calculate correct aim
    corrected_gamma = round(gamma - delta, decnumb)
    fake_aim = ["","",""]
    fake_aim[0] = round(aim[0], decnumb)
    fake_aim[1] = round(math.sin(math.radians(corrected_gamma)) * gamma_hyp, decnumb)
    fake_aim[2] = round(-math.cos(math.radians(corrected_gamma)) * gamma_hyp, decnumb)
    aim_ml = fake_aim
    aim = fake_aim

    return aim



def print_out_detailed(raw_aim, measured_lmd, normal_lmd, measured_ML_L, measured_ML_R, check_size_comp,
                       check_AP_tilt, alpha_AP, check_ML_tilt, delta_ML):
    fb = "Stereotaxe_Feedback.txt"
    with open (fb, "w") as f:
        f.write(f"Raw aim coordinates: {raw_aim}\n"
    		f"Measured lambda: {measured_lmd}\n"
    		f"Normal lambda: {normal_lmd}\n"
    		f"Measured ML_L: {measured_ML_L}\n"
    		f"Measured ML_R: {measured_ML_R}\n")

        # size_comp
    if check_size_comp:
        if size_comp_ratio != 1:
            size = "(smaller than usual)" if size_comp_ratio < 1 else "(bigger than usual)"
            with open(fb, "a") as f:
                f.write(f"\nSKULL SIZE COMPENSATION:\nFactor x = {round(size_comp_ratio, decnumb)} {size}\n"
                        f"Corrected aim coordinates: {aim_sc}\n")
        else:
            with open(fb, "a") as f:
                f.write("\nSKULL SIZE COMPENSATION:\n"
                        "No compensation necessary bc AP(measured lambda) = AP(normal lambda).\n")
    if check_AP_tilt:
        # AP_tilt
        if alpha_AP != 0:
            which_AP_tilt = "(Bregma < Lambda)" if alpha_AP > 0 else "(Bregma > Lambda)"
            with open(fb, "a") as f:
                f.write(f"\nAP TILT\n"
                        f"Skull AP tilt alpha: {round(alpha_AP, 2)} degree {which_AP_tilt}\n"
                        f"Corrected aim: {aim_ap}\n")
        else:
            with open(fb, "a") as f:
                f.write("\nAP TILT:\n"
                        "No compensation necessary bc DV(bregma) = DV(lambda).\n")

        # ML_tilt
    if check_ML_tilt:
        if delta_ML != 0:
            which_ML_tilt = "(left > right)" if delta_ML > 0 else "(right > left)"
            with open (fb, "a") as f:
                f.write(f"\nML TILT\n"
                        f"Skull ML tilt delta: {round(delta_ML, decnumb)} degrees {which_ML_tilt}\n"
                        f"Corrected aim: {aim_ml}\n")
        else:
            with open(fb, "a") as f:
                f.write("\nML TILT:\n"
                        "No compensation necessary bc DV(ML_L) = DV (ML_R).\n")


def calc(check_size_comp, check_AP_tilt, check_ML_tilt, aim, raw_aim, normal_lmd, measured_lmd, measured_ML_R, measured_ML_L):
    raw_calc(normal_lmd, measured_lmd, raw_aim, measured_ML_R, measured_ML_L)
    if check_size_comp:
        aim = size_comp(aim, normal_lmd, measured_lmd)
    if check_AP_tilt:
        aim = AP_tilt(aim, raw_aim, measured_lmd)
    if check_ML_tilt:
        aim = ML_tilt(aim, measured_ML_R, measured_ML_L)
    print("calc_done")

    return raw_sc, size_comp_ratio, raw_ap, alpha, raw_ml, delta, aim, aim_sc, aim_ap, aim_ml
