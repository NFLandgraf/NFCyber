import pickle, os
dir_path = os.getcwd()

input_done = False
import ST_GUI

with open("internal_pickle.py", "rb") as f:
    data = pickle.load(f)
    check_size_comp = data[0]
    check_AP_tilt = data[1]
    check_ML_tilt = data[2]
    normal_lmd = data[3]
    raw_aim = data[4]
    aim = data[5]
    measured_lmd = data[6]
    measured_ML_R = data[7]
    measured_ML_L = data[8]
    input_done = data[9]


if input_done:
    import ST_calc
    # receive input from Stereotaxe_calc
    raw_sc, size_comp_ratio, raw_ap, alpha_AP, raw_ml, delta_ML, aim, aim_sc, aim_ap, aim_ml = \
        ST_calc.calc(check_size_comp, check_AP_tilt, check_ML_tilt, aim, raw_aim, normal_lmd, measured_lmd, measured_ML_R, measured_ML_L)

    # prepare pickle input
    data = [check_size_comp, check_AP_tilt, check_ML_tilt, normal_lmd, measured_lmd, measured_ML_R, measured_ML_L, raw_aim,
            aim, raw_sc, size_comp_ratio, raw_ap, alpha_AP, raw_ml, delta_ML, aim_sc, aim_ap, aim_ml]
    with open("internal_pickle.py", "wb") as f:
        pickle.dump(data, f)

    ST_calc.print_out_detailed(raw_aim, measured_lmd, normal_lmd, measured_ML_L, measured_ML_R, check_size_comp,
                                       check_AP_tilt, alpha_AP, check_ML_tilt, delta_ML)

    # start Output
    import ST_output

    ST_calc.print_out_detailed(raw_aim, measured_lmd, normal_lmd, measured_ML_L, measured_ML_R, check_size_comp,
                           check_AP_tilt, alpha_AP, check_ML_tilt, delta_ML)

