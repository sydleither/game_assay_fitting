from ast import literal_eval

from game_assay.game_analysis import read_overview_xlsx


def get_cell_types(exp_name):
    parts = exp_name.split("_")
    s = f"{parts[1]}-{parts[2].lower()}"
    r = f"{parts[4]}-{parts[5]}"
    return s, r


def get_growth_rate_window(data_dir, exp_name):
    data = read_overview_xlsx(data_dir, exp_name)
    return literal_eval(data["Growth Rate Window"])


def abm_parameter_map():
    return {
        "A": "p_SS",
        "B": "p_SR",
        "C": "p_RS",
        "D": "p_RR",
        "A_00": "a_SS",
        "A_01": "a_SR",
        "A_10": "a_RS",
        "A_11": "a_RR",
        "r_0": "r_S",
        "r_1": "r_R",
        "k_0": "k_S",
        "k_1": "k_R",
    }
