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
