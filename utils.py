from string import ascii_uppercase


solver_kws = {
    "method": "RK45",
    "absErr": 1.0e-6,
    "relErr": 1.0e-4,
    "suppressOutputB": True,
    "max_step": 1,
    "dt": 0.01,
}

optimiser_kws = {
    "method": "least_squares",
    "xtol": 1e-8,
    "ftol": 1e-8,
    "max_nfev": 1000,
    "nan_policy": "omit",
    "verbose": 0,
}


def get_cell_types(exp_name):
    parts = exp_name.split("_")
    s = f"{parts[1]}-{parts[2].lower()}"
    r = f"{parts[4]}-{parts[5]}"
    return s, r


def get_plate_structure():
    seeding = [0.1, 0.3, 0.5, 0.7, 0.9]
    colids = [2, 3, 4, 5, 6]
    rowids = ascii_uppercase[1:4]
    return seeding, colids, rowids


def get_parameter_names():
    return [
        "p_SS",
        "p_SR",
        "p_RS",
        "p_RR",
        "a_SS",
        "a_SR",
        "a_RS",
        "a_RR",
        "r_S",
        "r_R",
    ]


def get_parameter_ranges(model):
    if model == "replicator":
        return [(0.03, 0.05)] * 4
    elif model == "lotka-volterra":
        return [(0.03, 0.05)] * 2 + [(-4e-6, -2e-7)] * 4


def get_colors():
    return {
        "Sensitive Wins": "#4C956C",
        "Coexistence": "#C28367",
        "Bistability": "#047495",
        "Resistant Wins": "#EF7C8E",
        "Unbounded Growth": "#767567",
        "Neutrality": "#767567"
    }


def label_data_type(data_dir):
    noisy = "Noisy " if "noisy" in data_dir else ""
    if "ode_egt" in data_dir:
        return f"{noisy}Exponential Growth ODE"
    if "ode_lv" in data_dir:
        return f"{noisy}Lotka-Volterra ODE"
    if "abm" in data_dir and "_" in data_dir:
        return data_dir.split("/")[1].split("_")[1].title() + " Spatial Agent-Based Model"
    return data_dir.replace("data/", "").title().replace("_", " ")
