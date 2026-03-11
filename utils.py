from string import ascii_uppercase


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
        "k_S",
        "k_R",
    ]
