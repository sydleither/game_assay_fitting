# --------------------------------------------------------------------
# Functions to help with analysing game assay data
# --------------------------------------------------------------------
import os
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import game_assay.myUtils as utils


# ---------------------------------------------------------------------------------------------------------------
def run_cellprofiler_on_well(
    well_id,
    cellprofiler_path,
    pipeline_file,
    dir_to_analyse,
    output_dir,
    create_out_dir=True,
    var_name_well="Metadata_Well",
    print_command=False,
    log_level=50,
    suppress_output=True,
):
    curr_out_dir = output_dir
    if create_out_dir:
        curr_out_dir = os.path.join(output_dir, well_id)
        os.makedirs(curr_out_dir, exist_ok=True)

    command = [
        cellprofiler_path,
        "-c",
        "-r",
        "-p",
        pipeline_file,
        "-i",
        dir_to_analyse,
        "-o",
        curr_out_dir,
        "-g",
        f"{var_name_well}={well_id}",
        "-L",
        str(log_level),
    ]

    if print_command:
        print("Running:", command)

    kws_dict = (
        {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if suppress_output else {}
    )
    subprocess.run(command, **kws_dict)


# ---------------------------------------------------------------------------------------------------------------
def load_cellprofiler_data(
    input_file,
    imaging_frequency=4,
    tags=["gfp", "texasred"],
    pop_names=["S", "R"],
    ignore_column=11,
    long_format=True,
):
    """
    Load cellprofiler data and return a pandas dataframe in either wide or long format (as requested)
    input_file: path to the cellprofiler output file to be loaded
    imaging_frequency: imaging frequency in hours
    tags: list of tags used in the cellprofiler output file (e.g. gfp and texasred)
    pop_names: list of population names (e.g. S and R)
    long_format: if True, return a long dataframe with columns: Time, WellId, CellType, Count
    """
    if type(input_file) == str:  # Allow user to input file name or data frame directly
        counts_raw_df = pd.read_csv(input_file, index_col=False)
    else:
        counts_raw_df = input_file
    counts_raw_df["WellId"] = counts_raw_df["FileName_%s" % tags[0]].str.split("_").str[0]
    counts_raw_df["RowId"] = counts_raw_df["WellId"].str[0]
    counts_raw_df["ColumnId"] = counts_raw_df["WellId"].str[1:].astype(int)
    if ignore_column is not None:
        counts_raw_df = counts_raw_df[counts_raw_df["ColumnId"] != 11]
    counts_raw_df["ImageId"] = (
        counts_raw_df["FileName_%s" % tags[0]]
        .str.split("_")
        .str[-1]
        .str.split(".")
        .str[0]
        .astype(int)
    )
    counts_raw_df["Time"] = (counts_raw_df["ImageId"] - 1) * imaging_frequency
    counts_raw_df["Count_total"] = (
        counts_raw_df["Count_%s_objects" % tags[0]] + counts_raw_df["Count_%s_objects" % tags[1]]
    )
    counts_raw_df["Frequency_%s_objects" % tags[0]] = (
        counts_raw_df["Count_%s_objects" % tags[0]] / counts_raw_df["Count_total"]
    )
    counts_raw_df["Frequency_%s_objects" % tags[1]] = (
        counts_raw_df["Count_%s_objects" % tags[1]] / counts_raw_df["Count_total"]
    )
    # Convert to long format if requested
    if long_format:
        tmp_list = []
        for measure in ["Count", "Frequency"]:
            counts_raw_df_long = counts_raw_df.melt(
                id_vars=["Time", "WellId", "ImageId", "RowId", "ColumnId"],
                value_vars=[
                    "%s_%s_objects" % (measure, tags[0]),
                    "%s_%s_objects" % (measure, tags[1]),
                ],
                var_name="CellType",
                value_name=measure,
            )
            counts_raw_df_long.replace(
                {
                    "CellType": {
                        "%s_%s_objects" % (measure, tags[0]): pop_names[0],
                        "%s_%s_objects" % (measure, tags[1]): pop_names[1],
                    }
                },
                inplace=True,
            )
            counts_raw_df_long.reset_index(drop=True, inplace=True)
            tmp_list.append(counts_raw_df_long)
        counts_raw_df = pd.merge(
            tmp_list[0],
            tmp_list[1],
            on=["Time", "WellId", "ImageId", "RowId", "ColumnId", "CellType"],
        )
        counts_raw_df.reset_index(drop=True, inplace=True)
    return counts_raw_df


# ---------------------------------------------------------------------------------------------------------------
def map_well_to_experimental_condition(well_id, experimental_conditions_df):
    """
    Retrieves the experimental condition for a well from the layout excel
    file in the structure used for the game pipeline.
    """
    row_id = well_id[0].lower()
    column_id = int(well_id[1:])
    return experimental_conditions_df.loc[row_id, column_id]


# ---------------------------------------------------------------------------------------------------------------
def plot_data(
    dataDf,
    x="Time",
    y="Confluence",
    style=None,
    hue=None,
    estimator=None,
    err_style="bars",
    errorbar=("ci", 95),
    linecolor="black",
    linewidth=2,
    palette="tab10",
    legend=False,
    markerstyle="o",
    markersize=12,
    markeredgewidth=0.5,
    markeredgecolor="black",
    lineplot_kws={},
    plot_drug=True,
    treatment_column="DrugConcentration",
    treatment_notation_mode="post",
    drug_bar_position=0.85,
    drug_colour="#683073",
    xlim=None,
    ylim=None,
    y2lim=1,
    title="",
    label_axes=False,
    ax=None,
    figsize=(10, 8),
    **kwargs,
):
    """
    Plot longitudinal data (e.g. cell counts), together with annotations of drug administration.
    :param dataDf: Pandas data frame with longitudinal data to be plotted.
    :param x: Name (str) of the column with the time information.
    :param y: Name (str) of the column with the metric to be plotted on the y-axis (e.g. cell count, confluence, etc).
    :param style: Name (str) of the column with the style information (e.g. cell type).
    :param hue: Name (str) of the column with the hue information (e.g. drug treatment).
    :param estimator: Name (str) of the estimator to use for the line plot (e.g. mean, median).
    :param err_style: Name (str) of the error style to use for the line plot (e.g. bars, band).
    :param errorbar: Tuple with the type of error bar to plot and the confidence interval (e.g. ('ci', 95)).
    :param linecolor: Colour of the line plot.
    :param linewidth: Width of the line plot.
    :param palette: Colour palette to use for the line plot.
    :param legend: Boolean; whether or not to show the legend.
    :param markerstyle: Style of the markers for the data points.
    :param markersize: Size of the markers for the data points.
    :param markeredgewidth: Width of the marker edge.
    :param markeredgecolor: Colour of the marker edge.
    :param lineplot_kws: Dictionary with additional keyword arguments for the line plot.
    :param plot_drug: Boolean; whether or not to plot the treatment schedule.
    :param treatment_column: Name (str) of the column with the information about the dose administered.
    :param treatment_notation_mode: Name (str) of the mode to use for the treatment notation (e.g. post, pre).
    :param drug_bar_position: Position of the drug bar when plotted across the top.
    :param drug_colour: Colour of the drug bar.
    :param xlim: x-axis limit.
    :param ylim: y-axis limit.
    :param y2lim: y2-axis limit.
    :param title: Title to put on the figure.
    :param label_axes: Boolean; whether or not to label the axes.
    :param ax: matplotlib axis to plot on. If none provided creates a new figure.
    :param figsize: Tuple, figure dimensions when creating new figure.
    :param kwargs: Other kwargs to pass to plotting functions.
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Set a number of defaults that I like
    lineplot_kws["errorbar"] = errorbar
    lineplot_kws["estimator"] = estimator
    lineplot_kws["hue"] = hue
    lineplot_kws["style"] = style
    lineplot_kws["err_style"] = err_style
    lineplot_kws["linecolor"] = lineplot_kws.get("linecolor", linecolor)
    lineplot_kws["linewidth"] = linewidth
    lineplot_kws["palette"] = palette
    lineplot_kws["legend"] = legend
    if hue == None:
        lineplot_kws["color"] = lineplot_kws["linecolor"]
        lineplot_kws.pop("palette")
    lineplot_kws.pop("linecolor", None)
    lineplot_kws["markerstyle"] = markerstyle
    lineplot_kws["markersize"] = markersize
    lineplot_kws["markeredgewidth"] = markeredgewidth
    lineplot_kws["markeredgecolor"] = markeredgecolor
    # Plot the data
    if (
        lineplot_kws["style"] == None
    ):  # If no style is specified, use the "marker" keyword. Otherwise points won't show up.
        lineplot_kws["marker"] = lineplot_kws["markerstyle"]
    else:
        lineplot_kws["markers"] = lineplot_kws["markerstyle"]
    lineplot_kws.pop("markerstyle")
    sns.lineplot(x=x, y=y, **lineplot_kws, ax=ax, data=dataDf)

    # Plot the drug concentration
    if plot_drug:
        plot_drug_bar(
            dataDf,
            ax,
            timeColumn=x,
            treatmentColumn=treatment_column,
            treatment_notation_mode=treatment_notation_mode,
            plotDrugAsBar=True,
            drugBarPosition=drug_bar_position,
            drugBarColour=drug_colour,
            y2lim=y2lim,
        )

    # Format the plot
    if xlim is not None:
        ax.set_xlim(0, xlim)
    if ylim is not None:
        ax.set_ylim(0, ylim)

    # Decorate the plot
    if label_axes == False:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_title(title)
    ax.tick_params(labelsize=kwargs.get("labelsize", 28))
    plt.tight_layout()


# ---------------------------------------------------------------------------------------------------------------
def plot_drug_bar(
    drug_data_df,
    ax,
    timeColumn="Time",
    treatmentColumn="DrugConcentration",
    treatment_notation_mode="post",
    plotDrugAsBar=True,
    drugBarPosition=0.85,
    y2lim=None,
    alpha=1.0,
    drugBarColour="black",
    decorateY2=False,
    zorder=1,
    **kwargs,
):
    """
    Plot the drug concentration as a bar on the top of the plot.
    drug_data_df: pandas dataframe containing the drug concentration data
    ax: matplotlib axis to plot on
    timeColumn: name of the column with the time information
    treatmentColumn: name of the column with the treatment information
    treatment_notation_mode: mode to use for the treatment notation (e.g. post, pre)
    plotDrugAsBar: boolean; whether or not to plot the drug
    drugBarPosition: position of the drug bar when plotted across the top
    y2lim: y2-axis limit
    alpha: transparency of the drug bar
    drugBarColour: colour of the drug bar
    decorateY2: boolean; whether or not to decorate the y2 axis
    zorder: zorder of the drug bar
    """

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    if plotDrugAsBar:
        drugConcentrationVec = utils.TreatmentListToTS(
            treatmentList=utils.ExtractTreatmentFromDf(
                drug_data_df,
                timeColumn=timeColumn,
                treatmentColumn=treatmentColumn,
                mode=treatment_notation_mode,
            ),
            tVec=drug_data_df[timeColumn],
        )
        drugConcentrationVec[drugConcentrationVec < 0] = 0
        max_height = (
            1 - drugBarPosition
        )  # Max height of the bar (above the drugBarPosition) relative to the y2 axis
        y2lim = (np.max(drugConcentrationVec) + 1e-12) if y2lim is None else y2lim
        drugConcentrationVec = np.array([x / y2lim for x in drugConcentrationVec])
        drugConcentrationVec = max_height * drugConcentrationVec + drugBarPosition
        ax2.fill_between(
            drug_data_df[timeColumn],
            drugBarPosition,
            drugConcentrationVec,
            step="post",
            color=drugBarColour,
            alpha=alpha,
            label="Drug Concentration",
            zorder=zorder,
        )
        ax2.axis("off")
    else:
        pass

    # Format y2 axis
    ax2.set_ylim([0, 1])
    ax2.tick_params(labelsize=28)
    if not decorateY2:
        ax2.set_yticklabels("")


# ---------------------------------------------------------------------------------------------------------------
def estimate_growth_rate(data_df, well_id=None, cell_type=None, growth_rate_window=[0, 24]):
    """
    Estimate the (exponential) growth rate of a cell population in a well.
    data_df: pandas dataframe containing the cell count data
    well_id: well identifier; if not None, assume data is from a single well
    cell_type: cell type identifier; if not None, assume data is from a single cell type
    growth_rate_window: time window for estimating the growth rate
    Returns: growth rate, intercept, lower bound of the growth rate, upper bound of the growth rate
    """
    # Filter the data
    curr_df = data_df.copy()
    if well_id is not None:
        curr_df = curr_df[curr_df["WellId"] == well_id]
    if cell_type is not None:
        curr_df = curr_df[curr_df["CellType"] == cell_type]
    curr_df = curr_df[
        curr_df["Time"].between(growth_rate_window[0], growth_rate_window[1], inclusive="both")
    ]

    # Fit a linear model to the log-transformed data. Use the theil-sen estimator
    x = curr_df["Time"].values - curr_df["Time"].values[0]  # Start the time at 0
    y = np.log(curr_df["Count"].values)  # Log-transform
    slope, intercept, low_slope, high_slope = stats.theilslopes(y, x)
    return slope, intercept, low_slope, high_slope


def calculate_fit(Y, Y_pred):
    return np.sqrt(np.mean((Y - Y_pred) ** 2)) / np.mean(Y)


# ---------------------------------------------------------------------------------------------------------------
def estimate_game_parameters(
    growth_rate_df,
    fraction_col="Fraction_Sensitive",
    growth_rate_col="GrowthRate",
    cell_type_col="CellType",
    cell_type_list=None,
    ci=0.95,
):
    """
    Estimate the game parameters from the growth rate data. The game parameters are the pay-off matrix entries,
    where we assume that the pay-off matrix is of the form:
    P = |p11 p12|
        |p21 p22|
    where pij is the pay-off for cell type i when interacting with cell type j. Which of the two cell types is
    Type 1 and Type 2, respectively, is determined by the order of the cell_type_list.
    Parameters
    ----------
    growth_rate_df : the growth rate data
    fraction_col : the column name for the population fraction
    growth_rate_col : the column name for the growth rate
    cell_type_col : the column name for the cell type
    cell_type_list : the list of cell types; used to determine the order of the pay-off matrix entries. If None, the order is determined by the order of the cell types in the data.
    method : the method to use for the estimation. Can be "ols" or "theil".
    ci : the confidence interval for the Theil estimator
    Returns
    -------
    params_dict : a dictionary with the pay-off matrix entries and the advantage of each cell type (game space position).
    """
    # Estimate the game parameters
    coeffs_dict = {}
    for cell_type in growth_rate_df[cell_type_col].unique():
        tmp_df = growth_rate_df[
            (growth_rate_df[cell_type_col] == cell_type)
            & (growth_rate_df["GrowthRate"].isna() == False)
        ]
        theil_result = stats.theilslopes(
            x=tmp_df[fraction_col], y=tmp_df[growth_rate_col], alpha=ci
        )
        best_fit_func = lambda x: (theil_result.slope * x) + theil_result.intercept
        Y_pred = theil_result.slope * tmp_df[fraction_col] + theil_result.intercept
        fit = calculate_fit(tmp_df[growth_rate_col], Y_pred)
        coeffs_dict[cell_type] = [
            best_fit_func(0),
            best_fit_func(1),
            theil_result.intercept,
            theil_result.slope,
            fit,
        ]
    # Transform into pay-off matrix entries. To do so, we need to find out the direction of the x-axis (i.e. whether
    # it's increasing for Type 1 or Type 2 as we go left to right).
    # The growth rate of the "index" population should be nan when their fraction is 0.
    try:
        no_deteced_growth_rate_df = growth_rate_df[growth_rate_df[growth_rate_col].isna()]
        avg_frac_when_no_growth_rate = no_deteced_growth_rate_df.groupby(cell_type_col).mean(
            numeric_only=True
        )[fraction_col]
        index_cell_type = avg_frac_when_no_growth_rate.idxmin()
    except:
        index_cell_type = cell_type_list[0]
    # Now we can compute the pay-off matrix entries
    cell_type_list = (
        growth_rate_df[cell_type_col].unique() if cell_type_list is None else cell_type_list
    )
    params_dict = {}
    for i, cell_type in enumerate(cell_type_list):
        pop_id = i + 1  # The population ID is 1-indexed (i.e. Type 1 and Type 2)
        if cell_type == index_cell_type:
            # The index cell type is the one whose self-interaction happens at fraction = 1
            params_dict["p%d%d" % (pop_id, pop_id)] = coeffs_dict[cell_type][1]
            params_dict["p%d%d" % (pop_id, pop_id % 2 + 1)] = coeffs_dict[cell_type][0]
            params_dict["r%d" % pop_id] = coeffs_dict[cell_type][2] + coeffs_dict[cell_type][3]
            params_dict["c%d%d" % (pop_id % 2 + 1, pop_id)] = -coeffs_dict[cell_type][3]
        else:
            params_dict["p%d%d" % (pop_id, pop_id)] = coeffs_dict[cell_type][0]
            params_dict["p%d%d" % (pop_id, pop_id % 2 + 1)] = coeffs_dict[cell_type][1]
            params_dict["r%d" % pop_id] = coeffs_dict[cell_type][2]
            params_dict["c%d%d" % (pop_id % 2 + 1, pop_id)] = coeffs_dict[cell_type][3]
        # Add quality of fit
        params_dict["fit%d" % pop_id] = coeffs_dict[cell_type][4]
    params_dict["fit"] = (params_dict["fit1"] + params_dict["fit2"]) / 2
    # Compute the game space position
    params_dict["Advantage_0"] = params_dict["p12"] - params_dict["p22"]
    params_dict["Advantage_1"] = params_dict["p21"] - params_dict["p11"]
    # Return the pay-off matrix entries
    return params_dict


# ---------------------------------------------------------------------------------------------------------------
def linear_range_obj_fn(X, Y, l, slope_ub=np.inf, slope_lb=-np.inf):
    """Objective function for finding the optimal linear range
    Args:
        X (array-like): independent variable
        Y (array-like): dependent variable
        l (float): regularization parameter
    """
    slope, intercept, _, _ = stats.theilslopes(Y, X)
    if slope > slope_ub:
        return np.inf
    if slope < slope_lb:
        return np.inf
    Y_pred = slope * X + intercept
    return calculate_fit(Y, Y_pred)


def opt_linear_range(X, Y, l=0, slope_ub=np.inf, slope_lb=-np.inf, min_pts=10):
    """Finds the optimal linear range for a given dataset
    Args:
        X (array-like): independent variable
        Y (array-like): dependent variable
        l (float): regularization parameter
    """
    if type(X) is list:
        X = np.array(X)
    if type(Y) is list:
        Y = np.array(Y)
    # for each subset of X, compute the objective function
    n = min(len(X), 25)
    loss_list = []
    subset_list = []
    for subset_length in range(min_pts, n):
        for start in range(n - subset_length):
            end = start + subset_length
            X_subset = X[start:end]
            Y_subset = Y[start:end]
            loss = linear_range_obj_fn(
                X_subset - X_subset[0], Y_subset, l, slope_ub=slope_ub, slope_lb=slope_lb
            )
            loss_list.append(loss)
            subset_list.append((start, end))
    # find the subset with the minimum loss
    min_loss_indx = np.argmin(loss_list)
    return subset_list[min_loss_indx]
