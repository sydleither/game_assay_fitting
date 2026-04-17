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
    count_threshold=0,
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
    # Clean count data
    for cell_type in tags:
        counts_raw_df["Mean_Count_%s_objects" % cell_type] = counts_raw_df.groupby("WellId")[
            "Count_%s_objects" % cell_type
        ].transform("mean")
        counts_raw_df.loc[
            counts_raw_df["Mean_Count_%s_objects" % cell_type] < count_threshold,
            "Count_%s_objects" % cell_type,
        ] = 0
        counts_raw_df.loc[
            counts_raw_df["Count_%s_objects" % cell_type] < 0, "Count_%s_objects" % cell_type
        ] = 0
    # Calculate frequencies
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
def estimate_growth_rate(data_df, growth_rate_window, well_id=None, cell_type=None, logspace=False):
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
    Y_pred = slope * x + intercept
    if logspace:
        error = np.sum(np.square(y - Y_pred))
    else:
        error = np.sum(np.square(np.exp(y) - np.exp(Y_pred)))
    bic = calculate_bic(curr_df, error, 2)
    return slope, intercept, low_slope, high_slope, error, bic


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
        error = np.sum(np.square(tmp_df[growth_rate_col] - Y_pred))
        coeffs_dict[cell_type] = [
            best_fit_func(0),
            best_fit_func(1),
            theil_result.intercept,
            theil_result.slope,
            error,
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
        params_dict["error%d" % pop_id] = coeffs_dict[cell_type][4]
    params_dict["error"] = (params_dict["error1"] + params_dict["error2"]) / 2
    # Compute the game space position
    params_dict["Advantage_0"] = params_dict["p12"] - params_dict["p22"]
    params_dict["Advantage_1"] = params_dict["p21"] - params_dict["p11"]
    # Return the pay-off matrix entries
    return params_dict


# ---------------------------------------------------------------------------------------------------------------
def optimize_growth_rate_window_per_exp(df):
    """Input: counts dataframe for a given experiment."""
    fits = []
    for plate in df["PlateId"].unique():
        df_plate = df[df["PlateId"] == plate]
        for well in df_plate["WellId"].unique():
            df_well = df_plate[df_plate["WellId"] == well]
            for cell_type in df_well["CellType"].unique():
                df_ct = df_well[df_well["CellType"] == cell_type]
                _, fit = optimize_growth_rate_window_per_cell(
                    df_ct, return_fit_df=True, filter_by_dominant_sign=False
                )
                fits.append(fit)
    fits = pd.concat(fits, ignore_index=True)
    fits = (
        fits[["Window_Start", "Window_End", "BIC"]]
        .groupby(["Window_Start", "Window_End"])
        .mean()
        .reset_index()
    )
    best_window = fits.loc[fits["BIC"].idxmin()]
    df["GrowthRate_window_start"] = best_window["Window_Start"]
    df["GrowthRate_window_end"] = best_window["Window_End"]
    return df


def optimize_growth_rate_window_per_well(df):
    """Input: counts dataframe for a given well."""
    df["GrowthRate_window_start"] = np.nan
    df["GrowthRate_window_end"] = np.nan
    df["BIC"] = np.nan
    for plate in df["PlateId"].unique():
        df_plate = df[df["PlateId"] == plate]
        for well in df_plate["WellId"].unique():
            fits = []
            df_well = df_plate[df_plate["WellId"] == well]
            for cell_type in df_well["CellType"].unique():
                df_ct = df_well[df_well["CellType"] == cell_type]
                _, fit = optimize_growth_rate_window_per_cell(
                    df_ct.copy(), return_fit_df=True, filter_by_dominant_sign=False
                )
                fits.append(fit)
            fit_df = pd.concat(fits, ignore_index=True)
            if len(fit_df) == 0:
                continue
            fit_df = (
                fit_df[["Window_Start", "Window_End", "BIC"]]
                .groupby(["Window_Start", "Window_End"])
                .mean()
                .reset_index()
            )
            best_window = fit_df.loc[fit_df["BIC"].idxmin()]
            df.loc[(df["PlateId"] == plate) & (df["WellId"] == well), "GrowthRate_window_start"] = (
                best_window["Window_Start"]
            )
            df.loc[(df["PlateId"] == plate) & (df["WellId"] == well), "GrowthRate_window_end"] = (
                best_window["Window_End"]
            )
    return df


def calculate_bic(well_df, sum_squared_residuals, k=2):
    if sum_squared_residuals == 0:
        return 0
    n_data_points = well_df.shape[0]
    bic = n_data_points * np.log(sum_squared_residuals / n_data_points) + k * np.log(n_data_points)
    return bic


def optimize_growth_rate_window_per_cell(
    well_df,
    min_window_size=5,
    max_window_size=20,
    metric_for_selection="BIC",
    filter_by_dominant_sign=True,
    top_quantile_of_slopes_to_include=0.25,
    plot_fits=False,
    return_fit_df=False,
    verbose=False,
):
    """
    This function will test different windows for growth rate estimation and select the optimal one
    based on a specified metric (e.g. BIC). The optimization is done by exhaustively testing all
    possible windows within a specified range of window sizes and start times. The function returns the
    best window and a summary dataframe with the fit parameters and metrics for all tested windows.
    The metric for selection can be specified with the metric_for_selection argument (e.g. "BIC" or "SumSquaredResiduals").
    In addition, the function allows optional filtering of the tested windows based on the dominant sign of
    the slope and the absolute value of the slope. The purpose is to deal with the fact that growth curves
    can have multiple phases (e.g. growth followed by plateau) and we want to make sure that the selected window
    is in the growth phase and not in the plateau phase. Similarly, when we drug cells, then there may
    be a delay before the drug acts. If that's the case, then we will probably want the growth rate estimation
    to focus on the later windows where the drug is acting. It's difficult to write a good general method
    to fix this, but one way I found works reasonably well is to look for what is the dominant sign of the
    slope across all windows (do we mostly see growth, or mostly decline?) and then additionally filter for
    those windows with the highest absolute growth rate (i.e. the most extreme slopes), so to get rid of
    windows with stagnation (either because we're on a plateau or because the window length is short).
    Input:
    - well_df: dataframe with columns "Time" and "Count" for a single well
    - min_window_size: minimum window size to test (in #time points, not hours)
    - max_window_size: maximum window size to test (in #time points, not hours)
    - min_cell_number: minimum cell number threshold for growth rate estimation (passed to the estimate_growth_rate function)
    - metric_for_selection: which metric to use for selecting the best window (options: "BIC" or "SumSquaredResiduals")
    - filter_by_dominant_sign: whether to filter the windows based on the dominant sign of the slope
    - top_quantile_of_slopes_to_include: what percentage of windows to keep based on the absolute value of the slope (e.g. if set to 0.25, we will keep the top 25% of windows with the highest absolute slope values)
    - plot_fits: whether to plot the fits for each window choice for visual checking (this can be very useful to check if the optimization is working well, but it will generate a lot of plots, so use with caution)
    Output:
    - best_window: a dictionary with the parameters of the best window (e.g. "WindowLength", "StartIdx", "Window", "Slope", "Intercept", "BIC", etc.)
    - fit_summary_df: a dataframe with the fit parameters and metrics for all tested windows, which can be used for further analysis or troubleshooting
    """
    if well_df["Count"].var() == 0:
        well_df["GrowthRate_window_start"] = well_df["Time"].min()
        well_df["GrowthRate_window_end"] = well_df["Time"].max()
        well_df["BIC"] = np.nan
        well_df["SumSquaredResiduals"] = np.nan
        if return_fit_df:
            return well_df, pd.DataFrame()
        return well_df

    # Sweep all possible start times and window sizes
    tmp_list = []
    for window_size in range(min_window_size, max_window_size + 1):
        # Generate a list of all start points for windows to test. This will depend on the
        # number of unique time points in the data and the window size we want to test (the
        # longer the window, the fewer options for start points we have, because the window has
        # to fit within the time range of the data).
        start_points_to_test_list = np.arange(len(well_df["Time"].unique()) - window_size)

        # Optional: plot the fits for each window choice to visually check if the optimization is working well
        if plot_fits:
            n_rows = start_points_to_test_list.shape[0] // 5 + 1
            n_cols = 5
            fig, ax_list = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(16, 8))

        # Optimisation: Loop through all start points for this window size
        for i, start_idx in enumerate(start_points_to_test_list):
            curr_window = well_df["Time"].unique()[[start_idx, start_idx + window_size]]

            # Estimate growth rate for this window
            curr_slope, curr_intercept, _, _, error, bic = estimate_growth_rate(
                well_df, growth_rate_window=curr_window, logspace=True
            )
            tmp_list.append(
                {
                    "WindowLength": window_size,
                    "StartIdx": start_idx,
                    "Window": curr_window,
                    "Window_Start": curr_window[0],
                    "Window_End": curr_window[1],
                    "Slope": curr_slope,
                    "Intercept": curr_intercept,
                    "StartIdx_at_edge": i == len(start_points_to_test_list) - 1,
                    "SumSquaredResiduals": error,
                    "BIC": bic,
                }
            )

            # Plot for visual checking
            if plot_fits:
                # fig, ax = plt.subplots(figsize=(6, 4))
                ax = ax_list.flatten()[i]
                sns.lineplot(
                    x="Time",
                    y="Count",
                    style="CellType",
                    color="red",
                    markers="s",
                    markeredgewidth=0.5,
                    markeredgecolor="black",
                    legend=False,
                    data=well_df,
                    ax=ax,
                )
                ax.set_yscale("log")

                # Add the growth rate fit line
                # 2. When estimating the growth rate, we are fitting an exponential curve to the data
                # We will overlay this curve here to see if it fits well and if the window is appropriately chosen
                x = np.arange(*curr_window, 0.1)
                y = curr_slope * (x - curr_window[0]) + curr_intercept
                ax.plot(x, np.exp(y), color="#09D3F2", linestyle="-", linewidth=2)

                # 3. Annotate the window which is used for the growth rate calculation
                ax.axvline(curr_window[0], color="black", linestyle="--")
                ax.axvline(curr_window[1], color="black", linestyle="--")
                ax.fill_betweenx(
                    [1e2, 3e4], curr_window[0], curr_window[1], color="#F2096E", alpha=0.1
                )
    fit_summary_df = pd.DataFrame(tmp_list)

    # Optional: Growth curves can have multiple phases. For example, a sigmoid will have a growth phase
    # followed by a plateau. The method above will give us an estimate of the growth rate for each window
    # we test and it is possible that the plateau phase has a better fit (e.g. lower BIC) than the growth
    # phase. Similarly, when we drug cells, then there may be a delay before the drug acts. If that's the
    # case, then we will probably want the growth rate estimation to focus on the later windows
    # where the drug is acting. It's difficult to write a good general method to fix this, but one way
    # I found works reasonably well is to look for what is the dominant sign of the slope across
    # all windows (do we mostly see growth, or mostly decline?) and then additionally filter for those
    # windows with the highest absolute growth rate (i.e. the most extreme slopes), so to get rid of
    # windows with stagnation (either because we're on a plateau or because the window length is short).
    # This is done here and can be controlled by:
    # - filter_by_dominant_sign: whether to filter the windows based on the dominant sign of the slope
    # - top_quantile_of_slopes_to_include: what percentage of windows to keep based on the absolute value of the slope (e.g. if set to 0.25, we will keep the top 25% of windows with the highest absolute slope values)
    if filter_by_dominant_sign:
        dominant_sign = np.sign(fit_summary_df["Slope"].median())
        if dominant_sign != 0:
            # Include top x% of windows with the dominant sign
            fit_summary_df = fit_summary_df[fit_summary_df["Slope"] * dominant_sign > 0].copy()
            threshold = np.quantile(
                np.abs(fit_summary_df["Slope"]), 1 - top_quantile_of_slopes_to_include
            )
            fit_summary_df = fit_summary_df[np.abs(fit_summary_df["Slope"]) >= threshold].copy()

    # Check that we have some valid fits (i.e. some windows where the growth rate could be estimated)
    if fit_summary_df.shape[0] == 0:
        print(well_df)
        raise ValueError("No valid windows found for growth rate estimation")

    # Identify the best window
    best_window = fit_summary_df.loc[fit_summary_df[metric_for_selection].idxmin()]

    # Quality control: make sure the best window is not at the very edge of the tested
    # options (which would suggest that the optimal window may be outside of the tested range)
    best_window_size, start_idx_at_edge = (
        best_window["WindowLength"],
        best_window["StartIdx_at_edge"],
    )
    if verbose:
        if best_window_size == min_window_size or best_window_size == max_window_size:
            print(
                "Warning: Best window is at the edge of the tested window sizes. Consider increasing the range of tested window sizes."
            )
        if start_idx_at_edge:
            print(
                "Warning: Best window start index is at the edge of the tested range. You might not have run the experiment long enough to capture the optimal window."
            )
    well_df["GrowthRate_window_start"] = best_window["Window"][0]
    well_df["GrowthRate_window_end"] = best_window["Window"][1]
    if return_fit_df:
        return well_df, fit_summary_df
    return well_df
