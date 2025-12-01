import os
import shlex
import subprocess
import warnings
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import Dict
from pandas.core.frame import DataFrame
import sys
sys.path.append("./")
import myUtils as utils

# ------------------------------------ Run cellprofiler ------------------------------------
def run_cellprofiler(cellprofiler_path, pipeline_file, image_path, 
                     image_groupings=None, images_to_analyse_list=None, 
                     output_dir="./", create_out_dir=True,
                     print_command=False, log_level=50, suppress_output=True, run_identifier=42):
    '''
    Run cellprofiler on a single well
    log_level: Set the verbosity for logging messages: 10 or DEBUG
               for debugging, 20 or INFO for informational, 30 or
               WARNING for warning, 40 or ERROR for error, 50 or
               CRITICAL for critical, 50 or FATAL for fatal.
               Otherwise, the argument is interpreted as the file
               name of a log configuration file (see
               http://docs.python.org/library/logging.config.html for
               file format). Taken from cellprofiler docs.
    '''
    # Cellprofiler will save its output to the same file and that file name is not changeable.
    # Only workaround is to make it output into a dedicated directory.
    output_dir
    if create_out_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Assemble cellprofiler command
    command = "{} -c -r -p {} -o {} -L {}".format(shlex.quote(cellprofiler_path), 
                                                  shlex.quote(pipeline_file), 
                                                  shlex.quote(output_dir), # Format the directory name to make it safe for the command line
                                                  log_level)
    
    # Add input images to the command
    if images_to_analyse_list is not None:
        # Generate a temporary file that holds all the images analysed by this run of cell profiler (input for the -file-list parameter)
        # This is necessary because the -file-list parameter does not accept wildcards
        # image_list = os.listdir(image_path)
        tmp_img_list_file = os.path.join(output_dir, "tmp_file_list_run%d.txt"%run_identifier)
        with open(tmp_img_list_file, "w") as f:
            for image in images_to_analyse_list:
                f.write(os.path.join(image_path, image) + "\n")
        
        # Assemble cellprofiler command
        command += " --file-list {}".format(shlex.quote(tmp_img_list_file))
    else:
        command += " -i {}".format(shlex.quote(image_path))

    # Add the groupings to the command
    if image_groupings is not None:
        image_groupings = ",".join(["%s=%s"%x for x in image_groupings.items()])
        command += " -g " + image_groupings
    
    # Run the command
    if print_command: print(command)
    kws_dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if suppress_output else {}
    subprocess.run(command, shell=True, **kws_dict)

# ------------------------------------ Plot data ------------------------------------
def plot_data(dataDf, timeColumn="Time", feature='CA153', 
              treatmentColumn="DrugConcentration", treatment_notation_mode="post",
              estimator=None, n_boot=100, err_style="bars", errorbar=('ci', 95),
              hue=None, style=None, legend=False, palette=None,
              plotDrug=True, plotDrugAsBar=True, drugBarPosition=0.85, drugBarColour="black",
              drugColorMap={"Encorafenib": "blue", "Binimetinib": "green", "Nivolumab": sns.xkcd_rgb["goldenrod"]},
              xlim=None, ylim=None, y2lim=1,
              markInitialSize=False, markPositiveCutOff=False, plotHorizontalLine=False, lineYPos=1, despine=False,
              titleStr="", decorateX=True, decorateY=True, decorateY2=True,
              markerstyle='o', markersize=12, markeredgewidth=0.5, linestyle="None", linecolor='black', linewidth=2,
              lineplot_kws={}, 
              ax=None, figsize=(10, 8), outName=None, **kwargs):
    '''
    Plot longitudinal treatment data, together with annotations of drug administration and events responsible for
    changes in treatment dosing (e.g. toxicity).
    :param dataDf: Pandas data frame with longitudinal data to be plotted.
    :param timeColumn: Name (str) of the column with the time information.
    :param feature: Name (str) of the column with the metric to be plotted on the y-axis (e.g. PSA, CA125, etc).
    :param treatmentColumn: Name (str) of the column with the information about the dose administered.
    :param plotDrug: Boolean; whether or not to plot the treatment schedule.
    :param plotDrugAsBar: Boolean, whether to plot drug as bar across the top, or as shading underneath plot.
    :param drugBarPosition: Position of the drug bar when plotted across the top.
    :param drugColorMap: Color map for colouring the shading when using different drugs.
    :param lw_events: Line width for vertical event lines.
    :param xlim: x-axis limit.
    :param ylim: y-axis limit.
    :param y2lim: y2-axis limit.
    :param markInitialSize: Boolean, whether or not to draw horizontal line at height of fist data point.
    :param plotHorizontalLine: Boolean, whether or not to draw horizontal line at position specified at lineYPos.
    :param lineYPos: y-position at which to plot horizontal line.
    :param despine: Boolean, whether or not to despine the plot.
    :param titleStr: Title to put on the figure.
    :param decorateX: Boolean, whether or not to add labels and ticks to x-axis.
    :param decorateY: Boolean, whether or not to add labels and ticks to y-axis.
    :param decorateY2: Boolean, whether or not to add labels and ticks to y2-axis.
    :param markersize: Size of markers for feature variable.
    :param linestyle: Feature variable line style.
    :param linecolor: Feature variable line color.
    :param ax: matplotlib axis to plot on. If none provided creates a new figure.
    :param figsize: Tuple, figure dimensions when creating new figure.
    :param outName: Name under which to save figure.
    :param kwargs: Other kwargs to pass to plotting functions.
    :return:
    '''
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Plot the data
    if style==None: # If no style is specified, use the "marker" keyword. Otherwise points won't show up.
        marker_dic = {"marker":markerstyle}
    else:
        marker_dic = {"markers":markerstyle}
    sns.lineplot(x=timeColumn, y=feature, hue=hue, style=style, errorbar=errorbar, err_style=err_style,
                 estimator=estimator, n_boot=n_boot, 
                 color=linecolor, legend=legend, palette=palette, linewidth=linewidth, 
                 markersize=markersize, markeredgewidth=markeredgewidth, markeredgecolor='black', 
                 **marker_dic, **lineplot_kws,
                 ax=ax, data=dataDf)

    # Plot the drug concentration
    if plotDrug:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        if hue is not None or style is not None:
            drug_data_df = dataDf.groupby(timeColumn).mean(numeric_only=True).reset_index()
        else:
            drug_data_df = dataDf
        if plotDrugAsBar:
            drugConcentrationVec = utils.TreatmentListToTS(
                treatmentList=utils.ExtractTreatmentFromDf(drug_data_df, timeColumn=timeColumn,
                                                           treatmentColumn=treatmentColumn,
                                                           mode=treatment_notation_mode),
                tVec=drug_data_df[timeColumn])
            drugConcentrationVec[drugConcentrationVec < 0] = 0
            drugConcentrationVec = np.array([x / (np.max(drugConcentrationVec) + 1e-12) for x in drugConcentrationVec])
            drugConcentrationVec = drugConcentrationVec / (1 - drugBarPosition) + drugBarPosition
            ax2.fill_between(drug_data_df[timeColumn], drugBarPosition, drugConcentrationVec,
                             step="post", color=drugBarColour, alpha=1., label="Drug Concentration")
            ax2.axis("off")
        else:
            currDrugBarPosition = drugBarPosition
            drugBarHeight = (1-drugBarPosition)/len(drugColorMap.keys())
            for drug in drugColorMap.keys():
                drugConcentrationVec = utils.TreatmentListToTS(
                    treatmentList=utils.ExtractTreatmentFromDf(drug_data_df, timeColumn=timeColumn,
                                                               treatmentColumn="%s Dose (mg)"%drug,
                                                               mode=treatment_notation_mode),
                    tVec=drug_data_df[timeColumn])
                drugConcentrationVec[drugConcentrationVec < 0] = 0
                # Normalise drug concentration to 0-1 (1=max dose(=initial dose))
                drugConcentrationVec = np.array([x / (np.max(drugConcentrationVec) + 1e-12) for x in drugConcentrationVec])
                # Rescale to make it fit within the bar at the top of the plot
                drugConcentrationVec = drugConcentrationVec * drugBarHeight + currDrugBarPosition
                ax2.fill_between(drug_data_df[timeColumn], currDrugBarPosition, drugConcentrationVec, step="post",
                                 color=drugColorMap[drug], alpha=0.5, label="Drug Concentration")
                ax2.hlines(xmin=drug_data_df[timeColumn].min(), xmax=drug_data_df[timeColumn].max(), 
                          y=currDrugBarPosition, linewidth=3, color="black")
                currDrugBarPosition += drugBarHeight
            # Line at the top of the drug bars
            ax2.hlines(xmin=drug_data_df[timeColumn].min(), xmax=drug_data_df[timeColumn].max(), 
                          y=currDrugBarPosition, linewidth=3, color="black")
        # Format y2 axis
        if y2lim is not None: ax2.set_ylim([0, y2lim])
        ax2.tick_params(labelsize=28)
        if not decorateY2:
            ax2.set_yticklabels("")

    # Format the plot
    if xlim is not None: ax.set_xlim(0, xlim)
    if ylim is not None: ax.set_ylim(0, ylim)
    if despine: sns.despine(ax=ax, trim=True, offset=50)

    # Draw horizontal lines (e.g. initial size)
    if plotHorizontalLine or markInitialSize or markPositiveCutOff:
        xlim = ax.get_xlim()[1]
        if markInitialSize: lineYPos = dataDf.loc[dataDf[timeColumn] == 0, feature]
        if markPositiveCutOff: lineYPos = 0.5 # cut-off value for positive is 0.5 copies/uL
        ax.hlines(xmin=0, xmax=xlim, y=lineYPos, linestyles=':', linewidth=4)

    # Decorate the plot
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(titleStr)
    ax.tick_params(labelsize=28)
    if not decorateX:
        ax.set_xticklabels("")
    if not decorateY:
        ax.set_yticklabels("")
    plt.tight_layout()
    if outName is not None: plt.savefig(outName)