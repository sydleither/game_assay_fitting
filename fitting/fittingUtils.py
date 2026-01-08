# ====================================================================================
# Various functions that I found useful in this project
# ====================================================================================
import re
import numpy as np
import pandas as pd
import os
import pickle
import scipy
from scipy.stats import t
from lmfit import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import fitting.myUtils as utils
from fitting.odeModels import create_model

# ====================================================================================
def residual(params, x, data, model, model_to_observation_map, solver_kws={}, residual_kws={}):
    time_step_adjust_up = residual_kws.get('time_step_adjust_up',1.25)
    time_step_adjust_down = residual_kws.get('time_step_adjust_down',0.75)
    begin_stabilisation_threshold = residual_kws.get('begin_stabilisation_threshold',1e-1)
    verbose = residual_kws.get('verbose', False)
    refresh_ics = residual_kws.get('refresh_ics', True)
    # Set initial conditions
    if refresh_ics:
        for var in model.stateVars:
            if var in model_to_observation_map.keys():
                params[var+'0'].value = data[model_to_observation_map[var]].iloc[0]
            else:
                params[var+'0'].value = 0
    model.SetParams(**params.valuesdict())
    converged = False
    max_step = min(time_step_adjust_up*model.max_step, solver_kws.get('max_step',np.inf))
    currSolver_kws = solver_kws.copy()
    currSolver_kws['numericalStabilisationB'] = (max_step < begin_stabilisation_threshold*solver_kws.get('max_step',np.inf)) or model.numericalStabilisationB
    while not converged:
        currSolver_kws['max_step'] = max_step
        model.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(data), **currSolver_kws)
        converged = model.successB
        if verbose and (not converged or currSolver_kws['numericalStabilisationB']): print([max_step, currSolver_kws['numericalStabilisationB'], np.any(model.solObj.y<0), model.solObj.status, model.solObj.success, model.errMessage])
        max_step = time_step_adjust_down*max_step if max_step < np.inf else 100
        currSolver_kws['numericalStabilisationB'] = max_step < begin_stabilisation_threshold*solver_kws.get('max_step',np.inf)
    # Turn off numerical stabilisation once we've been able to increase the time step above the threshold again
    if not currSolver_kws['numericalStabilisationB']: model.numericalStabilisationB = False
    # Interpolate to the data time grid
    t_eval = data.Time
    tmp_list = []
    for model_feature in model_to_observation_map:
        observed_feature = model_to_observation_map[model_feature]
        f = scipy.interpolate.interp1d(model.resultsDf.Time,model.resultsDf[model_feature],fill_value="extrapolate")
        modelPrediction = f(t_eval)
        scale_dict = residual_kws.get('residual_scale',{model_feature:1.})
        res_list = (data[observed_feature]-modelPrediction) / scale_dict[model_feature]
        tmp_list.append(res_list)
    return np.concatenate(tmp_list)

# ====================================================================================
def residual_multipleConditions(params, x, data, model, model_to_observation_map, split_by,
                                solver_kws={}, residual_kws={}):
    tmpList = []
    for condition in data[split_by].unique():
        currData = data[data[split_by]==condition]
        tmpList.append(residual(params, x, currData, model, model_to_observation_map, solver_kws=solver_kws, residual_kws=residual_kws.get(condition, {})))
    return np.concatenate(tmpList)

# ====================================================================================
def residual_multipleTxConditions(params, x, data, model, model_to_observation_map, solver_kws={}, residual_kws={}):
    return residual_multipleConditions(params, x, data, model, model_to_observation_map, split_by="DrugConcentration", 
                                       solver_kws=solver_kws, residual_kws=residual_kws)

# ====================================================================================
def PerturbParams(params):
    params = params.copy()
    for p in params.keys():
        currParam = params[p]
        if currParam.vary:
            params[p].value = np.random.uniform(low=currParam.min, high=currParam.max)
    return params

# ====================================================================================
def compute_r_sq(fit,dataDf,feature="Confluence"):
    tss = np.sum(np.square(dataDf[feature]-dataDf[feature].mean()))
    rss = np.sum(np.square(fit.residual))
    return 1-rss/tss

# ====================================================================================
def prepare_data(data_df, specs_dic, restrict_range=True, average=True, average_by="Time"):
    '''
    Prepare data for fitting. This function subsets the data based on the specifications provided in specs_dic.
    :param data_df: Pandas data frame with longitudinal data to be fitted.
    :param specs_dic: Dictionary with specifications for subsetting the data. The keys are the names of the columns
    in the data frame, and the values are the values to be selected.
    :param restrict_range: Boolean, whether or not to restrict the time range to the one specified in specs_dic.
    :param average: Boolean, whether or not to average across replicates.
    :return: Pandas data frame with the data to be fitted.
    '''
    # Prepare data
    cols_available = data_df.columns
    training_data_df = data_df.copy()
    # Subset the data as specified in specs_dic
    for filter_name, filter_value in specs_dic.items():
        # specs_dic might give other details too. Only apply those col names 
        # that actually exist in the dataframe
        if filter_name in cols_available:
            filter_value_list = filter_value if isinstance(filter_value, list) else [filter_value] # Users can specify one or multiple possible values. If only one value is specified, convert to list here
            training_data_df = training_data_df[np.isin(training_data_df[filter_name], filter_value_list)].copy()
    # Limit the time if requested
    if restrict_range and specs_dic.get("TimeRange", [-np.inf])[0]>=0:
        training_data_df['Time_original'] = training_data_df['Time']
        training_range = specs_dic["TimeRange"]
        training_data_df = training_data_df[(training_data_df.Time>=training_range[0]) &
                                            (training_data_df.Time<=training_range[1])].copy()
        training_data_df['Time'] -= training_range[0]
    # Average across replicates
    if average: 
        training_data_df = training_data_df.groupby(by=average_by).mean(numeric_only=True)
        training_data_df.reset_index(inplace=True)
    return training_data_df

# ====================================================================================
def load_fit(modelName, fitId=0, file_name_model=None, fitDir="./", model=None, load_bootstraps=False, file_name_bootstraps=None, **kwargs):
    file_name_model = "fitObj_fit_%d.p"%(fitId) if file_name_model is None else file_name_model
    fitObj = pickle.load(open(os.path.join(fitDir, file_name_model), "rb"))
    myModel = create_model(modelName, **kwargs) if model is None else model
    myModel.SetParams(**fitObj.params.valuesdict())
    if load_bootstraps:
        file_name_bootstraps = "bootstraps_fit_%d.csv"%(fitId) if file_name_bootstraps is None else file_name_bootstraps
        bootstraps_df = pd.read_csv(os.path.join(fitDir, file_name_bootstraps), index_col=0)
        return fitObj, myModel, bootstraps_df
    else:
        return fitObj, myModel

# ====================================================================================
def generate_fitSummaryDf(fitDir="./fits", identifierName=None, identifierId=1, alpha=0.95):
    '''
    Function to generate a summary data frame with the parameter estimates and confidence intervals.
    :param fitDir: Directory with the fit objects.
    :param identifierName: Name of the identifier to be added to the data frame.
    :param identifierId: Value of the identifier to be added to the data frame.
    :param alpha: Confidence interval.
    :return: Pandas data frame with the parameter estimates and confidence intervals.
    '''
    fitIdList = [int(re.findall(r'\d+', x)[0]) for x in os.listdir(fitDir) if
                 x.split("_")[0] == "fitObj"]
    identifierDic = {} if identifierName is None else {identifierName: identifierId}
    tmpDicList = []
    for fitId in fitIdList:
        fitObj = pickle.load(open(os.path.join(fitDir, "fitObj_fit_%d.p"%(fitId)), "rb"))
        tmpDicList.append({**identifierDic, "FitId": fitObj.fitId, "ModelName":fitObj.modelName,
                           "AIC": fitObj.aic, "BIC": fitObj.bic, "RSquared": fitObj.rSq,
                           "Success":fitObj.success, "Message":fitObj.message, "NumericalStabilisation":fitObj.numericalStabilisation,
                           **fitObj.params.valuesdict(),
                           **dict([(x+"_se",fitObj.params[x].stderr) for x in fitObj.params.keys()]),
                           **dict([(x+"_ci",t.ppf((1+alpha)/2.0, fitObj.ndata-fitObj.nvarys)*(fitObj.params[x].stderr if fitObj.params[x].stderr is not None else np.nan)) for x in fitObj.params.keys()])})
    return pd.DataFrame(tmpDicList)

# ====================================================================================
def scale_value(val, lower, upper):
    '''
    Function to scale a value to the range [0,1].
    '''
    return (val-lower)/(upper-lower)

# ====================================================================================
def plot_parameter_spider_plot(params, axes_dict, color='grey', alpha=0.8, hull_line_width=3, plot_axis_labels=False, fig_size=4, ax=None):  
    '''
    Function to plot a spider plot of the parameter estimates.
    :param params: Pandas series with the parameter estimates or lmfit Parameters object.
    :param axes_dict: Dictionary with the axes specifications; used to choose and normalise the axes.
    :param color: Color of the hull.
    :param alpha: Transparency of the hull.
    :param hull_line_width: Width of the hull line.
    :param fig_size: Size of the figure.
    :param plot_axis_labels: Boolean, whether or not to plot the axis labels.
    :param ax: Axis object.
    :return: Axis object.
    '''
    # Initialise the spider plot
    if ax is None:
        fig = plt.figure(figsize=(fig_size,fig_size))
        ax = plt.subplot(111, polar=True)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    variable_list = list(axes_dict.keys())
    n_vars = len(variable_list)
    angles = [n / float(n_vars) * 2 * np.pi for n in range(n_vars)]
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axis per variable + add labels labels yet
    if plot_axis_labels:
        plt.xticks(angles[:-1], variable_list)
    else:
        plt.xticks(angles[:-1], np.repeat('', n_vars))

    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_yticks([0,0.5,1], ["","",""], color="black", size=16)    
    ax.set_ylim(0, 1)

    # Plot the modelscape
    # Normalise each variable
    tmp_list = []
    for var in variable_list:
        if type(params) == pd.Series:
            val = params[var]
        else:
            val = params[var].value
        val_normed = scale_value(val, *axes_dict[var])
        tmp_list.append({"Parameter":var, "Value":val_normed})
    params_df = pd.DataFrame(tmp_list)
    
    # Plot
    values = params_df['Value'].values
    values = np.concatenate([values, values[:1]])
    ax.plot(angles, values, linewidth=hull_line_width, linestyle='solid', color='grey')
    ax.fill(angles, values, color=color, alpha=alpha)
    # # Add errorbars
    # for i,var in enumerate(varList):
    #     ci = paramsDf_cis.loc[(paramsDf_cis['Parameter']==var),['Lower_Bound','Upper_Bound']].values
    #     if len(ci)==0: 
    #         ax.plot(angles[i], values[i], marker='o', markersize=17, color='grey')           
    #     else:
    #         ci = [values[i]-scale_value(ci[0][0], *myBDic[var]), scale_value(ci[0][1], *myBDic[var])-values[i]]
    #         ax.errorbar(x=angles[i], y=values[i], yerr=np.reshape(ci,(2,1)), 
    #                     color="grey", linewidth=7)
    return ax

# ====================================================================================
def perform_bootstrap(fitObj, n_bootstraps=5, shuffle_params=True, prior_experiment_df=None, model_kws={},
                      residual_fun=residual,
                      model_to_observation_map={"TumourSize":'Confluence'}, varyICs=None,
                      split_by=None,
                      show_progress=True, plot_bootstraps=False, plot_kws={'ylim':None,'palette':None}, 
                      outName=None, **kwargs):
    '''
    Function to estimate uncertainty in the parameter estimates and model predictions using a
    parametric bootstrapping method. This means, it uses the maximum likelihood estimate (best fit
    based on least squared method) to generate n_bootstrap synthetic data sets (noise is generated
    by drawing from an error distribution N(0,sqrt(ssr/df))). Subsequently it fits to this synthetic
    data to obtain a distribution of parameter estimates (one estimate/prediction
    per synthetic data set).
    '''
    # Initialise
    nvarys = fitObj.nvarys
    residual_variance = np.sum(np.square(fitObj.residual)) / fitObj.nfree
    paramsToEstimateList = [param for param in fitObj.params.keys() if fitObj.params[param].vary]
    n_conditions = 1 if split_by is None else len(fitObj.data[split_by].unique())
    plot_kws['ylim'] = 1.3*np.max([fitObj.data[x].max() for x in model_to_observation_map.values()]) if plot_kws['ylim'] is None else plot_kws['ylim']
    plot_kws['palette'] = sns.color_palette("pastel", n_colors=n_bootstraps) if plot_kws['palette'] is None else plot_kws['palette']
    if plot_bootstraps:
        fig, ax_list = plt.subplots(1, len(model_to_observation_map), figsize=(len(model_to_observation_map)*8, 6))

    # 1. Perform bootstrapping
    parameterEstimatesMat = np.zeros((n_bootstraps, nvarys+1))  # Array to hold parameter estimates for CI estimation
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # i) Generate synthetic data by sampling from the error model (assuming a normal error distribution)
        tmpDataDf = fitObj.data.copy()
        residual_kws = kwargs.get('residual_kws', {})
        residual_scale = residual_kws.get('residual_scale', {})
        if split_by is None:
            for observed_feature in model_to_observation_map.values():
                # TODO: Adjust this when using a different residual scale for each variable (or really just any scaling that's not 1) !
                tmpDataDf.loc[np.isnan(tmpDataDf[observed_feature])==False, observed_feature] -= fitObj.residual # Obtain the model prediction
                tmpDataDf.loc[np.isnan(tmpDataDf[observed_feature])==False, observed_feature] += np.random.normal(loc=0, scale=np.sqrt(residual_variance),
                                                                            size=fitObj.ndata)
        else:
            curr_condition_index_start = 0
            curr_condition_index_end = 0
            tmp_list = []
            for condition in tmpDataDf[split_by].unique():
                for variable, observed_feature in model_to_observation_map.items():
                    # Extract the residuals for the current data set
                    curr_data_selection = (np.isnan(tmpDataDf[observed_feature])==False) & (tmpDataDf[split_by]==condition)
                    curr_condition_index_end += tmpDataDf[curr_data_selection].shape[0]
                    curr_residual = fitObj.residual[curr_condition_index_start:curr_condition_index_end]
                    curr_residual_scale = residual_scale.get(variable, 1.0)
                    curr_residual_var = np.sum(np.square(curr_residual * curr_residual_scale)) / fitObj.nfree
                    tmpDataDf.loc[curr_data_selection, observed_feature] -= curr_residual # Obtain the model prediction
                    tmpDataDf.loc[curr_data_selection, observed_feature] += np.random.normal(loc=0, scale=np.sqrt(curr_residual_var),
                                                                                size=curr_residual.shape[0])
                    tmp_list.append({"Condition":condition, "Feature":observed_feature, 
                                     "Range":[curr_condition_index_start, curr_condition_index_end]})
                    curr_condition_index_start = curr_condition_index_end
            data_to_residual_index_map = pd.DataFrame(tmp_list)

        # bestFitPrediction = tmpDataDf[feature] - fitObj.residual
        # tmpDataDf[feature] = bestFitPrediction + np.random.normal(loc=0, scale=np.sqrt(residual_variance),
        #                                                                size=fitObj.ndata)
        # ii) Fit to synthetic data
        tmpModel = create_model(fitObj.modelName, **model_kws)
        currParams = fitObj.params.copy()
        # Remove variation in initial synthetic data if not fitting initial conditions;
        # otherwise this will blow up the residual variance as no fit can ever do well on the IC
        areIcsVariedList = [fitObj.params[stateVar+'0'].vary for stateVar in tmpModel.stateVars]
        if (not np.any(areIcsVariedList)) or (varyICs==False): # Allow manual overwrite via varyICs, in case of only varying some of the ICs
            for observed_feature in model_to_observation_map.values():
                if n_conditions==1:
                    tmpDataDf.loc[0, observed_feature] = fitObj.data[observed_feature].iloc[0]
                else: # If fitting to multiple experiments simultaneously, remove variation from each experiment separately
                    tmpDataDf.loc[tmpDataDf.Time==0, observed_feature] = fitObj.data.loc[fitObj.data.Time==0, observed_feature].values
        # In developing our model we proceed in a series of steps. To propagate the error along
        # as we advance to the next step, allow reading in previous bootstraps here.
        if prior_experiment_df is not None:
            for var in prior_experiment_df.columns:
                if var == "SSR": continue
                currParams[var].value = prior_experiment_df[var].iloc[bootstrapId]
        # Generate a random initial parameter guess
        if shuffle_params:
            for param in paramsToEstimateList:
                currParams[param].value = np.random.uniform(low=currParams[param].min,
                                                            high=currParams[param].max)
        # Fit
        currFitObj = minimize(residual_fun, currParams, args=(0, tmpDataDf, tmpModel,
                                                          model_to_observation_map, kwargs.get('solver_kws', {}), 
                                                          kwargs.get('residual_kws', {})),
                              **kwargs.get('optimiser_kws', {}))
        # Record parameter estimates for CI estimation
        for i, param in enumerate(paramsToEstimateList):
            parameterEstimatesMat[bootstrapId, i] = currFitObj.params[param].value
        parameterEstimatesMat[bootstrapId, -1] = np.sum(np.square(currFitObj.residual))

        # Plot the synthetic data and the individual bootstrap fits. This is useful for i) understanding what
        # the method is doing, and ii) debugging.
        if plot_bootstraps:
            # fig, ax_list = plt.subplots(1, len(model_to_observation_map), figsize=(len(model_to_observation_map)*8, 6))
            for i, (variable, observed_feature) in enumerate(model_to_observation_map.items()):
                ax = ax_list[i] if len(model_to_observation_map)>1 else ax_list
                ax.plot(tmpDataDf.Time, tmpDataDf[observed_feature], linestyle="", marker='o', linewidth=3, color=plot_kws['palette'][bootstrapId])
                if split_by is None:
                    ax.plot(tmpDataDf.Time[np.isnan(tmpDataDf[observed_feature])==False], 
                            tmpDataDf.loc[np.isnan(tmpDataDf[observed_feature])==False,observed_feature]-currFitObj.residual,
                            linewidth=3, linestyle="-", color=plot_kws['palette'][bootstrapId])
                else:
                    for condition in tmpDataDf[split_by].unique():
                        # Extract the residuals for the current data set
                        curr_data_selection = (np.isnan(tmpDataDf[observed_feature])==False) & (tmpDataDf[split_by]==condition)
                        curr_time = tmpDataDf.Time[curr_data_selection]
                        curr_index_range = data_to_residual_index_map[(data_to_residual_index_map.Condition==condition) & 
                                                                      (data_to_residual_index_map.Feature==observed_feature)].Range.values[0]
                        curr_residual_scale = residual_scale.get(variable, 1.0)
                        curr_residual = currFitObj.residual[curr_index_range[0]:curr_index_range[1]]*curr_residual_scale
                        curr_model_prediction = tmpDataDf.loc[curr_data_selection,observed_feature]-curr_residual
                        ax.plot(curr_time, curr_model_prediction,
                                linewidth=3, linestyle="-", color=plot_kws['palette'][bootstrapId]) # xxx

    # Add the maximum likelihood estimate fit to the plot
    if plot_bootstraps:
        for i, (variable, observed_feature) in enumerate(model_to_observation_map.items()):
            ax = ax_list[i] if len(model_to_observation_map)>1 else ax_list
            if split_by is None:
                bestFitPrediction = fitObj.data.loc[np.isnan(fitObj.data[observed_feature])==False,observed_feature] - fitObj.residual
                ax.plot(fitObj.data.Time[np.isnan(tmpDataDf[observed_feature])==False], bestFitPrediction, linewidth=5, linestyle="-", color='k')
            else:
                for condition in fitObj.data[split_by].unique():
                    curr_data_selection = (np.isnan(fitObj.data[observed_feature])==False) & (fitObj.data[split_by]==condition)
                    curr_time = fitObj.data.Time[curr_data_selection]
                    curr_index_range = data_to_residual_index_map[(data_to_residual_index_map.Condition==condition) & 
                                                                  (data_to_residual_index_map.Feature==observed_feature)].Range.values[0]
                    curr_residual_scale = residual_scale.get(variable, 1.0)
                    curr_residual = fitObj.residual[curr_index_range[0]:curr_index_range[1]]*curr_residual_scale
                    bestFitPrediction = fitObj.data.loc[curr_data_selection,observed_feature] - curr_residual
                    ax.plot(curr_time, bestFitPrediction, linewidth=5, linestyle="-", color='k')
            ax.set_ylim(0,plot_kws['ylim'])
            ax.set_title(observed_feature)

    # Return results
    resultsDf = pd.DataFrame(parameterEstimatesMat, columns=paramsToEstimateList+['SSR'])
    if prior_experiment_df is not None: resultsDf = pd.concat([prior_experiment_df.drop('SSR',axis=1), resultsDf], axis=1)
    if outName is not None: resultsDf.to_csv(outName)
    return resultsDf

# ====================================================================================
def compute_confidenceInterval_prediction(fitObj, bootstrapResultsDf, alpha=0.95,
                                          treatmentScheduleList=None, atToProfile=None, at_kws={},
                                          initialConditionsDic=None, model_kws={},
                                          t_eval=None, n_time_steps=100,
                                          show_progress=True, 
                                          returnTrajectories=False, estimate_fractions=False,
                                          **kwargs):
    # Initialise
    if t_eval is None:
        if treatmentScheduleList is None:
            if atToProfile is None:
                currPredictionTimeFrame = [fitObj.data.Time.min(), fitObj.data.Time.max()]
            else:
                currPredictionTimeFrame = [0, at_kws.get('t_end', 20)]
        else:
            currPredictionTimeFrame = [treatmentScheduleList[0][0], treatmentScheduleList[-1][1]]
        t_eval = np.linspace(currPredictionTimeFrame[0], currPredictionTimeFrame[1], n_time_steps) if t_eval is None else t_eval
    n_timePoints = len(t_eval)
    n_stateVars = len(create_model(fitObj.modelName, **model_kws).stateVars)
    treatmentScheduleList = treatmentScheduleList if treatmentScheduleList is not None else utils.ExtractTreatmentFromDf(
        fitObj.data)
    initialScheduleList = at_kws.get('initialScheduleList', [])
    if 'initialScheduleList' in at_kws.keys(): 
        at_kws.pop('initialScheduleList') # Remove the schedule as it is not an argument to the AT simulation function
        at_kws['t_end'] = at_kws.get('t_end', initialScheduleList[-1][1]+250)
        at_kws['t_span'] = (initialScheduleList[-1][1], at_kws['t_end'])
    n_bootstraps = bootstrapResultsDf.shape[0]

    # 1. Perform bootstrapping
    modelPredictionsMat_mean = np.zeros(
        (n_bootstraps, n_timePoints, n_stateVars+2))  # Array to hold model predictions for CI estimation
    modelPredictionsMat_indv = np.zeros(
        (n_bootstraps, n_timePoints, n_stateVars+2))  # Array to hold model predictions with residual variance for PI estimation
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # Set up the model using the parameters from a bootstrap fit
        tmpModel = create_model(fitObj.modelName, **model_kws)
        currParams = fitObj.params.copy()
        for var in bootstrapResultsDf.columns:
            if var == "SSR": continue
            currParams[var].value = bootstrapResultsDf[var].iloc[bootstrapId]
        tmpModel.SetParams(**currParams)
        # Calculate confidence intervals for model prediction
        if initialConditionsDic is not None: tmpModel.SetParams(**initialConditionsDic)
        if atToProfile is None: # Do prediction on a fixed schedule
            tmpModel.Simulate(treatmentScheduleList=treatmentScheduleList, **kwargs.get('solver_kws', {}))
        else: # Do prediction on an adaptive schedule, which may be different for each replicate, depending on the dynamics
            if len(initialScheduleList)==0: # Simulate AT from t=0
                getattr(tmpModel, 'Simulate_'+atToProfile)(**at_kws, solver_kws=kwargs.get('solver_kws', {}))
            else: # Simulate AT with a prior schedule
                tmpModel.Simulate(treatmentScheduleList=initialScheduleList, **kwargs.get('solver_kws', {}))
                at_kws['refSize'] = at_kws.get('refSize', tmpModel.resultsDf.TumourSize.iloc[-1])
                getattr(tmpModel, 'Simulate_'+atToProfile)(**at_kws, solver_kws=kwargs.get('solver_kws', {}))
        tmpModel.Trim(t_eval=t_eval)
        residual_variance_currEstimate = bootstrapResultsDf['SSR'].iloc[
                                             bootstrapId] / fitObj.nfree  # XXX Not sure this is correct for hierarchical model structure. Thus, PIs not used in paper XXX
        for stateVarId, var in enumerate(['TumourSize']+tmpModel.stateVars):
            modelPredictionsMat_mean[bootstrapId, :, stateVarId] = tmpModel.resultsDf[var].values
            modelPredictionsMat_indv[bootstrapId, :, stateVarId] = tmpModel.resultsDf[var].values + np.random.normal(loc=0,
                                                                                                               scale=np.sqrt(
                                                                                                                   residual_variance_currEstimate),
                                                                                                               size=n_timePoints)
        modelPredictionsMat_mean[bootstrapId, :, -1] = tmpModel.resultsDf['DrugConcentration'].values # Add separately as don't want to add this to the individual prediction matrix

    # 3. Estimate confidence and prediction interval for model prediction
    tmpDicList = []
    # Compute the model prediction for the model with the MLE parameter estimates
    tmpModel.SetParams(**fitObj.params)  # Calculate model prediction for best fit
    if initialConditionsDic is not None: tmpModel.SetParams(**initialConditionsDic)
    if treatmentScheduleList is None: treatmentScheduleList = utils.ExtractTreatmentFromDf(fitObj.data)
    if atToProfile is None: # Do prediction on a fixed schedule
            tmpModel.Simulate(treatmentScheduleList=treatmentScheduleList, **kwargs.get('solver_kws', {}))
    else: # Do prediction on an adaptive schedule, which may be different for each replicate, depending on the dynamics
        if len(initialScheduleList)==0: # Simulate AT from t=0
            getattr(tmpModel, 'Simulate_'+atToProfile)(**at_kws, solver_kws=kwargs.get('solver_kws', {}))
        else: # Simulate AT with a prior schedule
            tmpModel.Simulate(treatmentScheduleList=initialScheduleList, **kwargs.get('solver_kws', {}))
            at_kws['refSize'] = at_kws.get('refSize', tmpModel.resultsDf.TumourSize.iloc[-1])
            getattr(tmpModel, 'Simulate_'+atToProfile)(**at_kws, solver_kws=kwargs.get('solver_kws', {}))
    tmpModel.Trim(t_eval=t_eval)
    for i, t in enumerate(t_eval):
        for stateVarId, var in enumerate(['TumourSize']+tmpModel.stateVars):
            tmpDicList.append({"Time": t, "Variable":var, "Estimate_MLE": tmpModel.resultsDf[var].iloc[i],
                               "DrugConcentration": tmpModel.resultsDf['DrugConcentration'].iloc[i],
                               "CI_Lower_Bound": np.percentile(modelPredictionsMat_mean[:, i, stateVarId], (1 - alpha) * 100 / 2),
                               "CI_Upper_Bound": np.percentile(modelPredictionsMat_mean[:, i, stateVarId],
                                                               (alpha + (1 - alpha) / 2) * 100),
                               "PI_Lower_Bound": np.percentile(modelPredictionsMat_indv[:, i, stateVarId], (1 - alpha) * 100 / 2),
                               "PI_Upper_Bound": np.percentile(modelPredictionsMat_indv[:, i, stateVarId],
                                                               (alpha + (1 - alpha) / 2) * 100)})
            if estimate_fractions and var != "TumourSize":
                # Compute the sensitive/resistant fractions, respectively
                stateVarId_totalSize = 0
                bootstrap_fractions_list = modelPredictionsMat_mean[:, i, stateVarId]/modelPredictionsMat_mean[:, i, stateVarId_totalSize]
                tmpDicList[-1] = {**tmpDicList[-1], 
                                    "Estimate_MLE_Fraction": tmpModel.resultsDf[var].iloc[i]/tmpModel.resultsDf["TumourSize"].iloc[i],
                                    "CI_Lower_Bound_Fraction": np.percentile(bootstrap_fractions_list, (1 - alpha) * 100 / 2),
                                    "CI_Upper_Bound_Fraction": np.percentile(bootstrap_fractions_list, (alpha + (1 - alpha) / 2) * 100),
                                    }
    modelPredictionDf = pd.DataFrame(tmpDicList)
    if returnTrajectories:
        # Format the trajectories for each bootstrap into a data frame
        tmp_list = []
        for bootstrap_id in range(n_bootstraps):
            tmp_df = pd.DataFrame(modelPredictionsMat_mean[bootstrap_id], columns=["TumourSize", "S", "R", "DrugConcentration"])
            tmp_df["Time"] = tmpModel.resultsDf.Time.values
            tmp_df["BootstrapId"] = bootstrap_id
            tmp_list.append(tmp_df)
        trajectories_df = pd.concat(tmp_list)
        return modelPredictionDf, trajectories_df
    else:
        return modelPredictionDf

# ====================================================================================
def benchmark_prediction_accuracy(fitObj, bootstrapResultsDf, dataDf, initialConditionsList=None, model_kws={},
                                  show_progress=True, **kwargs):
    # Initialise
    n_bootstraps = bootstrapResultsDf.shape[0]

    # Compute the r2 value for each bootstrap
    tmpDicList = []
    for bootstrapId in tqdm(np.arange(n_bootstraps), disable=(show_progress == False)):
        # Set up the model using the parameters from a bootstrap fit
        tmpModel = create_model(fitObj.modelName, **model_kws)
        currParams = fitObj.params.copy()
        for var in bootstrapResultsDf.columns:
            if var == "SSR": continue
            currParams[var].value = bootstrapResultsDf[var].iloc[bootstrapId]
        if initialConditionsList is not None:
            for var in initialConditionsList.keys():
                currParams[var].value = initialConditionsList[var]

        # Make prediction and compare to true data
        tmpModel.residual = residual(data=dataDf, model=tmpModel, params=currParams,
                                  x=None, feature="Confluence", solver_kws=kwargs.get('solver_kws', {}))
        r2Val = compute_r_sq(fit=tmpModel, dataDf=dataDf, feature="Confluence")

        # Save results
        tmpDicList.append({"Model":fitObj.modelName, "BootstrapId":bootstrapId,
                           "rSquared":r2Val})
    return pd.DataFrame(tmpDicList)

# ====================================================================================
def compute_confidenceInterval_parameters(fitObj, bootstrapResultsDf, paramsToEstimateList=None, alpha=0.95):
    # Initialise
    if paramsToEstimateList is None:
        paramsToEstimateList = [param for param in fitObj.params.keys() if fitObj.params[param].vary]

    # Estimate confidence intervals for parameters from bootstraps
    tmpDicList = []
    for i, param in enumerate(paramsToEstimateList):
        tmpDicList.append({"Parameter": param, "Estimate_MLE": fitObj.params[param].value,
                           "Lower_Bound": np.percentile(bootstrapResultsDf[param].values, (1 - alpha) * 100 / 2),
                           "Upper_Bound": np.percentile(bootstrapResultsDf[param].values,
                                                        (alpha + (1 - alpha) / 2) * 100)})
    return pd.DataFrame(tmpDicList)

# ====================================================================================
def test_model_on_well(well_id, fit_obj, bootstrap_df, data_df, delay=0,
                       n_bootstraps=5, significance_level_ci=0.95, show_progress=False,
                       t_eval=None, n_time_steps=None, atToProfile=None,
                       solver_kws={}, optimiser_kws={}, annotations_dic={}):
    '''
    Test the model on a single well. 
    well_id: well id to test
    fit_obj: model fit object
    bootstrap_df: bootstrap results data frame
    data_df: data frame containing the experimental data. Used for extracting the treatment schedule, seeding densities, and resistance fractions
    n_bootstraps: number of bootstraps to use for uncertainty estimation
    significance_level_ci: significance level for which to generate confidence intervals for, e.g. 0.95 for 95% confidence intervals
    solver_kws: solver keyword arguments
    optimiser_kws: optimiser keyword arguments
    '''
    # Extract the treatment/seeding protocol
    curr_data_df = prepare_data(data_df, specs_dic={"WellId":well_id}, 
                                restrict_range=False, average=True, average_by="Time")
    if atToProfile is None:
        treatment_schedule_list = utils.ExtractTreatmentFromDf(curr_data_df)
        n_passages = len(treatment_schedule_list)
        at_kws = {}
    else:
        n_passages = curr_data_df.PassageId.max() + 1
        treatment_schedule_list = []
        seeding_density_list = []
        for passage_id in np.arange(0, n_passages):
            # Get the passage info
            curr_passage_df = curr_data_df[curr_data_df.PassageId==passage_id]
            t_start = curr_passage_df.Time.min() + delay
            t_end = curr_passage_df.Time.max()
            drugConcentration = curr_passage_df.DrugConcentration.unique()[0]
            # Update the previous passage to run to the beginning of this one (i.e. will start simulating from the 
            # first measurement onwards. Cells were actually passaged a little before that (a few hours, but an 
            # unknown amount.))
            if passage_id > 0:
                treatment_schedule_list[-1][1] = t_start

            # Add new passage
            treatment_schedule_list.append([t_start, t_end, drugConcentration])

            # Add the density
            curr_seeding_density = curr_passage_df[curr_passage_df.Time==t_start].Count_Total.mean()
            seeding_density_list.append(curr_seeding_density)

        # Package up to feed to the ODE solver
        at_kws={"seeding_density":seeding_density_list, #np.mean(seeding_density_list),
                "treatment_schedule":treatment_schedule_list}


    # Simulate
    if n_time_steps is None and t_eval is None: n_time_steps = 2*(int(curr_data_df.Time.max())+1) # Set the number of time steps to be twice the number of time points in the data
    initial_conditions_dic = {"S0":curr_data_df['Count_Sensitive'].iloc[0],
                            "R0":curr_data_df['Count_Resistant'].iloc[0]}
    predictions_df, trajectories_df = compute_confidenceInterval_prediction(fitObj=fit_obj, 
                                                        bootstrapResultsDf=bootstrap_df[:n_bootstraps],
                                                        treatmentScheduleList=treatment_schedule_list,
                                                        initialConditionsDic=initial_conditions_dic,
                                                        atToProfile=atToProfile,
                                                        at_kws=at_kws,
                                                        returnTrajectories=True, estimate_fractions=True,
                                                        show_progress=show_progress, 
                                                        t_eval=t_eval, n_time_steps=n_time_steps,
                                                        alpha=significance_level_ci,
                                                        solver_kws=solver_kws, optimiser_kws=optimiser_kws)
    predictions_df = predictions_df.rename(columns={"Estimate_MLE":"Estimate_MLE_Count", 
                                                    "CI_Upper_Bound":"CI_Upper_Bound_Count", 
                                                    "CI_Lower_Bound":"CI_Lower_Bound_Count"})
    # Annotate the data frame
    predictions_df["Model"] = fit_obj.modelName
    predictions_df["WellId"] = well_id
    passage_start_time_list = [treatment_schedule_list[int(x)][0] for x in range(int(n_passages))]
    predictions_df["PassageId"] = predictions_df["Time"].apply(lambda x: np.max(np.where(np.float64(x) >= passage_start_time_list)[0]))

    # Save mle and bootstrap trajectories
    trajectories_df.reset_index(drop=True, inplace=True)
    mle_df = predictions_df.pivot(index=["Time", "DrugConcentration"], columns="Variable", values="Estimate_MLE_Count")
    mle_df.reset_index(inplace=True)
    mle_df["BootstrapId"] = "MLE"
    trajectories_df = pd.concat([mle_df, trajectories_df], axis=0)
    trajectories_df["WellId"] = well_id
    trajectories_df["PassageId"] = trajectories_df["Time"].apply(lambda x: np.max(np.where(np.float64(x) >= passage_start_time_list)[0]))
    trajectories_df["Model"] = fit_obj.modelName

    # Add annotations to data frames
    for key, value in annotations_dic.items():
        predictions_df[key] = value
        trajectories_df[key] = value
    
    return predictions_df, trajectories_df