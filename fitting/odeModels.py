# ====================================================================================
# ODE models
# ====================================================================================
from lmfit import Parameters
import numpy as np

from fitting.odeModelClass import ODEModel


# ======================== House Keeping Funs ==========================================
def create_model(modelName, **kwargs):
    funList = {"replicator": Replicator, "lotka-volterra": LotkaVolterra}
    return funList[modelName](**kwargs)


# ======================= Models =======================================
class Replicator(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "replicator"
        self.paramDic = {
            **self.paramDic,
            "p_SR": 1.0,
            "p_SS": 0.25,
            "p_RS": 0.5,
            "p_RR": 0.25,
            "S0": 750,
            "R0": 750,
        }
        self.stateVars = ["S", "R"]

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, _ = uVec
        dudtVec = np.zeros_like(uVec)
        f_R = R / (S + R)
        dudtVec[0] = (self.paramDic["p_SS"] * (1 - f_R) + self.paramDic["p_SR"] * f_R) * S
        dudtVec[1] = (self.paramDic["p_RR"] * f_R + self.paramDic["p_RS"] * (1 - f_R)) * R
        dudtVec[2] = 0
        return dudtVec

    def RunCellCountToTumourSizeModel(self, popModelSolDf):
        return popModelSolDf["S"].values + popModelSolDf["R"].values

    def get_params(self):
        params = Parameters()
        params.add("p_SS", value=1e-2, min=0, max=0.1, vary=True)
        params.add("p_RR", value=1e-2, min=0, max=0.1, vary=True)
        params.add("p_SR", value=1e-2, min=0, max=0.1, vary=True)
        params.add("p_RS", value=1e-2, min=0, max=0.1, vary=True)
        params.add("S0", value=50, min=0, max=1e4, vary=False)
        params.add("R0", value=50, min=0, max=1e4, vary=False)
        return params


class LotkaVolterra(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "replicator"
        self.paramDic = {
            **self.paramDic,
            "r_S": 0.001,
            "r_R": 0.001,
            "a_SR": 0.001,
            "a_SS": -0.001,
            "a_RS": 0.001,
            "a_RR": -0.001,
            "S0": 500,
            "R0": 250,
        }
        self.stateVars = ["S", "R"]

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, _ = uVec
        dudtVec = np.zeros_like(uVec)
        cc = 1 - ((S + R) / 1e4)
        dudtVec[0] = cc * (
            S * (self.paramDic["r_S"] + self.paramDic["a_SS"] * S + self.paramDic["a_SR"] * R)
        )
        dudtVec[1] = cc * (
            R * (self.paramDic["r_R"] + self.paramDic["a_RS"] * S + self.paramDic["a_RR"] * R)
        )
        dudtVec[2] = 0
        return dudtVec

    def RunCellCountToTumourSizeModel(self, popModelSolDf):
        return popModelSolDf["S"].values + popModelSolDf["R"].values

    def get_params(self):
        params = Parameters()
        params.add("r_S", value=1e-3, min=0.001, max=0.01, vary=True)
        params.add("r_R", value=1e-3, min=0.001, max=0.01, vary=True)
        params.add("a_SS", value=-1e-3, min=-0.01, max=-0.001, vary=True)
        params.add("a_RR", value=-1e-3, min=-0.01, max=-0.001, vary=True)
        params.add("a_SR", value=1e-3, min=-0.01, max=0.01, vary=True)
        params.add("a_RS", value=1e-3, min=-0.01, max=0.01, vary=True)
        params.add("S0", value=50, min=0, max=1e4, vary=False)
        params.add("R0", value=50, min=0, max=1e4, vary=False)
        return params
