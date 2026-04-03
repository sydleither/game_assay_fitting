# ====================================================================================
# ODE models
# ====================================================================================
from lmfit import Parameters
import numpy as np

from fitting.odeModelClass import ODEModel


# ======================== House Keeping Funs ==========================================
def get_models():
    return {"replicator": Replicator, "lotka-volterra": LotkaVolterra}


def create_model(modelName, **kwargs):
    return get_models()[modelName](**kwargs)


# ======================= Models =======================================
class Replicator(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "replicator"
        self.paramDic = {
            **self.paramDic,
            "p_SS": 0.05,
            "p_SR": 0.05,
            "p_RS": 0.05,
            "p_RR": 0.05,
            "S0": 100,
            "R0": 100,
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
        params.add("p_SS", value=0.05, min=-1, max=1, vary=True)
        params.add("p_RR", value=0.05, min=-1, max=1, vary=True)
        params.add("p_SR", value=0.05, min=-1, max=1, vary=True)
        params.add("p_RS", value=0.05, min=-1, max=1, vary=True)
        params.add("S0", value=100, min=1, max=1e4, vary=False)
        params.add("R0", value=100, min=1, max=1e4, vary=False)
        return params


class LotkaVolterra(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "lotka-volterra"
        self.paramDic = {
            **self.paramDic,
            "r_S": 0.1,
            "r_R": 0.1,
            "a_SR": 1e-4,
            "a_SS": -1e-4,
            "a_RS": 1e-4,
            "a_RR": -1e-4,
            "S0": 100,
            "R0": 100,
        }
        self.stateVars = ["S", "R"]

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, _ = uVec
        if S > 1e6 or R > 1e6:
            return np.zeros_like(uVec)
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = S * (
            self.paramDic["r_S"] + self.paramDic["a_SS"] * S + self.paramDic["a_SR"] * R
        )
        dudtVec[1] = R * (
            self.paramDic["r_R"] + self.paramDic["a_RS"] * S + self.paramDic["a_RR"] * R
        )
        dudtVec[2] = 0
        return dudtVec

    def RunCellCountToTumourSizeModel(self, popModelSolDf):
        return popModelSolDf["S"].values + popModelSolDf["R"].values

    def get_params(self):
        params = Parameters()
        params.add("r_S", value=0.1, min=0, max=0.5, vary=True)
        params.add("r_R", value=0.1, min=0, max=0.5, vary=True)
        params.add("a_SS", value=-1e-4, min=-1e-2, max=0, vary=True)
        params.add("a_RR", value=-1e-4, min=-1e-2, max=0, vary=True)
        params.add("a_SR", value=1e-4, min=-1e-2, max=1e-2, vary=True)
        params.add("a_RS", value=1e-4, min=-1e-2, max=1e-2, vary=True)
        params.add("S0", value=100, min=0, max=1e4, vary=False)
        params.add("R0", value=100, min=0, max=1e4, vary=False)
        return params
