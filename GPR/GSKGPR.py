import typing
import numpy as np
import numpy.typing as npt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from .GSKernel import GSkernel

class GSKGPR():
    """
    Generic String Kernel for Gaussian process regression. Based on Yutao Ma's code:
      https://github.com/tommayutao/ELP-Screening/
    """
    def __init__(self, X_train: npt.ArrayLike, Y_train: npt.ArrayLike, AA_matrix: str | Path | None = None) -> None:
        self.X_train = X_train
        self.Y_train = Y_train
        # Amino Acid interaction matrix
        self.AA_mat = AA_matrix
        # initiate best L parameter
        self.fit_l = None

    def fit(self, alpha_train: float, L_grid: typing.Iterable, bounds: npt.ArrayLike) -> GaussianProcessRegressor | None:
        # initializations
        max_likelihood = -np.inf
        cur_best_model = None
        best_l = None
        # grid search on L
        for l in L_grid:
            # GS kernel
            kernel = GSkernel(self.AA_mat, L = l,length_scale_bounds = bounds)
            # GPR to optimize sigma values
            gp = GaussianProcessRegressor(kernel = kernel, normalize_y = True, alpha = alpha_train/np.var(self.Y_train)) # type: ignore
            gp.fit(self.X_train, self.Y_train)
            likelihood = gp.log_marginal_likelihood_value_
            # update the best model
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                cur_best_model = gp
                best_l = l
        # set the best L value from grid search
        self.fit_l = best_l
        return cur_best_model