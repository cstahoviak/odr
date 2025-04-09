"""
Defines the Model class - an abstract base class for use with the ODR class.
"""
from abc import  ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Model(ABC):
    """
    The Model class - an abstract base class for use with the ODR class.

    This class mimics the interface of a scikit-learn "custom regressor" class
    that is designed to inherit from sklearn.base.BaseEstimator and
    sklearn.base.RegressorMixin.
    """
    def __init__(self, min_pts: int, std_x: np.ndarray, std_y: float):
        # Store the minimum number of observations required to fit the model.
        self._min_pts = min_pts

        # Store the known standard deviation of the explanatory variables
        self._std = std_x

        # Create the "error variance ratio" vector
        self._d = std_y / std_x

    @property
    def min_pts(self) -> int:
        """
        The minimum number of observations to fit the model parameters, i.e. the
        number of observations that define the "uniquely-determined" problem.
        """
        return self._min_pts

    @property
    def std(self) -> np.ndarray:
        """
        Returns the std deviation of each "explanatory" variable, i.e.
        (std_x1, std_x2, ..., std_xN)
        """
        return self._std

    @property
    def d(self) -> np.ndarray:
        """
        Returns the "error variance ratio" vector. The error variance ratio is
        the ratio of the standard deviation of the "dependent variable (y) to
        the standard deviation of the "dependent" variable, i.e.
        [std_y / std_x1, std_y / std_x2, ..., std_y / std_N]
        """
        return self._d

    @abstractmethod
    def fit(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the model parameters and residual vector."""
        pass

    @abstractmethod
    def predict(self,
                X: np.ndarray,
                beta: Optional[np.ndarray] = None,
                delta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the predicted y values, i.e. y = f(x_i + delta_i; beta).

        Args:
            X: The observed "explanatory" variables, shape (n, m).
            beta: The current estimate of the model parameters, shape (p,)
            delta: The current estimate of the error in each explanatory
                variable for each datapoint. This is an "interleaved" vector of
                shape (n*m,), e.g. for a three dimensional explanatory variable
                vector, we'd have delta =
                    [delta_11, delta_12, delta_13,
                     delta_21, delta_22, delta_23,
                     delta_31, delta_32, delta_33,
                     ...,      ...,     ...,
                    delta_N1, delta_N2, delta_N3]
        """
        pass

    def score(self, X, y) -> np.ndarray:
        """Returns the L2-norm by default."""
        return np.linalg.norm(self.predict(X) - y, ord=2)


class LinearModel2D(Model):
    """
    Defines a linear model in 2 dimensions.
    """
    def __init__(self, m: float, b: float, std_x: np.ndarray, std_y: float):
        # Call base class init at beginning of subclass init to ensure that the
        # parent class is properly initialized before any subclass-specific
        # initialization occurs.
        super().__init__(min_pts=2, std_x=std_x, std_y=std_y)

        # Store the parameters of the model
        self._m = m
        self._b = b

        self.param_vec_ = np.array([m, b])

    def fit(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        # Fit the model parameters via OLS.
        A = np.vstack([X, np.ones_like(X)]).T
        ols_result = np.linalg.lstsq(A, y)
        return ols_result[0], ols_result[1]

    def predict(self,
                X: np.ndarray,
                beta: Optional[np.ndarray] = None,
                delta: Optional[np.ndarray] = None) -> np.ndarray:
        if delta:
            # If provided, add error to explanatory variable X.
            X = X + delta

        if beta:
            # If estimated model parameters are given, use them.
            return beta[0] * X + beta[1]
        else:
            # Otherwise use the "true" model parameters.
            return self._m * X + self._b