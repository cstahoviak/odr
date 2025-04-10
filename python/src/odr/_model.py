"""
Defines the Model class - an abstract base class for use with the ODR class.
"""
from abc import  ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Model(ABC):
    """
    The Model class - an abstract base class for use with the ODR class.

    This class mimics the interface of a scikit-learn "custom regressor" class
    that is designed to inherit from sklearn.base.BaseEstimator and
    sklearn.base.RegressorMixin.
    """
    def __init__(self, min_pts: int,
                 std_x: Union[float, np.ndarray],
                 std_y: float):
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
    def fit(self, x: np.ndarray, y: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        """Returns the model parameters and residual vector."""
        pass

    @abstractmethod
    def predict(self,
                x: np.ndarray,
                beta: Optional[np.ndarray] = None,
                delta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the predicted y values, i.e. y = f(x_i + delta_i; beta).

        Args:
            x: The observed "explanatory" variables, shape (n, m).
            beta: The current estimate of the model parameters, shape (p,).
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

    @abstractmethod
    def jac_beta(self,
                 x: np.ndarray,
                 beta: np.ndarray,
                 delta: np.ndarray) -> np.ndarray:
        """
        Returns the Jacobian of the forward model (the model implemented by
        Model.predict) with respect to the model parameters, beta.

        Args:
            x: The observed "explanatory" variables, shape (n, m).
            beta: The current estimate of the model parameters, shape (p,).
            delta: The current estimate of the error in the observed
                "explanatory" variables, shape (n*m,).

        Returns:
            jac_beta: The Jacobian of the forward model with respect to the
                model parameters, beta, evaluated at the current estimate of
                beta and delta, shape (n,p)
        """
        pass

    @abstractmethod
    def jac_delta(self,
                  x: np.ndarray,
                  beta: np.ndarray,
                  delta: np.ndarray) -> np.ndarray:
        """
        Returns the Jacobian of the forward model (the model implemented by
        Model.predict) with respect to delta, the error in the "explanatory"
        variables.

        Args:
            x: The observed "explanatory" variables, shape (n, m).
            beta: The current estimate of the model parameters, shape (p,).
            delta: The current estimate of the error in the observed
                "explanatory" variables, shape (n*m,).

        Returns:
            jac_beta: The Jacobian of the forward model with respect to
                delta, error in the "explanatory" variables, evaluated at the
                current estimate of beta and delta, shape (n, n*m).
        """
        pass

    def score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns the L2-norm by default."""
        return np.linalg.norm(self.predict(x) - y, ord=2)


class LinearModel2D(Model):
    """
    Defines a linear model in 2 dimensions.
    """
    def __init__(self, m: float, b: float, *args, **kwargs):
        # Call base class init at beginning of subclass init to ensure that the
        # parent class is properly initialized before any subclass-specific
        # initialization occurs.
        super().__init__(2, *args, **kwargs)

        # Store the parameters of the model
        self._m = m
        self._b = b

        self.param_vec_ = np.array([m, b])

    def fit(self, x: np.ndarray, y: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        # Fit the model parameters via OLS.
        A = np.vstack([x, np.ones_like(x)]).T
        ols_result = np.linalg.lstsq(A, y)
        return ols_result[0], ols_result[1]

    def predict(self,
                x: np.ndarray,
                beta: Optional[np.ndarray] = None,
                delta: Optional[np.ndarray] = None) -> np.ndarray:
        if delta is not None:
            # If provided, add error to explanatory variable X.
            x = x + delta

        if beta is not None:
            # If estimated model parameters are given, use them.
            return beta[0] * x + beta[1]
        else:
            # Otherwise use the "true" model parameters.
            return self._m * x + self._b

    def jac_beta(self,
                 x: np.ndarray,
                 beta: np.ndarray,
                 delta: np.ndarray) -> np.ndarray:
        return np.column_stack((x + delta, np.ones_like(x)))

    def jac_delta(self,
                 x: np.ndarray,
                 beta: np.ndarray,
                 delta: np.ndarray) -> np.ndarray:
        return np.diag(np.full(len(x), beta[0]))
