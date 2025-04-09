"""
Unit tests for the OrthogonalDistanceRegression class.
"""
import numpy as np
import pytest

from odr import LinearModel2D, OrthogonalDistanceRegression

@pytest.fixture
def rng():
    # Create the random number generator (RNG).
    return np.random.default_rng()

def test_odr_linear(rng):
    """
    Validate the ODR method for regression of a 2D linear model.
    """
    # Define the linear model and known noise parameters
    m, b = rng.integers(low=1, high=10, size=2, endpoint=True)
    std_x, std_y = 0.5, 1
    model = LinearModel2D(m, b, std_x, std_y)

    # Generate the "truth" dataset
    n = 1000
    x = rng.uniform(low=0, high=100, size=n)
    y = model.predict(x)

    # Create the "measurement" dataset by adding Gaussian noise to both the
    # "explanatory" (x) and "dependent" (y) variables.
    x_noisy = x + rng.normal(loc=0, scale=std_x, size=n)
    y_noisy = y + rng.normal(loc=0, scale=std_y, size=n)

    # Fit the model parameters via OLS and compute the error
    ols_param_vec, ols_residuals = model.fit(x_noisy, y_noisy)
    m_ols, b_ols = ols_param_vec
    sqerr_m_ols = (m_ols - m) ** 2
    sqerr_b_ols = (b_ols - b) ** 2
    sqerr_total_ols = sqerr_m_ols + sqerr_b_ols

    # Define the initial guess for ODR by adding noise to the truth values
    beta0 = np.array([m, b]) + rng.normal(loc=0, scale=2, size=2)

    # Fit the model parameters via ODR
    odr = OrthogonalDistanceRegression(model)
    # TODO: Update the ODR class to meet the new Model interface
    odr.odr(x_noisy, beta0, np.ones_like(y_noisy), get_covar=False)
    m_odr, b_odr = odr.param_vec_

    # Validate that ODR is a better estimate of the true model parameters

    pass

def test_odr_polynomial():
    """
    TODO: Validate the ODR method for regression of a 2D polynomial model.
    """
    pass

def test_odr_3d():
    """
    TODO: Validate the ODR method for regression of a 3 dimensional surface.
      This example is most closely related to the ODR velocity estimation
      problem.
    """
    pass

