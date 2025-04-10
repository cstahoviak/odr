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
    std_x, std_y = 1, 1.5
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
    err_m_ols = np.abs(m_ols - m)
    err_b_ols = np.abs(b_ols - b)
    err_total_ols = err_m_ols + err_b_ols

    # Define a relatively poor initial guess
    beta0 = np.ones(2)

    # Fit the model parameters via ODR
    odr = OrthogonalDistanceRegression(model, converge_thres=1e-4)
    # TODO: Update the ODR class to meet the new Model interface
    odr.odr(x=x_noisy,
            y=y_noisy,
            beta0=beta0,
            weights=np.ones_like(y_noisy))
    m_odr, b_odr = odr.param_vec_
    err_m_odr = np.abs(m_odr - m)
    err_b_odr = np.abs(b_odr - b)
    err_total_odr = err_m_odr + err_b_odr

    # Validate that ODR is a better estimate of the true model parameters
    np.testing.assert_array_less(err_m_odr, err_m_ols)
    np.testing.assert_array_less(err_b_odr, err_b_ols)
    np.testing.assert_array_less(err_total_odr, err_total_ols)

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

