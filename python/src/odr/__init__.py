"""
The public interface of the ODR (Orthogonal Distance Regression) package.
"""

from ._model import Model, LinearModel2D
from ._odr import OrthogonalDistanceRegression

__all__ = [
    'Model',
    'LinearModel2D',
    'OrthogonalDistanceRegression'
]