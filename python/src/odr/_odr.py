"""
Implements the Orthogonal Distance Regression (ODR) algorithm described by
"A Stable and Efficient Algorithm for Nonlinear Orthogonal Distance Regression",
Boggs, Byrd and Schnabel, 1987.
"""
import logging
import warnings

import numpy as np
from scipy.linalg import block_diag

from odr import Model

class OrthogonalDistanceRegression:
    """
    An implementation of Paul T. Boggs' and Richard Byrd's Orthogonal Distance
    Regression (ODR) algorithm as described by "A Stable and Efficient
    Algorithm for Nonlinear Orthogonal Distance Regression.", 1987.
    """
    def __init__(self,
                 model: Model,
                 converge_thres: float = 1e-6,
                 max_iter: int = 50,
                 s_scale_factor: float = 1.0,
                 dtype: np.dtype = np.float32):
        """
        The Orthogonal Distance Regression constructor.
        Args:
            model: An odr.Model instance.
            converge_thres: The ODR algorithm will terminate once the step size
                in s (the step size in parameter vector beta) has dropped below
                'converge_thres'.
            max_iter: The maximum number of iteration of the ODR algorithm. If
                'max_iter' is reach before the convergence threshold on the step
                size s is met, the method will terminate.
            s_scale_factor: The scale factor applied to s, the per-iteration
                step size in the parameters, beta.
            dtype:
        """
        # Store the model
        self._model = model
        # ODR convergence threshold on step size s
        self._converge_thres = converge_thres
        # Max number of ODR iterations
        self._max_iterations = max_iter
        self._dtype = dtype

        # Parameter vector - to be estimated by ODR
        self.param_vec_ = None
        # Covariance of parameter estimate, shape (p,p)
        self.covariance_ = None
        # Number of iterations until convergence
        self.iter_ = 0

        # Step size scale factor
        # self.s_ = 10 * np.ones(self._model.min_pts, dtype=self._dtype)
        self.s_ = \
            s_scale_factor * np.ones(self._model.min_pts, dtype=self._dtype)

        # Create a random number generator (RNG)
        self._rng = np.random.default_rng()

        # Add logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def odr(self,
            x: np.ndarray,
            y: np.ndarray,
            beta0: np.ndarray,
            weights: np.ndarray,
            get_covar: bool = False):
        """
        TODO: Maybe implement this as __call__ instead?
        TODO: Update the generation of random noise data to use
            np.random.Generator.
        
        Args:
            x: The observed "explanatory" variables, shape (n, m).
            y: The observed "dependent" variable, shape (n,).
            beta0: The initial parameter estimate, shape (p,)
            weights: The weights associated with each data point, shape (n,)
            get_covar: True if also computing the estimate covariance.

        Returns:
            Nothing. Sets the param_vec_ and covariance_ (if get_covar is True)
            members.
        """
        # Dimensionality of the data
        p = beta0.shape[0]
        if x.ndim > 1:
            n, m = x.shape
        else:
            n, m = len(x), 1

        # Get the step-size scaling matrices.
        # [ S, T ] = self.getScalingMatrices()
        
        # Define the step-size scaling matrices by hand,
        S = np.diag(self.s_)  # s scaling matrix - 10 empirically chosen
        T = np.eye(n * m, dtype=self._dtype)  # t scaling matrix
        alpha = 1  # Lagrange multiplier

        # The m=1 case leads to a simplified form of the D matrix
        if m == 1:
            # Initialize the delta vector
            delta0 = self._rng.normal(0, self._model.std, n).astype(self._dtype)

            # Construct the weighted diagonal matrix D
            D = np.diag(np.multiply(weights, self._model.d)).astype(self._dtype)

        # TODO: Make this work for all m > 1
        # For m>1, the D matrix must be constructed as a "block-diagonal" matrix
        else:
            # Initialize the "interleaved" delta vector
            delta0_theta = np.random.normal(0, self._model.std[0],
                                            n).astype(self._dtype)
            delta0_phi = np.random.normal(0, self._model.std[1],
                                          n).astype(self._dtype)
            delta0 = np.column_stack((delta0_theta, delta0_phi))
            delta0 = delta0.reshape((m * n),)

            # Construct the weighted block-diagonal matrix D
            D_i = np.diag(self._model.d).astype(self._dtype)
            Drep = D_i.reshape(1, m, m).repeat(n, axis=0)
            Dblk = block_diag(*Drep)
            weights_diag = np.diag(np.repeat(weights, m)).astype(self._dtype)
            D = np.matmul(weights_diag, Dblk)

        # Construct the E matrix - E = D^2 + alpha*T^2 (ODR-1987 Prop. 2.1)
        E = np.matmul(D, D) + alpha * np.matmul(T, T)
        Einv = np.linalg.inv(E)

        # Initialize the ODR algorithm
        beta = beta0
        delta = delta0
        s = np.ones((p,), dtype=self._dtype)

        self.iter_ = 1
        while np.linalg.norm(s) > self._converge_thres:

            if m == 1:
                G, V, M = self._getJacobian2D(x, delta, beta, weights, E)
            else:
                # For all m > 1
                G, V, M = self._getJacobian3D(x, delta, beta, weights, E)

            # Update epsilon
            eps = self._model.predict(x, beta, delta) - y
            self._logger.debug(f'eps: {np.linalg.norm(eps)}')

            # Define the following to reduce the number of times certain matrix
            #   products (mp) are computed.
            mp1 = np.matmul(D, delta)
            mp2 = np.linalg.multi_dot([V, Einv, mp1])

            # Form the elements of the linear least squares problem. Note that
            # ybar is actually y 7 in Eqn. 2.13 of ODR-1987, but it's renamed
            # here to avoid a name conflict with the observed data y.
            Gbar = np.matmul(M, G)
            ybar = np.matmul(-M, np.subtract(eps, mp2))

            # Compute step s (the update in beta) via QR factorization of Gbar
            Q, R = np.linalg.qr(Gbar, mode='reduced')
            s = np.linalg.solve(R, np.matmul(Q.T, ybar))
            self._logger.debug(f's: {s}')

            # Compute the step size t (the update in delta)
            # t = -Einv*(V'*M^2*(eps + G*s - V*Einv*D*delta) + D*delta)
            t = np.matmul(
                -Einv,
                np.add(np.linalg.multi_dot(
                    [V.T, np.matmul(M, M), eps + np.matmul(G,s) - mp2]),
                    mp1))

            # Use s and t to iteratively update beta and delta, respectively
            beta = beta + np.matmul(S, s)
            delta = delta + np.matmul(T, t)
            self._logger.debug(f'delta: {np.linalg.norm(delta)}\n')

            self.iter_ += 1
            if self.iter_ > self._max_iterations:
                warnings.warn("ODR: max iterations reached.")
                break

        self._logger.info(f'ODR converged within a step-size tolerance of '
                          f'{self._converge_thres} within {self.iter_} '
                          f'iterations.')

        self.param_vec_ = beta
        if get_covar:
            self._getCovariance(Gbar, D, eps, delta, weights)
        else:
            self.covariance_ = float('nan') * np.ones((p,))

        return

    def _getJacobian2D(self, x, delta, beta, weights, E):
        """
        NOTE: We will use ODRPACK95 notation where the total Jacobian J has
        block components G, V and D:

        J = [G,          V;
             zeros(n,p), D]

        G - the Jacobian matrix of epsilon wrt/ beta and has no special
            properties.
        V - the Jacobian matrix of epsilon wrt/ delta and is a diagonal matrix.
        D - the Jacobian matrix of delta wrt/ delta and is a diagonal matrix.

        Args:
            x: The observed "explanatory" variables, shape (n, m).
            delta: The current estimate of the error in the explanatory
                variables, shape (n*m,).
            beta: The current estimate of the parameter vector, shape (p,).
            weights: The weights associated with each observed datapoint, shape
                (n,).
            E: A non-singular diagonal matrix defined by ODR-1987 Proposition
            2.1.
        """
        n = x.shape[0]
        p = beta.shape[0]

        # initialize
        # G = np.zeros((n, p), dtype=np.float32)
        # V = np.zeros((n, n), dtype=np.float32)
        M = np.zeros((n, n), dtype=np.float32)

        G = self._model.jac_beta(x, beta, delta)
        V = self._model.jac_delta(x, beta, delta)

        # TODO: Can this be vectorized?
        for i in range(n):
            # G[i, :] = weights[i] * np.array(
            #     [np.cos(x[i] + delta[i]), np.sin(x[i] + delta[i])])
            # V[i, i] = weights[i] * (
            #             -beta[0] * np.sin(x[i] + delta[i]) + beta[1] * np.cos(
            #         x[i] + delta[i]))

            # (ODR-1987 Prop. 2.1)
            w = V[i, i] ** 2 / E[i, i]
            M[i, i] = np.sqrt(1 / (1 + w))

        return G, V, M

    def _getJacobian3D(self, x, delta, beta, weights, E):
        """
        NOTE: We will use ODRPACK95 notation where the total Jacobian J has
        block components G, V and D:

        J = [G,          V;
             zeros(n,p), D]

        G - the Jacobian matrix of epsilon wrt/ beta and has no special
            properties.
        V - the Jacobian matrix of epsilon wrt/ delta. For the case where x_i
            is an m-dimensional vector (m > 1), V will have a "staircase"
            structure rather than being a purely diagonal matrix.
        D - the Jacobian matrix of delta wrt/ delta and is a diagonal matrix.

        Args:
            x: The observed "explanatory" variables, shape (n, m).
            delta: The current estimate of the error in the explanatory
                variables, shape (n*m,).
            beta: The current estimate of the parameter vector, shape (p,).
            weights: The weights associated with each observed datapoint, shape
                (n,).
            E: A non-singular diagonal matrix defined by ODR-1987 Proposition
            2.1.
        """

        n, m = x.shape
        p = beta.shape[0]
        # m = int(delta.shape[0] / n)

        theta = x[:, 0]
        phi = x[:, 1]

        # "Un-interleave" delta vector into (n x m) matrix
        delta = delta.reshape((n, m))
        delta_theta = delta[:, 0]
        delta_phi = delta[:, 1]

        # Defined to simplify the following calculations
        x1 = theta + delta_theta
        x2 = phi + delta_phi

        # Initialize
        G = np.zeros((n, p), dtype=self._dtype)
        V = np.zeros((n, n * m), dtype=self._dtype)
        M = np.zeros((n, n), dtype=self._dtype)

        # TODO: Can this be vectorized?
        for i in range(n):
            G[i, :] = weights[i] * np.array([np.cos(x1[i]) * np.cos(x2[i]), \
                                             np.sin(x1[i]) * np.cos(x2[i]), \
                                             np.sin(x2[i])])

            # V[i,2*i:2*i+2] = weights[i] * np.array([
            #     -beta[0]*np.sin(x1[i])*np.cos(x2[i]) + beta[1]*np.cos(x1[i])*np.cos(x2[i]), \
            #     -beta[0]*np.cos(x1[i])*np.sin(x2[i]) - beta[1]*np.sin(x1[i])*np.sin(x2[i]) + \
            #      beta[2]*np.cos(x2[i])])

            V[i, 2 * i] = weights[i] * (
                -beta[0] * np.sin(x1[i]) * np.cos(x2[i]) + \
                 beta[1] * np.cos(x1[i]) * np.cos(x2[i]))

            V[i, 2 * i + 1] = weights[i] * (
                -beta[0] * np.cos(x1[i]) * np.sin(x2[i]) - \
                 beta[1] * np.sin(x1[i]) * np.sin(x2[i]) + \
                 beta[2] * np.cos(x2[i]))

            # (ODR-1987 Prop. 2.1)
            w = (V[i, 2 * i] ** 2 / E[2 * i, 2 * i]) + (
                        V[i, 2 * i + 1] ** 2 / E[2 * i + 1, 2 * i + 1])
            M[i, i] = np.sqrt(1 / (1 + w))

        return G, V, M

    def _getWeights(self):
        pass

    def _getCovariance(self, Gbar, D, eps, delta, weights):
        """
        Computes the (pxp) covariance of the model parameters beta according to
        the method described in "The Computation and Use of the Asymtotic
        Covariance Matrix for Measurement Error Models", Boggs & Rogers (1989).
        """

        n = Gbar.shape[0]  # number of targets in the scan
        p = Gbar.shape[1]  # dimension of the model parameters
        m = int(delta.shape[0] / n)  # dimension of 'explanatory variable' vector

        # Form the complete residual vector, g
        g = np.vstack((np.reshape(eps, (n, 1)), np.reshape(delta, (n * m, 1))))

        # Residual weighting matrix, Omega
        W = np.diag(np.square(weights)).astype(self._dtype)
        Omega1 = np.column_stack((W, np.zeros((n, n * m), dtype=self._dtype)))
        Omega2 = np.column_stack(
            (np.zeros((n * m, n), dtype=self._dtype), np.matmul(D, D)))
        Omega = np.vstack((Omega1, Omega2))

        # Compute total weighted covariance matrix of model parameters (pxp)
        self.covariance_ = \
            (1 / (n - p) * np.linalg.multi_dot([g.T, Omega, g])) * \
            np.linalg.inv(np.matmul(Gbar.T, Gbar))

        return
