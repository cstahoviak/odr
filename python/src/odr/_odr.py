"""
Implements the Orthogonal Distance Regression (ODR) algorithm described by
"A Stable and Efficient Algorithm for Nonlinear Orthogonal Distance Regression",
Boggs, Byrd and Schnabel, 1987.
"""
import warnings

import numpy as np
from scipy.linalg import block_diag
from scipy.linalg.blas import sgemm, sgemv

from odr import Model

class OrthogonalDistanceRegression:
    """
    TODO: Write docstring
    """
    def __init__(self,
                 model: Model,
                 converge_thres: float = 0.0005,
                 max_iter:int = 50,
                 debug: bool = False,
                 dtype: np.dtype = np.float32):

        # Radar Doppler model (2D or 3D)
        self._model = model
        # ODR convergence threshold on step size s
        self._converge_thres = converge_thres
        # Max number of ODR iterations
        self._max_iterations = max_iter
        # Used for comparison to MATLAB implementation
        self._debug = debug
        self._dtype = dtype

        # Body-frame velocity vector - to be estimated by ODR
        self.param_vec_ = None
        # Covariance of parameter estimate, shape (p,p)
        self.covariance_ = None
        # Number of iterations till convergence
        self.iter_ = 0

        # Step size scale factor
        self.s_ = 10 * np.ones(self._model.min_pts, dtype=self._dtype)


    def odr(self,
            data: np.ndarray,
            beta0: np.ndarray,
            weights: np.ndarray,
            get_covar: bool = False):
        """
        TODO: Maybe implement this as __call__ instead?
        TODO: Update the generation of random noise data to use
            np.random.Generator.
        
        Args:
            data: The measurement data, shape (n, 3).
            beta0: The initial parameter estimate, shape (p,)
            weights: The weights associated with each data point, shape (n,)
            get_covar: True if also computing the estimate covariance.

        Returns:
            Nothing. Sets the param_vec_ and covariance_ (if get_covar is True)
            members.
        """
        # Unpack radar data (into column vectors)
        radar_doppler = data[:, 0]
        radar_azimuth = data[:, 1]
        radar_elevation = data[:, 2]

        # Dimensionality of the data
        n = data.shape[0]
        p = beta0.shape[0]
        m = self._model.d.shape[0]

        # Get the step-size scaling matrices.
        # [ S, T ] = self.getScalingMatrices()
        
        # Define the step-size scaling matrices by hand,
        S = np.diag(self.s_)  # s scaling matrix - 10 empirically chosen
        T = np.eye(n * m, dtype=self._dtype)  # t scaling matrix
        alpha = 1  # Lagrange multiplier

        # 2-dimensional target radar data
        if p == 2:
            # Initialize the delta vector
            delta0 = np.random.normal(0, self._model.std, n).astype(self._dtype)

            # Construct the weighted diagonal matrix D
            D = np.diag(np.multiply(weights, self._model.d)).astype(self._dtype)

        # 3-dimensional target radar data
        elif p == 3:
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

        else:
            raise ValueError("odr: Initial guess must be a 2D or 3D vector.")

        # Construct the E matrix - E = D^2 + alpha*T^2 (ODR-1987 Prop. 2.1)
        E = np.matmul(D, D) + alpha * np.matmul(T, T)
        Einv = np.linalg.inv(E)

        # Initialize the ODR algorithm
        beta = beta0
        delta = delta0
        s = np.ones((p,), dtype=self._dtype)

        self.iter_ = 1
        while np.linalg.norm(s) > self._converge_thres:

            if p == 2:
                G, V, M = self._getJacobian2D(data[:, 1], delta, beta, weights,
                                              E)
            elif p == 3:
                G, V, M = self._getJacobian3D(data[:, 1:3], delta, beta,
                                              weights, E)
            else:
                raise ValueError("odr: Initial guess must be a 2D or 3D vector.")

            doppler_predicted = self._model.predict(
                beta,
                np.column_stack((data[:, 1], data[:, 2])),
                np.zeros(n, dtype=self._dtype),
                delta)

            # Update epsilon
            eps = doppler_predicted - radar_doppler

            # Define the following to reduce the number of times certain matrix
            #   products are computed.
            prod1 = np.matmul(D, delta)
            prod2 = np.linalg.multi_dot([V, Einv, prod1])

            # Form the elements of the linear least squares problem
            Gbar = np.matmul(M, G)
            y = np.matmul(-M, np.subtract(eps, prod2))

            # Compute step s (the update in beta) via QR factorization of Gbar
            Q, R = np.linalg.qr(Gbar, mode='reduced')
            s = np.squeeze(np.linalg.solve(R, np.matmul(Q.T, y)))

            # Compute the step size t (the update in delta)
            # t = -Einv*(V'*M^2*(eps + G*s - V*Einv*D*delta) + D*delta)
            t = np.matmul(
                -Einv,
                np.add(np.linalg.multi_dot([V.T, np.matmul(M, M),
                                            eps + np.matmul(G,s) - prod2]), prod1))

            # Use s and t to iteratively update beta and delta, respectively
            beta = beta + np.matmul(S, s)
            delta = delta + np.matmul(T, t)

            self.iter_ += 1
            if self.iter_ > self._max_iterations:
                warnings.warn("ODR: max iterations reached.")
                break

        self.param_vec_ = beta
        if get_covar:
            self._getCovariance(Gbar, D, eps, delta, weights)
        else:
            self.covariance_ = float('nan') * np.ones((p,))

        return

    def _getJacobian2D(self, X, delta, beta, weights, E):
        """
        NOTE: We will use ODRPACK95 notation where the total Jacobian J has
        block components G, V and D:

        J = [G,          V;
             zeros(n,p), D]

        G - the Jacobian matrix of epsilon wrt/ beta and has no special
            properties.
        V - the Jacobian matrix of epsilon wrt/ delta and is a diagonal matrix.
        D - the Jacobian matrix of delta wrt/ delta and is a diagonal matrix.
        """

        n = X.shape[0]  # X is a column vector of azimuth values
        p = beta.shape[0]

        # initialize
        G = np.zeros((n, p), dtype=np.float32)
        V = np.zeros((n, n), dtype=np.float32)
        M = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            G[i, :] = weights[i] * np.array(
                [np.cos(X[i] + delta[i]), np.sin(X[i] + delta[i])])
            V[i, i] = weights[i] * (
                        -beta[0] * np.sin(X[i] + delta[i]) + beta[1] * np.cos(
                    X[i] + delta[i]))

            ## (ODR-1987 Prop. 2.1)
            w = V[i, i] ** 2 / E[i, i]
            M[i, i] = np.sqrt(1 / (1 + w));

        return G, V, M

    def _getJacobian3D(self, X, delta, beta, weights, E):
        """
        NOTE: We will use ODRPACK95 notation where the total Jacobian J has
        block components G, V and D:

        J = [G,          V;
             zeros(n,p), D]

        G - the Jacobian matrix of epsilon wrt/ beta and has no special
            properties.
        V - the Jacobian matrix of epsilon wrt/ delta and is a diagonal matrix.
        D - the Jacobian matrix of delta wrt/ delta and is a diagonal matrix.
        """

        n = X.shape[0]  # X is a column vector of azimuth values
        p = beta.shape[0]
        m = int(delta.shape[0] / n)

        theta = X[:, 0]
        phi = X[:, 1]

        ## "un-interleave" delta vector into (n x m) matrix
        delta = delta.reshape((n, m))
        delta_theta = delta[:, 0]
        delta_phi = delta[:, 1]

        ## defined to simplify the following calculations
        x1 = theta + delta_theta
        x2 = phi + delta_phi

        # initialize
        G = np.zeros((n, p), dtype=self._dtype)
        V = np.zeros((n, n * m), dtype=self._dtype)
        M = np.zeros((n, n), dtype=self._dtype)

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

            ## (ODR-1987 Prop. 2.1)
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
