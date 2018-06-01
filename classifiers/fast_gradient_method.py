"""Implementations of the fast gradient method, and the backtracking method

The fast gradient method uses a concept of "momentum" to speed up the descent.

The backtracking algorithm is a supplementary algorithm that leverages the
convexity of the objective function to its advantage to determine the most optimal learning rate
for the gradient descent.
"""

import numpy as np

def backtracking(X, y, beta, lmbda, eta_init, computegrad, computeobj, alpha=0.5, decay=0.8, max_iter=100):
    """Perform backtracking line search

    Parameters
    ----------
    X: np.ndarray
        Feature/predictor matrix of shape (n x d)

    y: np.array | np.ndarray
        Outcome/response array of shape (n,) or (n, 1)

    beta: np.array | np.ndarray
        Coefficient array of shape (d,) or (d, 1)

    lmbda: float
        Regularization coefficient

    eta_init: float
        Starting (maximum) step size

    computegrad: callable
        A callable that accepts X, y, beta and lambda, and computes the gradient

    computeobj: callable
        A callable that accepts X, y, beta and lambda, and computes the objective function

    alpha: float, alpha=0.5
        Constant used to define sufficient decrease condition

    decay: float, default=0.8
        Fraction by which we decrease t if the previous t doesn't work

    max_iter: int, default=100
        Maximum number of iterations to run the algorithm

    Returns
    -------
    eta: Step size to use

    Raises
    ------
    ValueError:
        if lmbda is negative
        if eta is non-positive
    """
    if lmbda < 0:
        raise ValueError("lmbda (regularization coefficient) must be strictly non-negative")

    if eta_init <= 0:
        raise ValueError("eta_init (initial learning rate) must be strictly positive")

    grad_beta = computegrad(X, y, beta, lmbda)
    norm_grad_beta_sq = np.linalg.norm(grad_beta) ** 2
    i = 0

    eta = eta_init

    beta_obj = computeobj(X, y, beta, lmbda)

    armijo_goldstein_condition_satisfied = False

    while i < max_iter and not armijo_goldstein_condition_satisfied:
        new_beta = beta - eta * grad_beta
        new_beta_obj = computeobj(X, y, new_beta, lmbda)
        armijo_goldstein_condition_satisfied = beta_obj - new_beta_obj >= alpha * eta * norm_grad_beta_sq

        eta *= decay
        i += 1

    return eta


def fastgradientdescent(X, y, beta_init, epsilon, computegrad, computeobj, eta=1, max_iter=1000, lmbda=0.1,
                        early_stopping=True, use_backtracking=True):
    """Run fast gradient descent

    Parameters
    ----------

    X: ndarray
        ndarray with shape (n, d)

    Y: array | ndarray
        array with shape (n,) or (n, 1)

    beta_init: array | ndarray
        Initialization parameters for regression coefficients,
        with shape (d, ) or (d, 1)

    epsilon: float
        Target accuracy to be achieved, ignored if early_stopping is False

    computegrad: callable
        A callable that accepts X, y, beta and lambda, and computes the gradient

    computeobj: callable
        A callable that accepts X, y, beta and lambda, and computes the objective function

    eta: float
        Initial learning rate for the algorithm

    max_iter: int
        The maximum number of iterations to run gradient descent

    lmbda: float
        Regularization parameter

    early_stopping: boolean, default=True
        If True, then the norm of the gradient is compared against epsilon,
        and if the norm is smaller, the iterative method is halted.

    use_backtracking: boolean, default=True
        If True, then backtracking line search is used at each iteration to find the
        best value of eta (learning rate)

    Returns
    -------
    betas
        A list of (at maximum) `max_iter` betas, each of which is a np.array of size (d,)
    """
    beta = beta_init
    beta_prev = beta
    betas = [beta_init]
    theta = np.zeros(beta_init.shape)
    gradient_norm = epsilon + 100

    i = 0

    while i < max_iter and ((not early_stopping) or gradient_norm > epsilon):
        gradient = computegrad(X, y, theta, lmbda)
        gradient_norm = np.linalg.norm(gradient)

        if use_backtracking:
            eta = backtracking(X, y, beta, lmbda, eta, computegrad, computeobj)

        beta = theta - eta*gradient
        theta = beta + (i/(i + 3))*(beta - beta_prev)

        beta_prev = beta
        betas.append(beta)

        i += 1

    return betas
