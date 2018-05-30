"""Each class in this file encapsulates the core mathematical logic
    for a linear classifier

Every class must support the following methods, with the following signatures:
    computegrad(X, y, beta, lmbda): Computes the gradient
    computeobj(X, y, beta, lmbda): Computes the objective function that is being optimized
    predict(X, beta): Computes the predictions for the given X
"""
import numpy as np

class BaseClassifier(object):
    def __init__(self):
        pass

    def computegrad(self, X, y, beta, lmbda):
        """Computes the gradient. Must be overridden.

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

        Returns
        -------
        The gradient w.r.t. beta
        """
        pass

    def computeobj(self, X, y, beta, lmbda):
        """Computes the objective function. Must be overridden.

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

        Returns
        -------
        The objective function w.r.t. beta
        """
        pass


class LogisticClassifier(BaseClassifier):
    """TODO: Cache yhat"""
    def __init__(self):
        pass

    def _calculate_one_minus_p(self, X, y, beta):
        yhat = X @ beta
        numerator = np.exp(-y*yhat)
        denominator = 1 + numerator

        # numerator should be n x 1
        assert(numerator.shape == y.shape)

        return numerator / denominator

    def computegrad(self, X, y, beta, lmbda):
        n = X.shape[0]
        one_minus_p = self._calculate_one_minus_p(X, y, beta)
        P = np.diag(one_minus_p)
        return 2*lmbda*beta - (X.T @ P @ y)/n

    def computeobj(self, X, y, beta, lmbda):
        y_hat = X @ beta

        assert(y_hat.shape == y.shape)

        exp_term = np.exp(-y_hat*y)

        assert(exp_term.shape == y_hat.shape)

        log_cost = np.mean(np.log(1 + exp_term))
        regularization_cost = lmbda*(np.linalg.norm(beta)**2)
        return log_cost + regularization_cost

    def _logit(self, yhat):
        return (1) / (1 + np.exp(-yhat))

    def predict(self, X, beta):
        # The logit function returns a probability (0 - 1)
        prob = self._logit(X @ beta)

        # We location-transform this to be in -0.5 to 0.5, and then use the sign
        return np.sign(prob - 0.5)
