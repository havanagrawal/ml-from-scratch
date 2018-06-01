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
        return 1 / (1 + np.exp(-yhat))

    def predict(self, X, beta):
        # The logit function returns a probability (0 - 1)
        prob = self._logit(X @ beta)

        # We location-transform this to be in -0.5 to 0.5, and then use the sign
        return np.sign(prob - 0.5)

    def predict_proba(self, X, beta):
        # The logit function returns a probability (0 - 1)
        return self._logit(X @ beta)


class LinearSVMClassifier(BaseClassifier):
    """Linear SVM with Huberized Hinge Loss

    Parameters
    ----------
    h: float, default=0.5
        Smoothing factor
    """
    def __init__(self, h=0.5):
        self.h = h

    def computeobj(self, X, y, beta, lmbda):
        h = self.h

        yhat = X @ beta
        yt = yhat * y # element wise product

        l_hh = np.zeros(yt.shape)

        mask_1 = yt > 1 + h
        mask_2 = np.abs(1 - yt) <= h
        mask_3 = yt < 1 - h

        l_hh[mask_1] = 0
        l_hh[mask_2] = ((1 + h - yt[mask_2])**2) / 4*h
        l_hh[mask_3] = 1 - yt[mask_3]

        return lmbda*(np.linalg.norm(beta)**2) + np.mean(l_hh)

    def computegrad(self, X, y, beta, lmbda):
        n, d = X.shape
        h = self.h

        yhat = X @ beta
        yt = yhat * y # element wise product

        mask_2 = np.abs(1 - yt) <= h
        mask_3 = yt < 1 - h

        temp = 2*y[mask_2]*(1 + h - yt[mask_2])/4*h

        grad_l_hh_1 = np.zeros(d)
        grad_l_hh_2 = -X[mask_2, :].T @ temp
        grad_l_hh_3 = -X[mask_3, :].T @ y[mask_3]

        gradient = 2*lmbda*beta + (grad_l_hh_1 + grad_l_hh_2 + grad_l_hh_3) / n

        return gradient

    def predict(self, X, beta):
        return np.sign(X @ beta)
