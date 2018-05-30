"""Classification using the Fast Gradient Method.

The name and style is inspired from the SGDClassifier in sklearn
"""

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from ._classifiers import LogisticClassifier, LinearSVMClassifier
from .fast_gradient_method import fastgradientdescent

_CLASSIFIERS = {
    'logistic': LogisticClassifier(),
    'svm': LinearSVMClassifier(),
}

class FGMClassifier(BaseEstimator):
    """Linear classifiers (SVM, logistic regression, a.o.) with fast gradient method

    By default, the model uses L2-Regularized.

    Parameters
    ----------
    algo: str
        One of 'logistic' or 'svm'

    lmbda: float, default=0
        Regularization coefficient.
        By default, the model is unregularized.

    epsilon: float, default=0.0001
        An estimate of desired accuracy.
        The descent algorithm will stop if the norm of the gradient is smaller than this value

    learning_rate: str, one of 'constant' or 'adaptive', default='adaptive'
        If constant, then eta is used as the learning rate
        If adaptive, then eta is updated after each iteration

    eta: float
        The learning rate that is used for the gradient descent.

    max_iter: int, default=100
        The maximum number of iterations for which fast gradient method should be run
    """
    def __init__(self, classifier='logistic', lmbda=0, epsilon=0.0001, learning_rate='adaptive', eta=1, max_iter=100):
        self.classifier_method = classifier
        self._classifier = _CLASSIFIERS[classifier]
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.eta = eta
        self.max_iter = max_iter

        self._betas = None
        self._coef = None

    @property
    def classifier(self):
        return self.classifier_method

    @classifier.setter
    def classifier(self, classifier):
        self.classifier = classifier
        self._classifier = _CLASSIFIERS[classifier]


    def fit(self, X, y, verbose=False):
        """Fit the model using fast gradient method

            Parameters
            ----------
            X: np.ndarray
                Feature/predictor matrix of shape (n x d)

            y: np.array | np.ndarray
                Outcome/response array of shape (n,) or (n, 1)
                y must contain only 1/-1
        """
        use_backtracking = self.learning_rate == 'adaptive'
        _, d = X.shape
        beta_init = np.zeros(d)
        self._betas = fastgradientdescent(X, y, beta_init, self.epsilon,
                                          self._classifier.computegrad, self._classifier.computeobj,
                                          eta=self.eta, max_iter=self.max_iter, lmbda=self.lmbda,
                                          use_backtracking=use_backtracking)

        self._coef = self._betas[-1]
        return self

    def predict(self, X):
        return self._classifier.predict(X, self._coef)

    def predict_proba(self, X):
        return self._classifier.predict_proba(X, self._coef)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
