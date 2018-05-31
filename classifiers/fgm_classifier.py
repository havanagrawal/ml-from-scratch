"""Classification using the Fast Gradient Method.

The name and style is inspired from the SGDClassifier in sklearn
"""
from collections import defaultdict, Counter
from itertools import combinations

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(process)d] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from ._classifiers import LogisticClassifier, LinearSVMClassifier
from .fast_gradient_method import fastgradientdescent

_CLASSIFIERS = {
    'logistic': LogisticClassifier(),
    'svm': LinearSVMClassifier(),
}

class FGMClassifier(BaseEstimator, ClassifierMixin):
    """Linear classifiers (SVM, logistic regression, a.o.) with fast gradient method

    By default, the model uses L2-Regularization.

    Multi-class support is provided using the One-vs-One approach, i.e. if there are
    n classes, then n*(n - 1)/2 models will be trained.

    Parameters
    ----------
    classifier: str
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

    verbose: boolean, default=False
        Whether or not to print progress messages
    """
    def __init__(self, classifier='logistic', lmbda=0, epsilon=0.0001, learning_rate='adaptive', eta=1, max_iter=100, verbose=False):
        self.models = defaultdict(dict)
        self.classifier = classifier
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.eta = eta
        self.max_iter = max_iter
        self.verbose = verbose
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for clz1, clz2 in combinations(self.classes, 2):
            if self.verbose:
                logging.info("Training for {} vs {}".format(clz1, clz2))
            self.models[clz1][clz2] = self._fit_class(X, y, clz1, clz2)

        return self

    def _fit_class(self, X, y, clz1, clz2):
        data_mask = np.logical_or(y == clz1, y == clz2)

        X = X[data_mask, :]
        y = y[data_mask]

        y = np.where(y == clz1, 1, -1)

        clf = FGMBinaryClassifier(self.classifier, self.lmbda, self.epsilon, self.learning_rate, self.eta, self.max_iter)
        clf.fit(X, y)
        return clf

    def predict(self, X):
        all_predictions = []

        for clz1, clz2 in combinations(self.classes, 2):
            model = self.models[clz1][clz2]

            pred = model.predict(X)
            pred = np.where(pred == 1, clz1, clz2)

            all_predictions.append(pred)

        all_predictions = np.array(all_predictions).T

        majority_vote_predictions = [int(Counter(i).most_common(1)[0][0]) for i in all_predictions]

        return majority_vote_predictions


class FGMBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Binary Linear classifiers (SVM, logistic regression, a.o.) with fast gradient method

    By default, the model uses L2-Regularization.

    Parameters
    ----------
    classifier: str
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
