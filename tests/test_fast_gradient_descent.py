import os
import sys

import unittest
import numpy as np

_curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_curdir + "/..")

from classifiers.fast_gradient_method import fastgradientdescent
from classifiers._classifiers import LogisticClassifier, LinearSVMClassifier

class TestFastGradientDescentWithLogistic(unittest.TestCase):
    def setUp(self):
        np.random.seed(558)
        self.n, self.d = 1000, 4
        self.X = np.random.randn(self.n, self.d)
        self.y = np.sign(self.X[:, 0]**2 - self.X[:, 1]**-2 + np.sin(self.X[:, 2]) - 10*self.X[:, 3])
        self._clf = LogisticClassifier()
        self.beta_init = np.zeros(self.d)
        self.epsilon = 0.0001
        self.lmbda = 0
        self.max_iter = 20

    def test_objective_only_drops_over_iterations(self):
        betas = fastgradientdescent(
            self.X, self.y, self.beta_init, self.epsilon,
            self._clf.computegrad, self._clf.computeobj, lmbda=self.lmbda,
            max_iter=self.max_iter
        )

        objs = [self._clf.computeobj(self.X, self.y, beta, self.lmbda) for beta in betas]

        # assert that the objective function is strictly non-ascending
        self.assertEqual(list(sorted(objs, reverse=True)), objs)

    def test_iteration_stops_for_early_stopping(self):
        epsilon = 1
        betas = fastgradientdescent(
            self.X, self.y, self.beta_init, epsilon,
            self._clf.computegrad, self._clf.computeobj, lmbda=self.lmbda,
            early_stopping=True, max_iter=self.max_iter
        )

        self.assertLessEqual(len(betas), self.max_iter)

    def test_epsilon_is_ignored_if_early_stopping_is_false(self):
        epsilon = 1
        betas = fastgradientdescent(
            self.X, self.y, self.beta_init, epsilon,
            self._clf.computegrad, self._clf.computeobj, lmbda=self.lmbda,
            early_stopping=False, max_iter=self.max_iter
        )

        # the returned betas include beta_init
        self.assertEqual(len(betas), self.max_iter + 1)

    def test_regularized_model_has_smaller_beta_norm(self):
        betas_1 = fastgradientdescent(
            self.X, self.y, self.beta_init, self.epsilon,
            self._clf.computegrad, self._clf.computeobj, lmbda=0,
            max_iter=self.max_iter
        )

        betas_2 = fastgradientdescent(
            self.X, self.y, self.beta_init, self.epsilon,
            self._clf.computegrad, self._clf.computeobj, lmbda=5,
            max_iter=self.max_iter
        )

        beta_1 = betas_1[-1]
        beta_2 = betas_2[-1]

        self.assertLess(np.linalg.norm(beta_2), np.linalg.norm(beta_1))
