import os
import sys

import unittest
import numpy as np

_curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_curdir + "/..")

import datasets
import datasets.spam
import datasets.imagenet_small

class TestDataSetLoading(unittest.TestCase):
    def test_spam_loads_via_datasets(self):
        X_train, X_test, y_train, y_test = datasets.load_spam()
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])
        self.assertEqual(X_train.shape[1], X_test.shape[1])

    def test_spam_loads_via_spam(self):
        X_train, X_test, y_train, y_test = datasets.spam.load_dataset()
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])
        self.assertEqual(X_train.shape[1], X_test.shape[1])

    def test_imagenet_loads_via_datasets(self):
        X_train, X_test, y_train, y_test = datasets.load_imagenet()
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])
        self.assertEqual(X_train.shape[1], X_test.shape[1])

    def test_imagenet_loads_via_imagenet(self):
        X_train, X_test, y_train, y_test = datasets.imagenet_small.load_dataset()
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])
        self.assertEqual(X_train.shape[1], X_test.shape[1])
