import os
import sys

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(process)d] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

_curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_curdir + "/..")

from classifiers import FGMClassifier
import datasets

def train_spam_untuned(classifier, X_train, X_test, y_train, y_test):
    clf = FGMClassifier(classifier=classifier, lmbda=0.1, max_iter=20, learning_rate='adaptive', eta=1)
    return fit_predict(clf, X_train, X_test, y_train, y_test)

def train_spam_tuned(X_train, X_test, y_train, y_test):
    param_grid = {
        'classifier': ['logistic', 'svm'],
        'lmbda': np.linspace(0, 1, 3)
    }
    clf = GridSearchCV(FGMClassifier(max_iter=10, verbose=True), param_grid, verbose=2)

    return fit_predict(clf, X_train, X_test, y_train, y_test)


def fit_predict(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    logging.info("Predicting...")
    logging.info("Training accuracy: {}".format(clf.score(X_train, y_train)))
    logging.info("Test accuracy: {}".format(clf.score(X_test, y_test)))

    return clf

def main():
    logging.info("Loading data...")
    X_train, X_test, y_train, y_test = datasets.load_spam(standardized=True, with_intercept=True)

    logging.info("Training logistic without tuning lambda...")
    train_spam_untuned('logistic', X_train, X_test, y_train, y_test)

    logging.info("Training svm without tuning lambda...")
    train_spam_untuned('svm', X_train, X_test, y_train, y_test)

    logging.info("Training with GridSearchCV...")
    clf = train_spam_tuned(X_train, X_test, y_train, y_test)

    logging.info("Best params: {}".format(clf.best_params_))
    logging.info("Best estimator: {}".format(clf.best_estimator_))

if __name__ == "__main__":
    main()
