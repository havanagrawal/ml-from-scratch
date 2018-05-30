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
from datasets import spam

def standardize(X_train, X_test):
    # Standardize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, X_test)

def train_spam_untuned(X_train, X_test, y_train, y_test):
    clf = FGMClassifier(lmbda=0.1, max_iter=50, learning_rate='adaptive', eta=1)
    logging.info("Training...")
    clf.fit(X_train, y_train)

    logging.info("Predicting...")
    print("Training accuracy: ", clf.score(X_train, y_train))
    print("Test accuracy: ", clf.score(X_test, y_test))

def train_spam_tuned(X_train, X_test, y_train, y_test):
    param_grid = {
        'lmbda': np.linspace(0, 1, 10)
    }
    clf = GridSearchCV(FGMClassifier(), param_grid, verbose=2)

    logging.info("Training...")
    clf.fit(X_train, y_train)

    logging.info("Predicting...")
    print("Training accuracy: ", clf.score(X_train, y_train))
    print("Test accuracy: ", clf.score(X_test, y_test))

def main():
    logging.info("Loading data...")
    X_train, X_test, y_train, y_test = spam.load_dataset()
    X_train, X_test = standardize(X_train, X_test)

    X_train = preprocessing.add_dummy_feature(X_train)
    X_test = preprocessing.add_dummy_feature(X_test)

    logging.info("Training without tuning lambda...")
    train_spam_untuned(X_train, X_test, y_train, y_test)

    logging.info("Training with GridSearchCV...")
    train_spam_tuned(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
