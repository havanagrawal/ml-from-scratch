import os
import sys

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(process)d] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from sklearn import preprocessing
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

def main():
    logging.info("Loading data...")
    X_train, X_test, y_train, y_test = spam.load_dataset()
    X_train, X_test = standardize(X_train, X_test)

    X_train = preprocessing.add_dummy_feature(X_train)
    X_test = preprocessing.add_dummy_feature(X_test)

    clf = FGMClassifier(lmbda=0.1, max_iter=20, learning_rate='adaptive', eta=1)
    logging.info("Training...")
    clf.fit(X_train, y_train)

    logging.info("Predicting...")
    predictions = clf.predict(X_test)
    print(accuracy_score(y_train, clf.predict(X_train)))
    print(accuracy_score(y_test, clf.predict(X_test)))

if __name__ == "__main__":
    main()
