from sklearn import preprocessing

def standardize(X_train, X_test):
    # Standardize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, X_test)

def load_spam(standardized=True, with_intercept=True):
    from .spam import load_dataset
    X_train, X_test, y_train, y_test = load_dataset()
    if standardized:
        X_train, X_test = standardize(X_train, X_test)

    return X_train, X_test, y_train, y_test
