# ML From Scratch

[![Build Status](https://travis-ci.org/havanagrawal/ml-from-scratch.svg?branch=master)](https://travis-ci.org/havanagrawal/ml-from-scratch) [![codecov](https://codecov.io/gh/havanagrawal/ml-from-scratch/branch/master/graph/badge.svg)](https://codecov.io/gh/havanagrawal/ml-from-scratch) ![Python Version](https://img.shields.io/badge/python-3.6-blue.svg)


## Introduction

This repository contains (what is hopefully productionized) snippets of code that are either just practice exercises, or logical extensions to the sklearn library.

A small section of this is also intended as a submission for DATA558/BIOST558 - Introduction to Machine Learning "Polished Code Release" assignment.

## Submission for BIOST558

The main submission for BIOST558 is the [FGMClassifier](classifiers/fgm_classifier.py) which packages the fast gradient method into a convenient class. The style and naming is inspired from sklearn's [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

## How to Run

Create a virtual environment:

```
python3 -m virtualenv biost558
source biost558/bin/activate
```

Install the requirements and package using:

```
pip install -r requirements.txt
python setup.py install
```

Now the classes can be imported as usual:

```python
from classifiers import FGMClassifier
from classifiers import FGMBinaryClassifier
```

The classes are sklearn-compatible, i.e. can be used in conjunction with [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), etc.

To run the Jupyter notebooks, you need to first install a local copy of Jupyter and ipykernel in the virtualenv:

```
python -m ipykernel install --user
```

Then the Jupyter notebook can be started as usual:

```
jupyter notebook
```

## Examples

Examples in the form of notebooks can be found in the [examples/notebooks](examples/notebooks) directory.

### Implementation Note

In all of the algorithms in this repository, unless explicitly stated otherwise, the convention for binary classification labels is -1/+1, as opposed to 0/1.
