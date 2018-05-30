# ML From Scratch

[![Build Status](https://travis-ci.org/havanagrawal/ml-from-scratch.svg?branch=master)](https://travis-ci.org/havanagrawal/ml-from-scratch) [![codecov](https://codecov.io/gh/havanagrawal/ml-from-scratch/branch/master/graph/badge.svg)](https://codecov.io/gh/havanagrawal/ml-from-scratch)


## Introduction

This repository contains (what is hopefully productionized) snippets of code that are either just practice exercises, or logical extensions to the sklearn library.

A small section of this is also intended as a submission for DATA558/BIOST558 - Introduction to Machine Learning "Polished Code Release" assignment.

## Submission for BIOST558

The main submission for BIOST558 is the [FGMClassifier](classifiers/fgm_classifier.py) which packages the fast gradient method into a convenient class. The style and naming is inspired from sklearn's [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

## Implementation Note

In all of the algorithms in this repository, unless explicitly stated otherwise, the convention for binary classification labels is -1/+1, as opposed to 0/1.

## Examples

Examples in the form of notebooks can be found in the [examples/notebooks](examples/notebooks) directory.
