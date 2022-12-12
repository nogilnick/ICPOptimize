# ICPOptimize
The Iterative Constrained Pathways Optimizer

ICP is a constrained linear model optimizer built with a focus on memory efficiency, flexibility, and solution interpretability.

## Description

This repository contains implementations of the Iterative Constrained Pathways (ICP) optimization method, the ICP Rule Ensemble (ICPRE), linear classifier, regressor, and other methods. Currently, hinge, least-squares, and absolute-value loss modes are supported, with support for other loss functions planned. Coefficients can be constrained by sign or by arbitrary intervals. L1 & L2 norm constraints are also supported.

Further discussion about and motivation for the methods can be found on my blog:

[nogilnick.github.io/Posts/63.html](https://nogilnick.github.io/Posts/63.html)

## Features

- Linear Classification using Hinge Loss
- Regression Support using L1 and L2 Penalties
- Arbitrary Interval Constraints
- L1 and L2 Coefficient Norm Constraints
- Useful Default Settings
- Support for DataFrames and Sparse Matrices

## Installation

Install via PyPi:

```pip install ICPOptimize```

PyPi Project:

[https://pypi.org/project/ICPOptimize/](https://pypi.org/project/ICPOptimize/)

## Examples

### Rule Ensemble Classifier
```python
from ICP.Models import ICPRuleEnsemble

...

IRE = ICPRuleEnsemble().fit(A[trn], Y[trn])
YP  = IRE.predict_proba(A)
```

### Linear Model
```python
from ICP.Models import ICPLinearRegressor

...
# Fit linear regressor with absolute loss and L_1 norm <= 10
ILR = ICPLinearRegressor(p='l1', L1=10.0).fit(A[trn], Y[trn])
YP  = ILR.predict(A)
```

Further examples are available on the ICPExamples GitHub page:

[https://github.com/nogilnick/ICPExamples](https://github.com/nogilnick/ICPExamples)
