# ICPOptimize
The Iterative Constrained Pathways Optimizer

ICP is a constrained linear model optimizer built with a focus on memory efficiency, flexibility, and solution interpretability.

## Description

This repository contains implementations of the Iterative Constrained Pathways (ICP) optimization method, the ICP Rule Ensemble (ICPRE), linear classifier, regressor, and other methods. Currently, hinge and least-squares loss are supported. Support for other loss functions is planned.

Further discussion about and motivation for the methods can be found on my blog: 

[nicholastsmith.wordpress.com/2021/05/18/the-iterative-constrained-pathways-optimizer/](https://nicholastsmith.wordpress.com/2021/05/18/the-iterative-constrained-pathways-optimizer/)

## Installation

Install via PyPi:

```pip install ICPOptimize```

PyPi Project:

[https://pypi.org/project/ICPOptimize/](https://pypi.org/project/ICPOptimize/)

## Examples

```python
from ICP.Models import ICPRuleEnsemble

...

IRE = ICPRuleEnsemble().fit(A[trn], Y[trn])
YP  = IRE.predict_proba(A)
```

Further examples are available on the ICPExamples GitHub page:

[https://github.com/nicholastoddsmith/ICPExamples](https://github.com/nicholastoddsmith/ICPExamples)
