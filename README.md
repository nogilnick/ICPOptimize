# ICPOptimize
The Iterative Constrained Pathways Optimizer

## Description

This repository contains implementations of both the Iterative Constrained Pathways (ICP) optimization method and the ICP Rule Ensemble (ICPRE). Further discussion about and motivation for the methods can be found on my blog: 

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
