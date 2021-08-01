# Lockout
[![PyPI Version][pypi-image]][pypi-url]

Sparsity Inducing Regularization of Fully Connected Neural Networks

## Install

```
pip install lockout [-- upgrade]
```

## Usage
[`PyTorch`](https://pytorch.org/) installation required.  


### 1. Import Lockout (and PyTorch)
```
import torch
from lockout import Lockout
```

### 2. Neural Network Architecture
```
```

## Paper

https://arxiv.org/abs/2107.07160

**Abstract:** Regularized regression and classification procedures attempt to fit a function <b>f</b>(<b>x,&omega;</b>) of multiple predictor variables <b>x</b>, to data {<b>x</b><sub>i</sub>,<b>y</b><sub>i</sub>}<sub>1</sub><sup>N</sup>, based on some loss criterion <b>L</b>(y,f) but adding a constraint <b>P</b>(<b>&omega;</b>) &le; t on the joint values of the parameters <b>&omega;</b> to improve accuracy. While there are efficient methods for finding solutions for all values of t &ge; 0 with some constraints <b>P</b> in the special case that <b>f</b> is a linear function, none exist for non linear functions such as Neural Networks (NN). Here we present a fast algorithm that provides all such solutions for any differentiable function <b>f</b> and loss <b>L</b>, and any constraint <b>P</b> that is an increasing monotone function of the absolute value of each parameter. Applications involving sparsity inducing regularization of arbitrary neural networks are discussed. Empirical results indicate that these sparse solutions are usually superior to their dense counterparts in both accuracy and interpretability. This improvement in accuracy can often make neural networks competitive with, and some times superior to, state of the art methods in the analysis of tabular data.


<!-- Badges -->

[pypi-image]: https://img.shields.io/pypi/v/lockout
[pypi-url]: https://pypi.org/project/lockout/
