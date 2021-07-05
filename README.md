# Lockout
Regularized regression and classification procedures attempt to fit a
function $f(\mathbf{x},\mathbf{w})$ of multiple predictor variables
$\mathbf{x}$, to data $\{\mathbf{x}_{i},y_{i}\}_1^N$, based on some loss criterion
$L(y,f)$ but adding a constraint $P(\mathbf{w})\leq t$ on the joint values
of the parameters $\mathbf{w}$ to improve accuracy. While there are efficient methods for finding solutions for all values of $t\geq0$ with some constraints $P$ in the special case that $f$ is a linear function, none exist for non linear functions such as Neural Networks (NN). Here we present a fast algorithm that provides all such solutions for any differentiable function $f$ and
loss $L$, and any constraint $P$ that is an increasing monotone
function of the absolute value of each parameter. Applications involving
sparsity inducing regularization of arbitrary neural networks are discussed.
Empirical results indicate that these sparse solutions are usually superior to their dense counterparts in both accuracy and interpretability. This improvement in accuracy can often make neural networks competitive with, and some times superior to, state of the art methods in the analysis of tabular data.
