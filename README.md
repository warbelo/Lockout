# Lockout
[![PyPI Version][pypi-image]][pypi-url]

Sparsity Inducing Regularization of Fully Connected Neural Networks

## Install

```
pip install lockout [-- upgrade]
```

## Usage
[`PyTorch`](https://pytorch.org/) installation required.  


### 1. Neural Network Architecture
To modify the architecture of the fully connected neural network change either: 
* The number of input features: n_features
* The number of layers: len(layer_sizes)
* The number of nodes in the i<em>th</em> layer: layer_sizes[i]
```
from lockout.pytorch_utils import FCNN

n_features  = 100       
layer_sizes = [10, 1]   
model_init  = FCNN(n_features, layer_sizes)
```
### 2. Create DataLoaders
Previous preprocessing and partitioning of the data is assumed.
```
from lockout.pytorch_utils import make_DataLoaders

dl_train, dl_valid, dl_test = make_DataLoaders(xtrain, xvalid, xtest, ytrain, yvalid, ytest)
```
### 3. Unconstrained Training
Modify the following hyperparameters to fit to your particular problem:
* lr: Learning rate
* loss_type: Type of loss function
    - loss_type=1 (Mean Squared Error) 
    - loss_type=2 (Mean Cross Entropy)
* optim_id: Optimizer 
    - optim_id = 1: Stochastic Gradient Descend
    - optim_id = 2: Adam
* epochs: Maximum number of epochs during training
* early_stopping: Number of epochs used in the convergence condition
* tol_loss: Maxumum change in the training loss function used in the convergence condition
* reset_weights: Whether or not to reset weights before starts training
```
from lockout import Lockout

lr = 1e-2
loss_type = 1
optim_id  = 1

# Instantiate Lockout
lockout_forward = Lockout(model_init, 
                          lr=lr, 
                          loss_type=loss_type, 
                          optim_id=optim_id)

# Train Neural Network Without Regularization
lockout_forward.train(dl_train, dl_valid, 
                      train_how="unconstrained",
                      epochs=10000,
                      early_stopping=20,
                      tol_loss=1e-6,
                      reset_weights=True)
```
The model at the validation minimum and the unconstrained model can be retrieved and/or saved.
```
from lockout.pytorch_utils import save_model

# Save Unconstrained Model
model_forward_unconstrined = lockout_forward.model_last
save_model(model_forward_unconstrined, 'model_forward_unconstrined.pth')

# Save Model At Validation Minimum
model_forward_best = lockout_forward.model_best_valid
save_model(model_forward_best, 'model_forward_best.pth')
```


## Paper

https://arxiv.org/abs/2107.07160

**Abstract:** Regularized regression and classification procedures attempt to fit a function <b>f</b>(<b>x,&omega;</b>) of multiple predictor variables <b>x</b>, to data {<b>x</b><sub>i</sub>,<b>y</b><sub>i</sub>}<sub>1</sub><sup>N</sup>, based on some loss criterion <b>L</b>(y,f) but adding a constraint <b>P</b>(<b>&omega;</b>) &le; t on the joint values of the parameters <b>&omega;</b> to improve accuracy. While there are efficient methods for finding solutions for all values of t &ge; 0 with some constraints <b>P</b> in the special case that <b>f</b> is a linear function, none exist for non linear functions such as Neural Networks (NN). Here we present a fast algorithm that provides all such solutions for any differentiable function <b>f</b> and loss <b>L</b>, and any constraint <b>P</b> that is an increasing monotone function of the absolute value of each parameter. Applications involving sparsity inducing regularization of arbitrary neural networks are discussed. Empirical results indicate that these sparse solutions are usually superior to their dense counterparts in both accuracy and interpretability. This improvement in accuracy can often make neural networks competitive with, and some times superior to, state of the art methods in the analysis of tabular data.


<!-- Badges -->

[pypi-image]: https://img.shields.io/pypi/v/lockout
[pypi-url]: https://pypi.org/project/lockout/
