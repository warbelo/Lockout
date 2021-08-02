# Lockout
[![PyPI Version][pypi-image]][pypi-url]

Sparsity Inducing Regularization of Fully Connected Neural Networks

## Install

```
pip install lockout [-- upgrade]
```

## Usage
[`PyTorch`](https://pytorch.org/) installation required.  

### **1.** Neural Network Architecture
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

### **2.** Create DataLoaders
Previous preprocessing and partitioning of the data is assumed.
```
from lockout.pytorch_utils import make_DataLoaders

dl_train, dl_valid, dl_test = make_DataLoaders(xtrain, xvalid, xtest, ytrain, yvalid, ytest)
```

### **3.** Unconstrained Training
Modify the following hyperparameters according to your particular problem:
* lr: Learning rate
* loss_type: Type of loss function
    - loss_type=1 (Mean Squared Error) 
    - loss_type=2 (Mean Cross Entropy)
* optim_id: Optimizer 
    - optim_id = 1: Stochastic Gradient Descend
    - optim_id = 2: Adam
* epochs: Maximum number of epochs during training
* early_stopping: Number of epochs used in the convergence condition
* tol_loss: Maximum change in the training loss function used in the convergence condition
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

The model at the validation minimum and the unconstrained model can be retrieved and saved for further use.
```
from lockout.pytorch_utils import save_model

# Save Unconstrained Model
model_forward_unconstrained = lockout_forward.model_last
save_model(model_forward_unconstrained, 'model_forward_unconstrained.pth')

# Save Model At Validation Minimum
model_forward_best = lockout_forward.model_best_valid
save_model(model_forward_best, 'model_forward_best.pth')
```

Loss and accuracy curves can be retrieved for analysis or graphing. For regression problems, R2 is computed as the accuracy.
```
df0 = lockout_forward.path_data
df0.head()
```
<p align="left">
  <img src="Doc/path_data0.png" width="400" title="Loss and accuracy curves for unconstrained training">
</p>

```
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(figsize=(9,6))
axes.plot(df0["iteration"], df0["train_loss"], label="Training", linewidth=4)
axes.plot(df0["iteration"], df0["valid_loss"], label="Validation", linewidth=4)
axes.legend(fontsize=16)
axes.set_xlabel("iteration", fontsize=16)
axes.set_ylabel("Loss Function", fontsize=16)
axes.tick_params(axis='both', which='major', labelsize=14)
axes.set_title("Unconstrained", fontsize=16)
axes.grid(True, zorder=2)
plt.show()
```
<p align="left">
  <img src="Doc/loss_vs_iter_forward.png" width="400" title="Loss vs iteration for unconstrained training">
</p>

### **4.** Lockout Training: Option 1
Within this option, the network is first trained until the regularization path is found (Path 1). Then, the constraint t<sub>0</sub> is iteratively decreased during training with a stepsize &Delta;t<sub>0</sub> inversely proportional to the number of epochs (Path 2).
A small &Delta;t<sub>0</sub> is necessary to stay on the regularization path.
<br>
Modify the following hyperparameters according to your particular problem:
* input_model: input model, either unconstrained or at validation minimum
* regul_type: list of tuples (or dictionary) of the form [(layer_name, regul_id)] where:
    - layer_name: layer name in the input model (string)
    - regul_id = 1: L1 regularization
    - regul_id = 2: Log regularization (see get_constraint function)
* regul_path: list of tuples (or dictionary) of the form [(layer_name, path_flg)] where:
    - path_flg = True: the constraint t<sub>0</sub> will be iteratively decreased in this layer
    - path_flg = False: the constraint t<sub>0</sub> will be kept constant in this layer
* epochs: maximum number of epochs used to bring the network to the regularization path (Path 1)
* epochs2: maximum number of epochs used while training decreasing t<sub>0</sub> (Path 2)

```
from lockout import Lockout

regul_type = [('linear_layers.0.weight', 1)]
regul_path = [('linear_layers.0.weight', True)]

# Instantiate Lockout
lockout_option1 = Lockout(lockout_forward.model_best_valid,
                          lr=1e-2, 
                          loss_type=1,
                          regul_type=regul_type,
                          regul_path=regul_path)

# Train Neural Network With Lockout
lockout_option1.train(dl_train, dl_valid, 
                      train_how="decrease_t0", 
                      epochs=5000,
                      epochs2=20000,
                      early_stopping=20, 
                      tol_loss=1e-5)
```

The model at the validation minimum can be retrieved and saved for further use.
```
from lockout.pytorch_utils import save_model

# Save Model At Validation Minimum
model_lockout_option1 = lockout_option1.model_best_valid
save_model(model_lockout_option1, 'model_lockout_option1.pth')
```

Path data can be retrieved for analysis or graphing.
```
df1 = lockout_forward.path_data
df1.head()
```
<p align="left">
  <img src="Doc/path_data1.png" width="900" title="Path data for lockout training">
</p>

Test accuracy can be computed using the models previously trained.
```
import torch
from lockout.pytorch_utils import dataset_r2

device = torch.device('cpu')
r2_test_forward, _  = dataset_r2(dl_test, model_forward_best, device)
r2_test_lockout1, _ = dataset_r2(dl_test, model_lockout_option1, device)
print("Test R2 (unconstrained) = {:.3f}".format(r2_test_forward))
print("Test R2 (lockout)       = {:.3f}".format(r2_test_lockout1))
```
<p align="left">
  <img src="Doc/r2_test1.png" width="280" title="Test R2">
</p>

Feature importance can be computed and graphed.
```
import matplotlib.pyplot as plt
import numpy as np
from lockout.pytorch_utils import get_features_importance

importance = get_features_importance(model_lockout_option1, 'linear_layers.0.weight')

fig, axes = plt.subplots(figsize=(9,6))
x_pos = np.arange(len(importance))
axes.bar(x_pos, importance, zorder=2)
axes.set_xticks(x_pos)
axes.set_xticklabels(importance.index, rotation='vertical')
axes.set_xlim(-1,len(x_pos))
axes.tick_params(axis='both', which='major', labelsize=14)
axes.set_ylabel('Importance', fontsize=16)
axes.set_xlabel('feature', fontsize=16)
axes.set_title('Lockout', fontsize=16)
axes.grid(True, zorder=1)
plt.show()
```
<p align="left">
  <img src="Doc/feature_importance_lockout1.png" width="500" title="feature importance with lockout">
</p>


## Paper

https://arxiv.org/abs/2107.07160

<p align="left">
  <img src="Doc/abstract.png" width="900" title="paper abstract">
</p>

<!-- **Abstract:** Regularized regression and classification procedures attempt to fit a function $f(\bm{x}, \bm{\omega})$ of multiple predictor variables $\bm{x}$, to data $\{\bm{x}_i, \bm{y}_i\}^N_i$, based on some loss criterion $L(y,f)$ but adding a constraint $P(\bm{\omega})\le t_0$ on the joint values of the parameters $\bm{\omega}$ to improve accuracy. While there are efficient methods for finding solutions for all values of $t_0 \ge 0$ with some constraints $P$ in the special case that $f$ is a linear function, none exist for non linear functions such as Neural Networks (NN). Here we present a fast algorithm that provides all such solutions for any differentiable function $f$ and loss $L$, and any constraint $P$ that is an increasing monotone function of the absolute value of each parameter. Applications involving sparsity inducing regularization of arbitrary neural networks are discussed. Empirical results indicate that these sparse solutions are usually superior to their dense counterparts in both accuracy and interpretability. This improvement in accuracy can often make neural networks competitive with, and some times superior to, state of the art methods in the analysis of tabular data. -->

<!-- Badges -->

[pypi-image]: https://img.shields.io/pypi/v/lockout
[pypi-url]: https://pypi.org/project/lockout/
