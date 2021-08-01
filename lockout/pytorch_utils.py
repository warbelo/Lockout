# ==================================================================================================
# Author: Wilmer Arbelo-Gonzalez
# 
# This module provides general pytorch tools
# --------------------------------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler



# ==================================================================================================
class FCNN(nn.Module):
    def __init__(self, n_features, layer_sizes):
        """
        Input:
        - n_features:  number of input features (integer)
        - layer_sizes: nodes per layers (list of integers)

        Output:
        - Fully Connected Neural Network
        """
        super(FCNN, self).__init__()
        
        weight_dims   = [int(n_features)] + layer_sizes
        self.n_layers = len(layer_sizes)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(weight_dims[i], weight_dims[i+1]) for i in range(self.n_layers)]
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        for n, linear_layer in enumerate(self.linear_layers, start=1):
            x = linear_layer(x)
            if n != self.n_layers:
                x = self.relu(x)
        return x



# ==================================================================================================
def weight_reset(model):
    """
    Input:
    - model (torch model)

    Output:
    - reset model's learnable parameters (those with "reset_parameters" attribute)
    """
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()



# ==================================================================================================
def get_features_importance(model, layer_name, cols=None, tol=0):
    """
    Input:
    - model (torch model)
    - layer_name (str): layer where features importance is to be computed
    - cols (list or Pandas Index): feature names
    - tol (float): threshold above which features importance is returned
    
    Output:
    - importance_sorted (Pandas Series): features importance where indeces are the feature names
    """
    ww = model.state_dict()[layer_name]
    if cols is not None:
        importance = pd.Series(abs(ww).sum(dim=0), index=cols)
    else:
        importance = pd.Series(abs(ww).sum(dim=0))
    importance = importance/importance.max()
    importance_sorted = importance.sort_values(ascending=False)
    idx = importance_sorted > tol
    return importance_sorted[idx]



# ==================================================================================================
def save_model(model, name):
    """
    Input:
    - model (torch model) to be saved
    - name (str or path) of model when saved 
    """
    torch.save(model.state_dict(), name)


# ==================================================================================================
def load_data_reg(folder):
    """
    Input:
    - Name of the folder with the data (str or path)
    
    Output:
    (X, xtrain, xvalid, xtest, Y, ytrain, yvalid, ytest) read from CVS files
    and transformed to proper torch tensors for regression
    """
# Read X, xtrain, xvalid, and xtest (if given)
    X, xtrain, xvalid, xtest = None, None, None, None
    if os.path.exists(os.path.join(folder, 'X.csv')):
        df = pd.read_csv(os.path.join(folder, 'X.csv'), 
                        index_col=False, header=None)
        X = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'xtrain.csv')):
        df = pd.read_csv(os.path.join(folder, 'xtrain.csv'), 
                        index_col=False, header=None)
        xtrain = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'xvalid.csv')):
        df = pd.read_csv(os.path.join(folder, 'xvalid.csv'), 
                        index_col=False, header=None)
        xvalid = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'xtest.csv')):
        df = pd.read_csv(os.path.join(folder, 'xtest.csv'), 
                        index_col=False, header=None)
        xtest = torch.tensor(df.values).float()
    
# Read Y, ytrain, yvalid, and ytest (if given)
    Y, ytrain, yvalid, ytest = None, None, None, None
    if os.path.exists(os.path.join(folder, 'Y.csv')):
        df = pd.read_csv(os.path.join(folder, 'Y.csv'), 
                        index_col=False, header=None)
        Y = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'ytrain.csv')):
        df = pd.read_csv(os.path.join(folder, 'ytrain.csv'), 
                        index_col=False, header=None)
        ytrain = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'yvalid.csv')):
        df = pd.read_csv(os.path.join(folder, 'yvalid.csv'), 
                        index_col=False, header=None)
        yvalid = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'ytest.csv')):
        df = pd.read_csv(os.path.join(folder, 'ytest.csv'), 
                        index_col=False, header=None)
        ytest = torch.tensor(df.values).float()
    return X, xtrain, xvalid, xtest, Y, ytrain, yvalid, ytest


# ==================================================================================================
def load_data_clf(folder):
    """
    Input:
    - Name of the folder with the data (str or path)
    
    Output:
    (X, xtrain, xvalid, xtest, Y, ytrain, yvalid, ytest) read from CVS files
    and transformed to proper torch tensors for classification
    """
# Read X, xtrain, xvalid, and xtest (if given)
    X, xtrain, xvalid, xtest = None, None, None, None
    if os.path.exists(os.path.join(folder, 'X.csv')):
        df = pd.read_csv(os.path.join(folder, 'X.csv'), 
                        index_col=False, header=None)
        X = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'xtrain.csv')):
        df = pd.read_csv(os.path.join(folder, 'xtrain.csv'), 
                        index_col=False, header=None)
        xtrain = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'xvalid.csv')):
        df = pd.read_csv(os.path.join(folder, 'xvalid.csv'), 
                        index_col=False, header=None)
        xvalid = torch.tensor(df.values).float()
    if os.path.exists(os.path.join(folder, 'xtest.csv')):
        df = pd.read_csv(os.path.join(folder, 'xtest.csv'),
                        index_col=False, header=None)
        xtest = torch.tensor(df.values).float()
    
# Read Y, ytrain, yvalid, and ytest (if given)
    Y, ytrain, yvalid, ytest = None, None, None, None
    if os.path.exists(os.path.join(folder, 'Y.csv')):
        df = pd.read_csv(os.path.join(folder, 'Y.csv'), 
                        index_col=False, header=None)
        Y = torch.tensor(df.values).long()
    if os.path.exists(os.path.join(folder, 'ytrain.csv')):
        df = pd.read_csv(os.path.join(folder, 'ytrain.csv'), 
                        index_col=False, header=None)
        ytrain = torch.tensor(df.values).long()
    if os.path.exists(os.path.join(folder, 'yvalid.csv')):
        df = pd.read_csv(os.path.join(folder, 'yvalid.csv'), 
                        index_col=False, header=None)
        yvalid = torch.tensor(df.values).long()
    if os.path.exists(os.path.join(folder, 'ytest.csv')):
        df = pd.read_csv(os.path.join(folder, 'ytest.csv'), 
                        index_col=False, header=None)
        ytest = torch.tensor(df.values).long()
    return X, xtrain, xvalid, xtest, Y, ytrain, yvalid, ytest


# ==================================================================================================
def dataset_r2(data_loader, model, device, y_mean=None):
    """
    Input:
    - data_loader: torch DataLoader previously created
    - model:       torch model previously trained
    - device: 'gpu' or 'cpu'
    
    Output:
    - R2 score for the given data set
    - Mean of Y (dummy output for iterative use)
    """
# Put model in evaluation mode
    model.eval()
    ss_tot = 0.0
    ss_res = 0.0

# Loop over mini batches (no gradients need to be computed)
    with torch.no_grad():
    # .Find mean of y and number of observations
        if y_mean is None:
            y_mean = 0.0
            n_points = 0
            for i, (xx, yy, _) in enumerate(data_loader, start=1):
                batch_size = len(yy)
                y_mean = yy.sum()
                n_points += batch_size
            y_mean = y_mean/float(n_points)

    # .Find total sum of squares and residuals
        for i, (xx, yy, _) in enumerate(data_loader, start=1):
            xx = xx.to(device)
            yy = yy.to(device)
            yy_pred = model(xx)

            ss_tot += torch.square(yy - y_mean).sum()
            ss_res += torch.square(yy - yy_pred).sum()

        r2_score = 1.0 - (ss_res/ss_tot)
    return r2_score.item(), y_mean


# ==================================================================================================
def dataset_accuracy(data_loader, model, device):
    """
    Input:
    - data_loader: torch DataLoader previously created
    - model:       torch model previously trained
    - device: 'gpu' or 'cpu'
    
    Output:
    - Accuracy for the given data set
    """
# Put model in evaluation mode
    model.eval()
    correct_total = 0
    n_points = 0

# Loop over mini batches (no gradients need to be computed)
    with torch.no_grad():
        for i, (xx, yy, _) in enumerate(data_loader, start=1):
            batch_size = len(yy)
            xx = xx.to(device)
            yy = yy.to(device)
            yy_pred = model(xx)
        
# .compute correct predictions (target size must be [batch_size])
            correct = correct_predictions(yy_pred, yy.view(-1))
            correct_total += correct
        
# .number of points looped over after the ith mini-batch
            n_points += batch_size
    return correct_total.item()/n_points


# ==================================================================================================
def correct_predictions(y_output, y_target):
    """
    Input:
    - 2D torch tensor (y_output) of shape (batch_size, n_classes) resulting 
      from running the model
    - 1D torch tensor (y_target) of shape (batch_size) with the 
      corresponding class for each observation
      
    Output:
    - number of correct predictions (integer)
    """
    correct = 0
    prob_softmax = nn.Softmax(dim=1)
    _, predictions = torch.max(prob_softmax(y_output), dim=1)
    correct = (predictions == y_target).sum(dim=0)
    return correct


# ==================================================================================================
def valid_epoch_reg(data_loader, model, loss_type, device):
    """
    Input:
    - data_loader: torch DataLoader previously created
    - model:       torch model previously trained
    - loss_type:   loss function previously created/instantiated
    - device: 'gpu' or 'cpu'
    
    Output:
    - Mean loss function over the entire data set
    """
# Put model in evaluation mode
    model.eval()
    loss_fun = 0.
    n_points = 0

# Loop over mini batches (no gradients need to be computed)
    with torch.no_grad():
        for i, (xx, yy, _) in enumerate(data_loader, start=1):
            xx = xx.to(device)
            yy = yy.to(device)
            yy_pred = model(xx)
        
# .compute loss function
            batch_size = len(yy)
            loss = loss_type(yy_pred, yy)
            loss_fun += batch_size*loss.item()
        
# .number of points used after the ith mini-batch
            n_points += batch_size
    return loss_fun/n_points


# ==================================================================================================
def valid_epoch_clf(data_loader, model, loss_type, device):
    """
    Input:
    - data_loader: torch DataLoader previously created
    - model:       torch model previously trained
    - loss_type:   loss function previously created/instantiated
    - device: 'gpu' or 'cpu'
    
    Output:
    - Mean loss function over the entire data set
    """
# Put model in evaluation mode
    model.eval()
    loss_fun = 0.
    n_points = 0

# Loop over mini batches (no gradients need to be computed)
    with torch.no_grad():
        for i, (xx, yy, _) in enumerate(data_loader, start=1):
            xx = xx.to(device)
            yy = yy.to(device)
            yy_pred = model(xx)
        
# .compute loss function
            batch_size = len(yy)
            loss = loss_type(yy_pred, yy.view(-1))
            loss_fun += batch_size*loss.item()
        
# .number of points used after the ith mini-batch
            n_points += batch_size
    return loss_fun/n_points


# ==================================================================================================
class dataset_tabular(Dataset):
    '''
    Input:
    - X values (xtensor) as a 2D torch tensor with dimension 
      [# of points, # of features]
    - Y values (ytensor) as a 1D torch tensor with dimension 
      [# of points]
      
    Output:
    - Torch Dataset for tabular data with a single output
    '''
    def __init__(self, xtensor, ytensor):
        self.x = xtensor
        self.y = ytensor
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx], idx


# ==================================================================================================
def make_DataLoaders(xtrain, xvalid, xtest, ytrain, yvalid, ytest, 
                     batch_size=100000, num_workers = 0):
    """
    Input:
    -Torch tensors with xtrain, xvalid, xtest, ytrain, yvalid, ytest
    -Torch Dataset previously defined
    -batch size (integer)
    -Number of CPUs num_workers (integer)
    
    Output:
    -Torch DataLoaders for training, validation, and testing datasets. 
    """
# Create datasets
    train_dataset = dataset_tabular(xtrain, ytrain)
    valid_dataset = dataset_tabular(xvalid, yvalid)
    test_dataset  = dataset_tabular(xtest, ytest)
    
# Create Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size = batch_size,
                                                   shuffle = True, 
                                                   num_workers=num_workers)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                                   batch_size = batch_size,
                                                   shuffle = True, 
                                                   num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size = batch_size,
                                                   shuffle = False, 
                                                   num_workers=num_workers)
    return train_dataloader, valid_dataloader, test_dataloader


# ==================================================================================================
def sgn(x):
    """PyTorch sign function"""
    return torch.sign(x)


# ==================================================================================================
def normalize_x(xtrain, xvalid, xtest):
    """
    Input: 
    - Torch tensors with training, validation, and testing data sets (X only)
    
    Output:
    - Torch tensors with normalized training, validation, and testing 
    data sets: xtrain, xvalid, xtest
        X = (X - X_mean)/X_std
    """
    scaler = StandardScaler()
    scaler.fit(xtrain.numpy())
    x_train = torch.from_numpy(scaler.transform(xtrain.numpy()))
    x_valid = torch.from_numpy(scaler.transform(xvalid.numpy()))
    x_test = torch.from_numpy(scaler.transform(xtest.numpy()))
    return x_train, x_valid, x_test


# ==================================================================================================
def normalize_xy(xtrain, xvalid, xtest, ytrain, yvalid, ytest):
    """
    Input: 
    - Torch tensors with training, validation, and testing data sets 
      (X and Y)
    
    Output:
    - Torch tensors with normalized training, validation, and testing 
      data sets: xtrain, xvalid, xtest, ytrain, yvalid, ytest
        X = (X - X_mean)/X_std
        Y = (Y - Y_mean)/Y_std
    """
    scaler = StandardScaler()
    scaler.fit(xtrain.numpy())
    x_train = torch.from_numpy(scaler.transform(xtrain.numpy()))
    x_valid = torch.from_numpy(scaler.transform(xvalid.numpy()))
    x_test = torch.from_numpy(scaler.transform(xtest.numpy()))
#     
    scaler = StandardScaler()
    scaler.fit(ytrain.numpy())
    y_train = torch.from_numpy(scaler.transform(ytrain.numpy()))
    y_valid = torch.from_numpy(scaler.transform(yvalid.numpy()))
    y_test = torch.from_numpy(scaler.transform(ytest.numpy()))
    return x_train, x_valid, x_test, y_train, y_valid, y_test
