# ==================================================================================================
# Author: Wilmer Arbelo-Gonzalez
# 
# This module provides general functions/classes for Lockout
# --------------------------------------------------------------------------------------------------
import os
import copy
import tqdm
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tools_pytorch import valid_epoch_clf, valid_epoch_reg, dataset_accuracy, sgn, weight_reset,    \
                          dataset_r2


# ==================================================================================================
class Lockout():
    '''
    Input:
    -
    
    Output:
    -
    '''
# INITIALIZE LOCKOUT CLASS
    def __init__(self, input_model, lr=None, loss_type=2, tol_grads=1e-2, beta=0.7, optim_id=1, 
                 regul_type=None, regul_path=None, t0=None, t0_grid=None, t0_points=None, 
                 device=None, optim_params=None, save_weights = (False, None)):
        """
        - loss_type = 1: nn.MSELoss(reduction='mean')
        - loss_type = 2: nn.CrossEntropyLoss(reduction='mean')
        """
    # .Variables Initialization
        if lr is not None:
            self.__lr = torch.tensor(lr)
        else:
            raise TypeError("Learning rate (lr) is required")
        
        self.__input_model = input_model
        self.__model_layers = list(self.__input_model.state_dict().keys())

        if regul_type is not None:
            self.__regul_type = dict(regul_type)
            self.__layer_names = list(self.__regul_type.keys())
        else:
            self.__regul_type = regul_type
            self.__layer_names = None
        
        if regul_path is not None:
            self.__regul_path = dict(regul_path)
        else:
            self.__regul_path = regul_path
        
        self.__optim_id = optim_id
        if optim_params is not None:
            self.__optim_params = dict(optim_params)
        else:
            self.__optim_params = {}
        
        self.__loss_type = loss_type
        self.__tol_grads = tol_grads
        self.__beta = torch.tensor(beta)

        if device is not None:
            self.__device = device
        else:
            self.__device = torch.device('cpu')

        if t0 is not None:
            self.__t0_init = dict(t0)
            self.__t0_flag = True
        else:
            self.__t0_init = {}
            self.__t0_flag = False

        self.__t0_grid = t0_grid
        if t0_points is not None:
            self.__t0_points = dict(t0_points)
        else:
            self.__t0_points = t0_points

        self.__t0_final = {}
        self.__t0_interval = {}
        self.__path_t0 = {}
        self.__path_loss = {}
        self.__path_accuracy = {}
        self.__path_sparcity = {}
        self.__sign_w = {}
        
        self.weights_count = {}
        self.features_count = {}
        self.__save_weights = save_weights
        if save_weights[0]:
            self.weight_iters = pd.DataFrame()
        
    # .Some input variables check-ups...
        if self.__t0_flag:
            if len(self.__regul_type) != len(self.__t0_init):
                raise TypeError("length of 'regul_type' and 't0' do not match")

        if self.__regul_type is not None:
            for key in self.__regul_type:
                if key in self.__model_layers:
                    continue
                else:
                    raise TypeError("'{}' in 'regul_type' not in input model".format(key))

        if self.__regul_path is not None:
            for key in self.__regul_path:
                if key in self.__model_layers:
                    continue
                else:
                    raise TypeError("'{}' in 'regul_path' not in input model".format(key))

        if self.__t0_grid is not None:
            for key in self.__t0_grid:
                if key in self.__model_layers:
                    continue
                else:
                    raise TypeError("'{}' in 't0_grid' not in input model".format(key))

        if self.__t0_points is not None:
            for key in self.__t0_points:
                if key in self.__model_layers:
                    continue
                else:
                    raise TypeError("'{}' in 't0_points' not in input model".format(key))

        if self.__regul_path is not None:
            if len(self.__regul_type) != len(self.__regul_path):
                raise TypeError("Lengths of 'regul_type' and 'regul_path' do not match")

        if self.__regul_path is not None:
            if sorted(self.__regul_type.keys()) != sorted(self.__regul_path.keys()):
                raise TypeError("Layer names in 'regul_type' and 'regul_path' do not match")

    # .Variable Initialization (per layer)
        cols_sparsity = []
        cols_t0 = []
        if self.__regul_type is not None:
            for key in self.__layer_names:
                ww = self.__input_model.state_dict()[key].detach()

            # ..Weight Counts, Sparcity
                self.__sign_w[key] = sgn(ww)
                self.weights_count[key] = float(ww.numel())
                self.features_count[key] = float(ww.sum(dim=0).numel())
                sparcity = (ww.sum(dim=0) != 0.0).sum()/self.features_count[key]
                self.__path_sparcity[key] = sparcity.item()
                cols_sparsity.append('sparcity__'+key)
                cols_t0.append('t0_calc__'+key)
                cols_t0.append('t0_used__'+key)

    # .Set up loss function 
        if (self.__loss_type == 1):
            self.loss_func = nn.MSELoss(reduction='mean')
            self.__calc_loss = mean_squared_error_loss
            self.__valid_epoch = valid_epoch_reg
        elif (self.__loss_type == 2):
            self.loss_func = nn.CrossEntropyLoss(reduction='mean')
            self.__calc_loss = cross_entropy_loss
            self.__valid_epoch = valid_epoch_clf
        else:
            raise TypeError("loss type = '{}' is not implemented".format(self.__loss_type))
            
    # .Set up output DataFrame(s)
        cols_output = ['iteration'] + cols_sparsity + cols_t0
        cols_output += ['train_loss', 'valid_loss', 'test_loss', 'train_accu', 'valid_accu', 'test_accu']
        self.path_data = pd.DataFrame(columns=cols_output)
    


# TRAIN METHOD
    def train(self, dl_train, dl_valid, dl_test=None, epochs=10000, early_stop=20, tol_loss=1e-5, 
              epochs2=None, train_how='until_path', reset_weights = True):
        """
        """
    # .Variables Initialization
        self.__reset_weights = reset_weights
        self.__early_stop = early_stop
        self.__tol_loss  = tol_loss
        self.path_data = pd.DataFrame(columns=self.path_data.columns)
        epochs_path1 = epochs
        if epochs2 is not None:
            epochs_path2 = epochs2
        else:
            epochs_path2 = epochs

    # .Train Model: Decreasing t0
        if train_how == 'decrease_t0':
            if self.__t0_flag:
                raise TypeError("'t0' is NOT required for current 'train_how' option")
            
            if self.__t0_grid is not None:
                raise TypeError("'t0_grid' is NOT required for current 'train_how' option")
            
            if self.__t0_points is not None:
                raise TypeError("'t0_points' is NOT required for current 'train_how' option")
            
            if self.__regul_path is None:
                raise TypeError("Missing input variable 'regul_path'")
            
            if self.__regul_type is None:
                raise TypeError("Missing input variable 'regul_type'")
            
            for key in self.__layer_names:
                ww = self.__input_model.state_dict()[key].detach()
                t0_tmp, _ = get_constraint(ww.flatten(), 
                                           reg_type=self.__regul_type[key], 
                                           beta=self.__beta)
                self.__t0_init[key] = t0_tmp
                self.__path_t0[key] = t0_tmp
                if self.__regul_path[key]:
                    self.__t0_final[key] = self.__get_t0_final(key)
                    self.__t0_interval[key] = self.__t0_final[key] - self.__t0_init[key]

            self.__optim_id = 1
            self.__optim_params = {}
            self.__train_path1(dl_train, dl_valid, dl_test, epochs_path1)
            self.path_data = self.path_data.append(self.__temp_data, ignore_index=True)[:-1]
            print("Path 1: Early stopping = {}".format(self.__early_stop_flag))
            print("        Last iteration = {}".format(self.__last_iter))
            
            self.__train_path2(dl_train, dl_valid, dl_test, epochs_path2)
            self.path_data = self.path_data.append(self.__temp_data, ignore_index=True)
            print("Best validation at iteration = {}".format(self.__best_iter_valid))
            self.model_last = copy.deepcopy(self.__model)
            self.model_best_valid = copy.deepcopy(self.__model_valid_min)

    # .Train Model: Constant t0
        elif train_how == 'constant_t0':
            if self.__t0_flag == False:
                raise TypeError("Missing input variable 't0'")
            
            if self.__regul_type is None:
                raise TypeError("Missing input variable 'regul_type'")
            
            if self.__regul_path is not None:
                raise TypeError("'regul_path' is NOT required for current 'train_how' option")
            
            if self.__t0_grid is not None:
                raise TypeError("'t0_grid' is NOT required for current 'train_how' option")
            
            if self.__t0_points is not None:
                raise TypeError("'t0_points' is NOT required for current 'train_how' option")
            
            for key in self.__layer_names:
                self.__path_t0[key] = self.__t0_init[key]
            
            self.__optim_id = 1
            self.__optim_params = {}
            self.__train_path1(dl_train, dl_valid, dl_test, epochs_path1)
            self.path_data = self.path_data.append(self.__temp_data, ignore_index=True)
            print("Early stopping = {}".format(self.__early_stop_flag))
            print("Last iteration = {}".format(self.__last_iter))
            print("Best validation at iteration = {}".format(self.__best_iter_valid))
            self.model_last = copy.deepcopy(self.__model)
            self.model_best_valid = copy.deepcopy(self.__model_valid_min)

    # .Train Model: Until it Hits The Path
        elif train_how == 'until_path':
            if self.__regul_type is None:
                raise TypeError("Missing input variable 'regul_type'")
            
            if self.__t0_flag:
                raise TypeError("'t0' is NOT required for current 'train_how' option")
            
            if self.__regul_path is not None:
                raise TypeError("'regul_path' is NOT required for current 'train_how' option")
            
            if self.__t0_grid is not None:
                raise TypeError("'t0_grid' is NOT required for current 'train_how' option")
            
            if self.__t0_points is not None:
                raise TypeError("'t0_points' is NOT required for current 'train_how' option")
            
            for key in self.__layer_names:
                ww = self.__input_model.state_dict()[key].detach()
                t0_tmp, _ = get_constraint(ww.flatten(), 
                                           reg_type=self.__regul_type[key], 
                                           beta=self.__beta)
                self.__t0_init[key] = t0_tmp
                self.__path_t0[key] = t0_tmp
            
            self.__optim_id = 1
            self.__optim_params = {}
            self.__train_path1(dl_train, dl_valid, dl_test, epochs_path1)
            self.path_data = self.path_data.append(self.__temp_data, ignore_index=True)
            print("Early stopping = {}".format(self.__early_stop_flag))
            print("Last iteration = {}".format(self.__last_iter))
            print("Best validation at iteration = {}".format(self.__best_iter_valid))
            self.model_last = copy.deepcopy(self.__model)
            self.model_best_valid = copy.deepcopy(self.__model_valid_min)

    # .Train Model: Sampling discrete set of t0
        elif train_how == 'sampling_t0':
            if self.__t0_flag:
                raise TypeError("'t0' is NOT required for current 'train_how' option")
            
            if self.__regul_type is None:
                raise TypeError("Missing input variable 'regul_type'")
            
            if self.__regul_path is None:
                raise TypeError("Missing input variable 'regul_path'")
            
            if self.__t0_grid is not None and self.__t0_points is not None:
                raise TypeError("'t0_grid' and 't0_points' are mutually exclusive for current 'train_how' option")
            
            self.__t0_grid_dict = {}
            for key in self.__layer_names:
                ww = self.__input_model.state_dict()[key].detach()
                t0_tmp, _ = get_constraint(ww.flatten(), 
                                           reg_type=self.__regul_type[key], 
                                           beta=self.__beta)
                self.__t0_init[key] = t0_tmp
                self.__path_t0[key] = t0_tmp
                if self.__regul_path[key]:
                    self.__t0_final[key] = self.__get_t0_final(key)

        # ..When 't0_points' is given
            if self.__t0_points is not None:
                tmp = pd.Series(self.__t0_points).value_counts()
                t0_size = tmp.index[0]
                if len(tmp) != 1:
                    raise TypeError("'t0_points': Only same number of points per layer is allowed")
                if tmp.index[0] == 1:
                    raise TypeError("'t0_points': One point per layer is nor allowd. Use a different train_how option instead")
                tmp = pd.Series(self.__regul_path).value_counts()
                if tmp[True] != len(self.__t0_points):
                    raise TypeError("'t0_points': Number of layers to be sampled do not match with t0_points keys")

                for key in self.__layer_names:
                    if self.__regul_path[key]:
                        if key in self.__t0_points:
                            t0_tmp = np.linspace(self.__t0_init[key], 
                                                 self.__t0_final[key], 
                                                 num=self.__t0_points[key], 
                                                 endpoint=True)
                            self.__t0_grid_dict[key] = torch.from_numpy(t0_tmp)
                        else:
                            raise KeyError("'t0_points': Only layers where regul_path = True are supposed to be sampled")

        # ..When 't0_grid' is given
            if self.__t0_grid is not None:
                dct_tmp = {}
                for key in self.__t0_grid:
                    dct_tmp[key] = len(self.__t0_grid[key])
                tmp = pd.Series(dct_tmp).value_counts()
                t0_size = tmp.index[0]
                if len(tmp) != 1:
                    raise TypeError("'t0_grid': Only same number of points per layer is allowed")
                if tmp.index[0] == 1:
                    raise TypeError("'t0_grid': One point per layer is nor allowd. Use a different train_how option instead")
                tmp = pd.Series(self.__regul_path).value_counts()
                if tmp[True] != len(self.__t0_grid):
                    raise TypeError("'t0_grid': Number of layers to be sampled do not match with t0_points keys")

                for key in self.__layer_names:
                    if self.__regul_path[key]:
                        if key in self.__t0_grid:
                            self.__t0_grid_dict[key] = self.__t0_grid[key]
                            self.__path_t0[key] = self.__t0_grid_dict[key][0]
                        else:
                            raise KeyError("'t0_grid': Only layers where regul_path = True are supposed to be sampled")

        # ..Loop over t0 grid
            input_model_flag = True
            self.__optim_id = 1
            self.__optim_params = {}
            iterator = tqdm.notebook.tqdm(range(1, t0_size + 1), desc='t0')
            for n in iterator:
                for key in self.__layer_names:
                    if self.__regul_path[key]:
                        self.__t0_init[key] = self.__t0_grid_dict[key][n-1]
                self.__train_path1(dl_train, dl_valid, dl_test, epochs_path1, use_input_model=input_model_flag)
                self.path_data = self.path_data.append(self.__temp_data.tail(1), ignore_index=True)
                input_model_flag = False
                print("Early stopping = {} ({})".format(self.__early_stop_flag, self.__last_iter))
                self.model_last = copy.deepcopy(self.__model)
                self.model_best_valid = copy.deepcopy(self.__model_valid_min)
            print("Best validation at iteration = {}".format(self.__best_iter_valid))

    # .Standard Training With Early Stop On Validation Loss
        elif train_how == 'unconstraint':
            if self.__t0_flag:
                warnings.warn("'t0' is not required for current 'train_how' option")
            
            if self.__regul_type is not None:
                warnings.warn("'regul_type' is not required for current 'train_how' option")

            if self.__regul_path is not None:
                warnings.warn("'regul_path' is not required for current 'train_how' option")

            if self.__t0_grid is not None:
                warnings.warn("'t0_grid' is not required for current 'train_how' option")
            
            if self.__t0_points is not None:
                warnings.warn("'t0_points' is not required for current 'train_how' option")
            
            self.__train_forward(dl_train, dl_valid, dl_test, epochs_path1)
            self.path_data = self.path_data.append(self.__temp_data, ignore_index=True)
            print("Early stopping = {}".format(self.__early_stop_flag))
            print("Last iteration = {}".format(self.__last_iter))
            print("Best validation at iteration = {}".format(self.__best_iter_valid))
            self.model_last = copy.deepcopy(self.__model)
            self.model_best_valid = copy.deepcopy(self.__model_valid_min)

        else:
            raise TypeError("train_how = '{}' is not valid".format(train_how))
        


# TRAIN PATH 1
    def __train_path1(self, dl_train, dl_valid, dl_test, epochs, use_input_model=True):
        """
        """
        n_iterations = epochs*len(dl_train)
        if n_iterations < 1:
            TypeError("Number of iterations below threshold. Increase number of epochs")

    # .Variables Initialization
        if use_input_model:
            self.__model = copy.deepcopy(self.__input_model)
        self.__t0 = {}
        self.__t0.update(self.__t0_init)
        self.__iteration = 1
        self.__early_stop_count = 0
        self.__early_stop_flag  = False
        self.__temp_data = pd.DataFrame(columns=self.path_data.columns)
        self.__best_iter_valid = 0
        self.__best_loss_valid = np.inf
        self.__ymean = {'train': None, 'valid': None, 'test': None}

    # .Set up optimizer
        optimizer = self.__setup_optimizer()

    # .Loop Over Number of Epochs
        iteration = 0
        iterator = tqdm.notebook.tqdm(range(1, epochs+1), desc='Epochs1')
        for n_epoch in iterator:

        # ..Loop Over Mini-batches
            for ibatch, (xx, yy_true, _) in enumerate(dl_train, start=1):

            # ...Compute Validation Loss (over entire epoch)
                self.__path_loss['valid_loss'] = self.__valid_epoch(dl_valid, 
                                                 self.__model, 
                                                 self.loss_func, 
                                                 self.__device)
                
            # ...Compute Test Loss (over entire epoch)
                if dl_test is not None:
                    self.__path_loss['test_loss'] = self.__valid_epoch(dl_test, 
                                                    self.__model, 
                                                    self.loss_func, 
                                                    self.__device)
                
            # ...Compute Train Loss (over minibatch)
                self.__model.train()
                xx = xx.to(self.__device)
                yy_true = yy_true.to(self.__device)
                yy_pred = self.__model(xx)
                loss = self.__calc_loss(yy_pred, yy_true, self.loss_func)
                self.__path_loss['train_loss'] = loss.item()

            # ...Store Output data
                self.__store_output_data(dl_train, dl_valid, dl_test, 
                                         self.__model, self.__regul_type)

            # ...Check For Validation Minimum
                if self.__path_loss['valid_loss'] < self.__best_loss_valid:
                    self.__model_valid_min = copy.deepcopy(self.__model)
                    self.__best_loss_valid = self.__path_loss['valid_loss']
                    self.__best_iter_valid = self.__iteration

            # ...Check For Convergence (Train)
                self.__check_convergence(iteration, self.__t0)
                if self.__early_stop_flag:
                    break

            # ...Compute Gradients
                optimizer.zero_grad()
                loss.backward()

            # ...Modify Gradients (Using Lockout)
                for name, weight in self.__model.named_parameters():
                    if name in self.__layer_names:
                        weight.grad = self.__lockout_grad_update(
                                           weight.detach(), 
                                           weight.grad.detach(), 
                                           self.__t0[name], 
                                           self.__regul_type[name]
                                      )
                
            # ...Update Weights 
                optimizer.step()

            # ...Set Weights That Change Sign to Zero
                for name, weight in self.__model.named_parameters():
                    if name in self.__layer_names:
                        sign_w_next = sgn(weight.detach())
                        mask0 = ((sign_w_next != self.__sign_w[name]) & (self.__sign_w[name] != 0.0))
                        with torch.no_grad():
                            weight[mask0] = torch.tensor(0.0)
                        self.__sign_w[name] = sign_w_next

            # ...Compute Sparcity
                self.__get_sparsity()

            # ...Compute Constraint Values
                self.__get_constraint_values()

                iteration += 1
                self.__iteration += 1

        # ..Stop Training (By Early Stopping) 
            if self.__early_stop_flag:
                self.__last_iter = self.__iteration
                break
            else:
                self.__last_iter = self.__iteration - 1
        


# TRAIN PATH 2
    def __train_path2(self, dl_train, dl_valid, dl_test, epochs):
        """
        """
        n_iterations = epochs*len(dl_train)
        if n_iterations < 1:
            TypeError("Number of iterations below threshold. Increase number of epochs")
        n_iterations -= 1

    # .Variables Initialization
        self.__temp_data = pd.DataFrame(columns=self.path_data.columns)

    # .Set up optimizer
        optimizer = self.__setup_optimizer()

    # .Loop Over Number of Epochs
        iteration = 0
        iterator = tqdm.notebook.tqdm(range(1, epochs+1), desc='Epochs2')
        for n_epoch in iterator:

        # ..Loop Over Mini-batches
            for ibatch, (xx, yy_true, _) in enumerate(dl_train, start=1):

            # ...Compute Validation Loss (over entire epoch)
                self.__path_loss['valid_loss'] = self.__valid_epoch(dl_valid, 
                                                 self.__model, 
                                                 self.loss_func, 
                                                 self.__device)
                
            # ...Compute Test Loss (over entire epoch)
                if dl_test is not None:
                    self.__path_loss['test_loss'] = self.__valid_epoch(dl_test, 
                                                    self.__model, 
                                                    self.loss_func, 
                                                    self.__device)
                
            # ...Compute Train Loss (over minibatch)
                self.__model.train()
                xx = xx.to(self.__device)
                yy_true = yy_true.to(self.__device)
                yy_pred = self.__model(xx)
                loss = self.__calc_loss(yy_pred, yy_true, self.loss_func)
                self.__path_loss['train_loss'] = loss.item()

            # ...Store Output data
                self.__store_output_data(dl_train, dl_valid, dl_test, 
                                         self.__model, self.__regul_type)

            # ...Check For Validation Minimum
                if self.__path_loss['valid_loss'] < self.__best_loss_valid:
                    self.__model_valid_min = copy.deepcopy(self.__model)
                    self.__best_loss_valid = self.__path_loss['valid_loss']
                    self.__best_iter_valid = iteration

            # ...Compute Gradients
                optimizer.zero_grad()
                loss.backward()

            # ...Modify Gradients (Using Lockout)
                for name, weight in self.__model.named_parameters():
                    if name in self.__layer_names:
                        weight.grad = self.__lockout_grad_update(
                                           weight.detach(), 
                                           weight.grad.detach(), 
                                           self.__t0[name], 
                                           self.__regul_type[name]
                                      )
                
            # ...Update Weights 
                optimizer.step()

            # ...Set Weights That Change Sign to Zero
                for name, weight in self.__model.named_parameters():
                    if name in self.__layer_names:
                        sign_w_next = sgn(weight.detach())
                        mask0 = ((sign_w_next != self.__sign_w[name]) & (self.__sign_w[name] != 0.0))
                        with torch.no_grad():
                            weight[mask0] = torch.tensor(0.0)
                        self.__sign_w[name] = sign_w_next

            # ...Compute Sparcity
                self.__get_sparsity()

            # ...Compute Constraint Values
                self.__get_constraint_values()

            # ...Update (reduce) t0
                self.__update_t0(iteration, n_iterations)

                iteration += 1
                self.__iteration += 1
        
        

# TRAIN FORWARD
    def __train_forward(self, dl_train, dl_valid, dl_test, epochs):
        """
        """
        n_iterations = epochs*len(dl_train)
        if n_iterations < 1:
            TypeError("Number of iterations below threshold. Increase number of epochs")

    # .Variables Initialization
        self.__model = copy.deepcopy(self.__input_model)
        if self.__reset_weights:
            self.__model.apply(weight_reset)
        self.__iteration = 1
        self.__early_stop_count = 0
        self.__early_stop_flag  = False
        self.__temp_data = pd.DataFrame(columns=self.path_data.columns)
        self.__best_iter_valid = 0
        self.__best_loss_valid = np.inf
        self.__t0 = None
        self.__ymean = {'train': None, 'valid':None, 'test':None}

    # .Set up optimizer (...write method to set up optim...)
        optimizer = self.__setup_optimizer()

    # .Loop Over Number of Epochs
        iteration = 0
        iterator = tqdm.notebook.tqdm(range(1, epochs+1), desc='Epochs')
        for n_epoch in iterator:
    # ..Loop Over Mini-batches
            for ibatch, (xx, yy_true, _) in enumerate(dl_train, start=1):
    # ...Compute Validation Loss (over entire epoch)
                self.__path_loss['valid_loss'] = self.__valid_epoch(dl_valid, 
                                                 self.__model, 
                                                 self.loss_func, 
                                                 self.__device)
                
    # ...Compute Train Loss (over minibatch)
                self.__model.train()
                xx = xx.to(self.__device)
                yy_true = yy_true.to(self.__device)
                yy_pred = self.__model(xx)
                loss = self.__calc_loss(yy_pred, yy_true, self.loss_func)
                self.__path_loss['train_loss'] = loss.item()

    # ...Store Output data
                self.__store_output_data(dl_train, dl_valid, dl_test, 
                                         self.__model, self.__regul_type)

    # ...Check For Validation Minimum
                if self.__path_loss['valid_loss'] < self.__best_loss_valid:
                    self.__model_valid_min = copy.deepcopy(self.__model)
                    self.__best_loss_valid = self.__path_loss['valid_loss']
                    self.__best_iter_valid = self.__iteration

    # ...Check For Convergence (Train)
                self.__check_convergence(iteration, self.__t0)
                if self.__early_stop_flag:
                    break

    # ...Compute Gradients
                optimizer.zero_grad()
                loss.backward()

    # ...Update Weights 
                optimizer.step()

                iteration += 1
                self.__iteration += 1

    # ..Stop Training (By Early Stopping) 
            if self.__early_stop_flag:
                self.__last_iter = self.__iteration
                break
            else:
                self.__last_iter = self.__iteration - 1



# SET UP OPTIMIZER
    def __setup_optimizer(self):
        """
        Input:
        - 
        
        Output:
        - 
        """
    # .Set up Stochastic Gradient Descend (SGD)
        if self.__optim_id == 1:
            opt_dict = {'momentum': 0, 
                        'dampening': 0, 
                        'weight_decay': 0, 
                        'nesterov': False}
            for key in self.__optim_params:
                if key in opt_dict:
                    opt_dict[key] = self.__optim_params.get(key)
                else:
                    warnings.warn("'{}' not an input parameter of selected optimizer".format(key))
            optimizer = optim.SGD(self.__model.parameters(), lr=self.__lr, **opt_dict)

    # .Set up Adam
        elif self.__optim_id == 2:
            opt_dict = {'betas': (0.9, 0.999), 
                        'eps': 1e-8, 
                        'weight_decay': 0, 
                        'amsgrad': False}
            for key in self.__optim_params:
                if key in opt_dict:
                    opt_dict[key] = self.__optim_params.get(key)
                else:
                    warnings.warn("'{}' not an input parameter of selected optimizer".format(key))
            optimizer = optim.Adam(self.__model.parameters(), lr=self.__lr, **opt_dict)
        else:
            raise TypeError("Optimizer optim_id = {} is not implemented".format(self.__optim_id))
        return optimizer



# UPDATE t0
    def __update_t0(self, it, n_it):
        """
        Input:
        - it (int): current iteration
        - n_it (int): total number of iterations in the run
        
        Output:
        - Updated t0 values per layer
        """
        for key in self.__layer_names:
            if self.__regul_path[key]:
                self.__t0[key] = self.__t0_init[key] + self.__t0_interval[key]*it/float(n_it)



# GRADIENTS UPDATE WITH LOCKOUT
    def __lockout_grad_update(self, weights, grads, t0, reg_type):
        """
        """
    # ..Compute P(w) and dP(w)/dw
        w_shape = weights.size()
        gr2d = grads.detach()
        w1d = torch.flatten(weights)
        Pw, p1d = get_constraint(w1d, reg_type=reg_type, beta=self.__beta)
        
    # ..Compute g=-grads, 'gamma', and sort gamma (in descending order)
        g1d = -torch.flatten(gr2d)
        gamma = abs(g1d)/(p1d + 1e-12)
        _, indx1d = torch.sort(gamma, descending=True)
        
    # ...Modify Gradients Accordingly:
        grmin = torch.zeros(w_shape).fill_(self.__tol_grads).to(self.__device)
        grads = sgn(gr2d)*torch.max(abs(gr2d), grmin)
        gr1d  = torch.flatten(grads.detach())
        pjsj  = self.__lr*abs(gr1d[indx1d])
        
    # ...Left side: sign(g) != sign(w) elements
        mask_ds = (sgn(g1d[indx1d]) != sgn(w1d[indx1d])) & (w1d[indx1d] != 0.0)
        DS_sum = pjsj[mask_ds].sum()
        pjsj[mask_ds] = 0.0
        mask_dsc = ~mask_ds
        indx1d_dsc = indx1d[mask_dsc]
        left_side = torch.cumsum(pjsj, dim=0) - pjsj
        
    # ...Right side
        pjsj_dsc = torch.zeros(len(g1d)).to(self.__device)
        pjsj_dsc[:] = pjsj[:]
        mask_w0 = (w1d[indx1d] == 0.0)
        pjsj_dsc[mask_w0] = 0.0
        right_side = pjsj_dsc.sum() - torch.cumsum(pjsj_dsc, dim=0) + DS_sum - Pw + t0
        
    # ...Difference
        ds = right_side - left_side
        mask_w0 = (~mask_w0).float()
        pjsj_new = -mask_w0**(1-sgn(ds))*sgn(g1d[indx1d])*sgn(ds)*torch.min(pjsj, abs(ds))
        
    # ...Modify Gradients
        indx2d = np.unravel_index(indx1d_dsc.cpu(), shape=w_shape)
        grads[indx2d] = pjsj_new[mask_dsc]/self.__lr
    # 
        return grads


        
# STORE LOCKOUT OUTPUT DATA
    def __store_output_data(self, dl_train, dl_valid, dl_test, model, layers):
        """
        """
        dict_tmp = {'iteration': int(self.__iteration)}
        dict_tmp.update(self.__path_loss)
        if self.__regul_type is not None:
            for key in layers:
                dict_tmp['t0_calc__'+key] = self.__path_t0[key].item()
                dict_tmp['t0_used__'+key] = self.__t0[key].item()
                dict_tmp['sparcity__'+key] = self.__path_sparcity[key]
        if self.__loss_type == 1:
            self.__path_accuracy['train_accu'], self.__ymean['train'] = dataset_r2(dl_train, model, 
                                                                                   self.__device, 
                                                                                   self.__ymean['train'])
            self.__path_accuracy['valid_accu'], self.__ymean['valid'] = dataset_r2(dl_valid, model, 
                                                                                   self.__device, 
                                                                                   self.__ymean['valid'])
            if dl_test is not None:
                self.__path_accuracy['test_accu'], self.__ymean['test'] = dataset_r2(dl_test, model, 
                                                                                     self.__device, 
                                                                                     self.__ymean['test'])
            dict_tmp.update(self.__path_accuracy)
        elif self.__loss_type == 2:
            self.__path_accuracy['train_accu'] = dataset_accuracy(dl_train, model, self.__device)
            self.__path_accuracy['valid_accu'] = dataset_accuracy(dl_valid, model, self.__device)
            if dl_test is not None:
                self.__path_accuracy['test_accu'] = dataset_accuracy(dl_test, model, self.__device)
            dict_tmp.update(self.__path_accuracy)
        df_tmp = pd.DataFrame(dict_tmp, index=[0])
        self.__temp_data = self.__temp_data.append(df_tmp, ignore_index=True)

        if self.__save_weights[0]:
            wtmp = self.__model.state_dict()[self.__save_weights[1]].detach().view(-1).numpy()
            self.weight_iters = self.weight_iters.append(pd.Series(wtmp), ignore_index=True)



# COMPUTE CONSTRAINT (t0)
    def __get_constraint_values(self):
        """
        Output:
        - updated constraint values 'self.__path_t0'.
        """
        for key in self.__layer_names:
            ww = self.__model.state_dict()[key].detach()
            t0_tmp, _ = get_constraint(ww.flatten(), 
                                       reg_type=self.__regul_type[key], 
                                       beta=self.__beta)
            self.__path_t0[key] = t0_tmp



# COMPUTE SPARSITY
    def __get_sparsity(self):
        """
        Output:
        - updated sparcity 'self.__path_sparcity'.

                   # of non zero features
        sparcity = ----------------------
                   total # of features
        """
        for key in self.__layer_names:
            wfeatures = self.__model.state_dict()[key].detach().sum(dim=0)
            sparcity = (wfeatures != 0.0).sum()/self.features_count[key]
            self.__path_sparcity[key] = sparcity.item()



# CHECK CONVERGENCE (TRAIN)
    def __check_convergence(self, iteration, t0_tmp):
        """
        """
        train_loss_change = abs(self.__temp_data.loc[iteration, 'train_loss'] - 
                                self.__temp_data.loc[max(0, iteration-1), 'train_loss'])
        conv_flg = True
        if t0_tmp is not None:
            for key in self.__layer_names:
                if abs(self.__path_t0[key] - t0_tmp[key]) < 1e-3*abs(self.__t0_init[key]):
                    continue
                else:
                    conv_flg = False
                    break
        
        if train_loss_change < self.__tol_loss and conv_flg:
            self.__early_stop_count += 1
        else:
            self.__early_stop_count = 0
        if self.__early_stop_count >= self.__early_stop:
            self.__early_stop_flag = True



# COMPUTE t0 FINAL
    def __get_t0_final(self, key):
        """
        Input:
        - key (str): layer name
        
        Output:
        - Final (minimum) value of the constraint (t0_final) along the path
        reg_type=1  =>  0.0
        reg_type=2  =>  N_weights * log(beta)
        """
        if self.__regul_type[key] == 1:
            t0_min = torch.tensor(0.0)
        elif self.__regul_type[key] == 2:
            t0_min = self.weights_count[key]*torch.log(self.__beta)
        return t0_min



# ==================================================================================================
def get_constraint(w, reg_type=1, beta=0.7):
    """
    Input:
    - w (1D torch tensor): weights
    - reg_type (integer): id for the type of regularization
    - beta (float): beta for regularization corresponding to reg_type=2
    
    Output:
    - Values of the regularization constraint (reg_value) as a tensor
      reg_type=1  =>  sum( abs(w_i) ) (lasso)
      reg_type=2  =>  sum( log((1 - beta)*abs(w_i) + beta) )
      
    - Values of the derivative of the regularization constraint with respect
      to w (reg_deriv) as a 1D tensor
      reg_type=1  =>  1.0
      reg_type=2  =>  (1 - beta)/((1 - beta)*abs(w_i) + beta)
    """
    if reg_type == 1:
        reg_value = abs(w).sum()
        reg_deriv = torch.ones(len(w))
    elif reg_type == 2:
        reg_value = torch.log((1.0 - beta)*abs(w) + beta).sum()
        reg_deriv = (1.0 - beta)/((1.0 - beta)*abs(w) + beta)
    else:
        raise TypeError("constraint type = {} is not implemented".format(reg_type))
    return reg_value, reg_deriv



# ==================================================================================================
def mean_squared_error_loss(ypred, ytrue, loss):
    """
    Torch nn.MSELoss(reduction='mean')
    """
    return loss(ypred, ytrue)



# ==================================================================================================
def cross_entropy_loss(ypred, ytrue, loss):
    """
    Torch nn.CrossEntropyLoss(reduction='mean')
    """
    return loss(ypred, ytrue.view(-1))


