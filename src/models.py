import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import numpy as np
import warnings
import sys  
import time
import random
import sklearn
from tqdm import tqdm

from loss_functions import get_WFL


class ASC_BR_Multilabel_Rehearsal(nn.Module):
    def __init__(self, Tx, alarm_num_in, plugin, rateo = 0.5, alarm_num_out=None, hparams=None):
        super().__init__()
        '''Reduced baseline model: feed-forward NN without sequential layers'''
        if alarm_num_out is None:
            alarm_num_out = alarm_num_in
        if 'hp_activation' in hparams.keys():
            activation = hparams['hp_activation']
        else:
            warnings.warn('Warning! Using default value for activation: tanh')
            activation = 'tanh'
        if hparams is not None and 'hp_dropout_rate' in hparams.keys():
            dropout_rate = hparams['hp_dropout_rate']
        else:
            warnings.warn('Warning! Using default value for dropout_rate : 0.5')
            dropout_rate = 0.5

        self.alarm_num_in = alarm_num_in
            
        self.net = nn.Sequential(
                        nn.Linear(alarm_num_in, 512),
                        get_activation(activation),
                        nn.Dropout(dropout_rate),
                        nn.Linear(512, 256),
                        get_activation(activation),
                        nn.Dropout(dropout_rate),
                        nn.Linear(256, 128),
                        get_activation(activation),
                        nn.Dropout(dropout_rate),
                        nn.Linear(128, alarm_num_out),
                        nn.Sigmoid())
        
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = get_WFL(0.8, 2.0)
        self.rateo = rateo
        self.plugin = plugin
    
    def forward(self, X):
        output = self.net(X)
        return torch.squeeze(output, 1)

    def train_step(self, batch_data, memory_batch_size):
        sample, y_true = self.plugin.train_step_create_batch_data(batch_data, memory_batch_size)
       
        
        # Forward step
        
        #y_pred = self.net(sample)
        y_pred = torch.squeeze(self.net(sample), 1)
        # Loss estimation
        loss = self.criterion(y_pred, y_true)
        loss = torch.mean(loss)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        
        # Backward step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().cpu().numpy()

    def fit(self, epochs, train_dataset, batch_size, dist_fn, device):
        new_data_batch_size = int(batch_size * self.rateo)
        old_data_batch_size = int(batch_size * (1 - self.rateo))
        
        #if len(self.plugin.x_memory) == 0:
        #    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        #else:
        #    train_dataloader = DataLoader(train_dataset, batch_size=new_data_batch_size, shuffle=True, num_workers=0)
        times = {}
        start = time.time()
        train_dataloader = self.plugin.before_training(train_dataset)
        end = time.time()
        times["before_training_time"] = abs(end - start)
        
        history = {}
        start = time.time()
        for epoch in range(epochs):
            print(f"EPOCH: {epoch + 1}/{epochs}")
            epoch_start = time.time()
            
            # Training step
            self.net.train()
            prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataloader.dataset)/train_dataloader.batch_size))
            loss_epoch = []
            for _, batch_data in prog_bar:
                loss = self.train_step(batch_data, old_data_batch_size)
                loss_epoch.append(loss)
            loss_epoch = np.mean(loss_epoch)
            if 'loss' not in history.keys():
                history['loss'] = []
            history['loss'].append(loss_epoch)

            epoch_end = time.time()

            
            print(f"Time: {print_time_taken(epoch_start, epoch_end)} | loss: {loss_epoch:.4f}\n")


        end = time.time()
        times["train_time"] = abs(end - start)
        print(f"Training time: {print_time_taken(start, end)}") 

        start = time.time()
        self.plugin.add_data_to_memory(train_dataloader, dist_fn)
        end = time.time()
        times["add_data_to_memory_time"] = abs(end - start)
        
        return history, times

    def summary(self):
        summary(self.net, (self.alarm_num_in,))
