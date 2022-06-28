import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
import sys  
import time
import random
import sklearn
from tqdm import tqdm

from loss_functions import get_WFL
from rehearsal_utils import memory_update,rehearsal_memory_resize

class BasePlugin:

    def __init__(self, mem_size, device):
        super().__init__()
        self.mem_size = mem_size
        self.x_tasks_memory = [] # It shold be a matrix
        self.y_tasks_memory = [] # It should be a matrix
        self.x_memory = []
        self.y_memory = []
        self.device = device

    def before_training(self, dataset):
        if len(self.x_memory) == 0:
            train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
        else:
            train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
        return train_dataloader
        

    def train_step_create_batch_data(self, batch_data, memory_batch_size):
        return batch_data[0], batch_data[1]

    def add_data_to_memory(self, train_dataloader, dist_fn):
       # do nothing
       pass

    def get_tasks_in_memory(self):
        return []

class RehearsalPlugin(BasePlugin):
    def __init__(self, mem_size, device):
        super().__init__(mem_size, device)
        self.tasks_in_mem = []
        self.task_number = 0

        self.dataset_count = None
        self.num_samples_seen = 0
        self.num_labels_seen = 0
        self.rho=0

    def _get_sample_from_memory(self, memory_batch_size):
        if len(self.x_memory) >= memory_batch_size:
            sample_indexes = np.random.randint(len(self.x_memory), size = memory_batch_size, dtype=int)
            x_memory_selected = torch.from_numpy(np.array(self.x_memory)[sample_indexes]).to(self.device)
            y_memory_selected = torch.from_numpy(np.array(self.y_memory)[sample_indexes]).to(self.device)
        else:
            x_memory_selected = None
            y_memory_selected = None
        return x_memory_selected, y_memory_selected

    def train_step_create_batch_data(self, batch_data, memory_batch_size):
        x_memory_sampled, y_memory_sampled = self._get_sample_from_memory(memory_batch_size)
        sample = batch_data[0] if x_memory_sampled is None else torch.cat((batch_data[0], x_memory_sampled), 0)
        y_true = batch_data[1].type(torch.float32) if y_memory_sampled is None else torch.cat((batch_data[1].type(torch.float32), y_memory_sampled.type(torch.float32)), 0)
        sample, y_true = sklearn.utils.shuffle(sample, y_true)
        return sample, y_true

    def update_target_distribution(self,y_batch):
        if self.target_count is None:
            num_labels = y_batch.shape[1]
            self.target_count = np.zeros(num_labels)

        self.dataset_count = self.dataset_count + y_batch.sum(axis=1)
        self.num_samples_seen+=y_batch.shape[0]
        self.num_labels_seen+=y_batch.sum()
        self.target_distribution = (self.dataset_count** self.rho)/self.num_labels_seen 
        self.target_count = self.target_distribution*self.num_samples_seen



class OCDM_Plugin(RehearsalPlugin):
    def __init__(self, mem_size, device,rho=0):
        super().__init__(mem_size, device)

    def add_data_to_memory(self, train_dataloader, dist_fn):
        prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataloader.dataset)/train_dataloader.batch_size))
        for _, batch_data in prog_bar:
            x_batch = batch_data[0].cpu().detach().numpy()
            y_batch = batch_data[1].cpu().detach().numpy()
            
            self.update_target_distribution(y_batch)

            memory_update(self.target_distribution,x_batch, y_batch, self.x_memory,  self.y_memory, dist_fn, self.mem_size, self.tasks_in_mem, self.task_number)
        self.task_number += 1
        
    def get_tasks_distribution_in_memory(self):
        values, counts = np.unique(self.tasks_in_mem, return_counts=True)
        tasks_count = np.zeros(14)
        for i in range(len(values)):
            tasks_count[values[i]] = counts[i]
        return tasks_count


# From a memory of mem_size, the new samples of a task are added to memory by resizing dinamycally the number of samples to keep in memory for each task.
class BAT_OCDM_Plugin(RehearsalPlugin):

    def __init__(self, mem_size, device):
        super().__init__(mem_size, device)

    def add_data_to_memory(self, train_dataloader, dist_fn):
        prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataloader.dataset)/train_dataloader.batch_size))
        
        self.num_tasks += 1 # New task to add, the counter is incremented
        mem_size = round(self.mem_size / self.num_tasks)


        # Create and update memory of new task
        self.x_tasks_memory.append([]) # Appending an empty list
        self.y_tasks_memory.append([])
        for _, batch_data in prog_bar:
            x_batch = batch_data[0].cpu().detach().numpy()
            y_batch = batch_data[1].cpu().detach().numpy()

            self.update_target_distribution(y_batch)

            memory_update(self.target_distribution,x_batch, y_batch, self.x_tasks_memory[-1], self.y_tasks_memory[-1], dist_fn, mem_size)

        # Resizing the old memories
        for task in range(len(self.x_tasks_memory)):
            new_x_memory, new_y_memory = rehearsal_memory_resize(self.target_distribution,self.x_tasks_memory[task], self.y_tasks_memory[task], dist_fn, mem_size)
            self.x_tasks_memory[task] = new_x_memory
            self.y_tasks_memory[task] = new_y_memory
        
        self.x_memory = []
        self.y_memory = []
        for task in range(len(self.x_tasks_memory)):
            for index in range(len(self.x_tasks_memory[task])):
                self.x_memory.append(self.x_tasks_memory[task][index])
                self.y_memory.append(self.y_tasks_memory[task][index])


        print(f"Number of memories {len(self.x_tasks_memory)}")
        for i in range(len(self.x_tasks_memory)):
            print(f"Mem {i} has lenght of {len(self.x_tasks_memory[i])}")
        print(f"Total memory length is {len(self.x_memory)}")

    def get_tasks_distribution_in_memory(self):
        tasks_count = np.zeros(14)
        for i in range(len(self.x_tasks_memory)):
            tasks_count[i] = len(self.x_tasks_memory[i])
        return tasks_count


class ReservoirSampling_Plugin(RehearsalPlugin):

    def __init__(self, mem_size, device):
        super().__init__(mem_size, device)
        self.num_tasks = 0
        self.num_data = 0

    def add_data_to_memory(self, train_dataloader, dist_fn):
        #num_data = len(train_dataloader.dataset)
        x_task_data_full = []
        y_task_data_full = []
        prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataloader.dataset)/train_dataloader.batch_size))
        for _, batch_data in prog_bar:
            x_batch = batch_data[0].cpu().detach().numpy()
            y_batch = batch_data[1].cpu().detach().numpy()
            for i in range(len(x_batch)):
                x_task_data_full.append(x_batch[i])
                y_task_data_full.append(y_batch[i])
        prev_num_data = self.num_data
        self.num_data += len(x_task_data_full)
        for i in range(0, self.num_data):
            if i < prev_num_data:
                continue
            dataset_index = i - prev_num_data
            
            if len(self.x_memory) <= self.mem_size:
                
                self.x_memory.append(x_task_data_full[dataset_index])
                self.y_memory.append(y_task_data_full[dataset_index])
                self.tasks_in_mem.append(self.num_tasks)
            else:
                j = random.randint(0, i)
                if j < self.mem_size:
                    #print(f"dataset_size: {len(x_task_data_full)}   index: {dataset_index}    i: {i}  prev_num: {prev_num_data}  num_data: {self.num_data}")
                    self.x_memory[j] = x_task_data_full[dataset_index]
                    self.y_memory[j] = y_task_data_full[dataset_index]
                    self.tasks_in_mem[j] = self.num_tasks
        del x_task_data_full
        del y_task_data_full

        self.num_tasks += 1 # New task to add, the counter is incremented

    def get_tasks_in_memory(self):
        tasks_count = np.zeros(14)
        unique, counts = np.unique(self.tasks_in_mem, return_counts=True)
        dict_counts = dict(zip(unique, counts))
        for key in dict_counts:
            tasks_count[key] = dict_counts[key]
        return tasks_count



# Same as MultiTaskSingleDynamicMemoryPlugin but keeping a proportional memory size based on the number of samples of a task
class task_based_random_Plugin(RehearsalPlugin):

    def __init__(self, mem_size, device):
        super().__init__(mem_size, device)
        self.num_tasks = 0

    def add_data_to_memory(self, train_dataloader, dist_fn):
        prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataloader.dataset)/train_dataloader.batch_size))
        
        self.num_tasks += 1 # New task to add, the counter is incremented
        mem_size = round(self.mem_size / self.num_tasks)

        # Resizing the old memories
        for task in range(len(self.x_tasks_memory)):
            sample_indexes = np.random.randint(len(self.x_tasks_memory[task]), size = mem_size, dtype=int)
            new_x_memory = self.x_tasks_memory[task][sample_indexes]
            new_y_memory = self.y_tasks_memory[task][sample_indexes]

            self.x_tasks_memory[task] = new_x_memory
            self.y_tasks_memory[task] = new_y_memory

        self.x_tasks_memory.append([]) # Appending an empty list
        self.y_tasks_memory.append([])
        x_task_data_full = []
        y_task_data_full = []

        for _, batch_data in prog_bar:
            x_batch = batch_data[0].cpu().detach().numpy()
            y_batch = batch_data[1].cpu().detach().numpy()
            for i in range(len(x_batch)):
                x_task_data_full.append(x_batch[i])
                y_task_data_full.append(y_batch[i])
        
        sample_indexes = np.random.randint(len(x_task_data_full), size = mem_size, dtype=int)
        self.x_tasks_memory[-1] = np.array(x_task_data_full)[sample_indexes]
        self.y_tasks_memory[-1] = np.array(y_task_data_full)[sample_indexes]
        del x_task_data_full
        del y_task_data_full
        
        self.x_memory = []
        self.y_memory = []
        for task in range(len(self.x_tasks_memory)):
            for index in range(len(self.x_tasks_memory[task])):
                self.x_memory.append(self.x_tasks_memory[task][index])
                self.y_memory.append(self.y_tasks_memory[task][index])
        
        print(f"Number of memories {len(self.x_tasks_memory)}")
        for i in range(len(self.x_tasks_memory)):
            print(f"Mem {i} has lenght of {len(self.x_tasks_memory[i])}")
        print(f"Total memory length is {len(self.x_memory)}")

    def get_tasks_in_memory(self):
        tasks_count = np.zeros(14)
        for i in range(len(self.x_tasks_memory)):
            tasks_count[i] = len(self.x_tasks_memory[i])
        return tasks_count

