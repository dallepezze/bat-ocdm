import numpy as np


        
def labels_count(data_sample):
    count_alarms = np.array([0] * 15)
    for seq in data_sample:
        count_alarms[np.where(seq == 1)] += 1
    return count_alarms
        
        
def label_distribution(data_sample, rho = 1):
    count_alarms = labels_count(data_sample)
    count_alarms = count_alarms ** rho
    denom_sum = np.sum(count_alarms)
    return count_alarms / denom_sum


def min_index(target_distribution,data, dist_fn):
    min_dist = float("inf")
    min_index = -1
    prob_counts = labels_count(data)
    for j in range(len(data)):
        new_counts = prob_counts - data[j]
        prob_distribution = new_counts / np.sum(new_counts)
        
        dist = dist_fn(prob_distribution, target_distribution)
        if dist < min_dist:
            min_dist = dist
            min_index = j
    return min_index


# Applies the OCDM (Optimizing Class Distribution in Memory) algorithm from this paper: https://openreview.net/forum?id=HavXnq6KyT3
def memory_update(target_distribution,x_batch, y_batch, x_memory, y_memory, dist_fn, mem_size = 500, tasks_in_mem = None, task_number = None):
    batch_size = len(y_batch)
    remaining_size = mem_size - len(x_memory)
    if tasks_in_mem is not None and task_number is None:
        raise ValueError("task_number must not be None when task_in_mem is not None")

    if remaining_size > 0:
        size_sample = min(remaining_size, batch_size)
        sample_indexes = np.random.randint(len(y_batch), size = size_sample)
        new_X_data = x_batch[sample_indexes]
        new_y_data = y_batch[sample_indexes]
        for i in range(len(new_X_data)):
            x_memory.append(new_X_data[i])
            y_memory.append(new_y_data[i])
            if tasks_in_mem is not None:
                tasks_in_mem.append(task_number)
    else:
        x_omega = np.concatenate((x_memory, x_batch), axis=0)
        y_omega = np.concatenate((y_memory, y_batch), axis=0)

        for k in range(batch_size): 
            i = min_index(target_distribution,y_omega, dist_fn) #O(M + bt - k)
            
            x_omega = np.delete(x_omega, i, axis=0)
            y_omega = np.delete(y_omega, i, axis=0)
            if tasks_in_mem is not None:
                if i < mem_size:
                    tasks_in_mem[i] = task_number
        for i in range(mem_size):
            x_memory[i] = x_omega[i]
            y_memory[i] = y_omega[i]


def rehearsal_memory_resize(target_distribution,x_memory, y_memory, dist_fn, new_mem_size = 500):
    x_memory_np = np.array(x_memory)
    y_memory_np = np.array(y_memory)

    items_to_remove = len(x_memory) - new_mem_size
    for k in range(items_to_remove):
        i = min_index(target_distribution,y_memory_np, dist_fn)
        x_memory_np = np.delete(x_memory_np, i, axis=0)
        y_memory_np = np.delete(y_memory_np, i, axis=0)
    x_memory = []
    y_memory = []
    for i in range(len(x_memory_np)):
        x_memory.append(x_memory_np[i])
        y_memory.append(y_memory_np[i])
    return x_memory, y_memory
    



