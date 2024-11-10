import torch
import numpy as np

class DataLoader:
    def __init__(self, data, targets, batch_size=32, shuffle=True):
        if isinstance(data, np.ndarray):
            self.data = torch.FloatTensor(data)
        else:
            self.data = data
            
        if isinstance(targets, np.ndarray):
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = targets
            
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)
        
    def __iter__(self):
        self.index = 0

        self.indices = list(range(self.n_samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

        return self
    
    def __next__(self):
        if self.index >= self.n_samples:
            raise StopIteration
            
        # Get indices for current batch
        batch_indices = self.indices[self.index:min(self.index + self.batch_size, self.n_samples)]
        
        # Get data and targets for current batch
        batch_data = self.data[batch_indices]
        batch_targets = self.targets[batch_indices]
        
        self.index += self.batch_size
        
        return batch_data, batch_targets
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    @property
    def dataset(self):
        return self.data