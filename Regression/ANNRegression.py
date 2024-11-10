import torch
import torch.nn as nn

class ANNRegression(nn.Module):
    def __init__(self, input_size, hidden_neurons):
        super(ANNRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_neurons)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_neurons, 1)
        
    def predict(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def train_(self, device, train_loader, test_loader, optimizer, criterion, num_epochs):
        test_losses = []
        
        for epoch in range(num_epochs):
            self.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.predict(data)
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx == train_loader.dataset.__len__():
                    print(
                        f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]',
                        f'Loss: {loss.item():.6f}'
                    )
                    
            self.eval()
            test_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.predict(data)
                    test_loss += criterion(output, target).item()
                    
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)