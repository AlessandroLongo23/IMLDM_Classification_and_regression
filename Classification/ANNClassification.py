import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class ANNClassification(nn.Module):
    def __init__(self, input_size, hidden_neurons, silent=True):
        super(ANNClassification, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_neurons)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_neurons, 2)
        self.softmax = nn.Softmax(dim=1)
        self.train_losses = []
        self.silent = silent

    def predict(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    
    def train_(self, device, train_loader, optimizer, criterion, num_epochs):
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0 

            for data, target in train_loader:
                data, target = data.to(device), target.to(device).long().view(-1)

                optimizer.zero_grad()
                output = self.predict(data)
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_epoch_loss)

            if not self.silent:
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.6f}")

        if not self.silent:
            self.plot_training_loss()

    def eval_(self, device, val_loader, criterion, silent=True):
        self.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).long().view(-1)
                output = self.predict(data)

                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_loss /= len(val_loader.dataset)
        val_error = 1 - correct / total
        if not silent:
            print(f"Validation Loss: {val_loss:.6f}, Validation Error: {val_error:.2f}")

        return val_loss, val_error

    def plot_training_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()