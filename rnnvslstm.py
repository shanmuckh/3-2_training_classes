import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def generate_data(samples=2000, seq_len=100):
    X = np.random.rand(samples, seq_len, 1)
    y = np.random.randint(0, 2, size=samples)

    X[:, 0, 0] = y  # store label only at first timestep
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

X, y = generate_data()

train_X, test_X = X[:1500], X[1500:]
train_y, test_y = y[:1500], y[1500:]
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 16, batch_first=True)  
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
def train(model, name):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(25):
        opt.zero_grad()
        out = model(train_X)
        loss = loss_fn(out, train_y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(test_X).argmax(dim=1)
        acc = (preds == test_y).float().mean()
        print(f"{name} Accuracy: {acc.item()*100:.2f}%")

train(RNNModel(), "RNN")
train(LSTMModel(), "LSTM")

