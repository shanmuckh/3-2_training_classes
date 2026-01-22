import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = trainset.classes

images, labels = next(iter(trainloader))
plt.figure(figsize=(6,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i].permute(1,2,0))
    plt.title(classes[labels[i]])
    plt.axis("off")
plt.show()

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64*32*32,10)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Linear(64*8*8,10)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//r),
            nn.ReLU(),
            nn.Linear(channels//r, channels),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,_,_ = x.size()
        y = x.mean(dim=(2,3))
        y = self.fc(y).view(b,c,1,1)
        return x * y

class CNN_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,32,3,padding=1)
        self.se = SEBlock(32)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32*16*16,10)

    def forward(self,x):
        x = torch.relu(self.conv(x))
        x = self.se(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

def train_model(model, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        for x,y in trainloader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x),y)
            loss.backward()
            optimizer.step()

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in testloader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total*100

basic = BasicCNN()
train_model(basic)
acc_basic = evaluate(basic)

improved = ImprovedCNN()
train_model(improved)
acc_improved = evaluate(improved)

attention = CNN_SE()
train_model(attention)
acc_attention = evaluate(attention)

print("Basic CNN Accuracy:", acc_basic)
print("Improved CNN Accuracy:", acc_improved)
print("CNN with Attention Accuracy:", acc_attention)

with torch.no_grad():
    fmap = basic.features[0](images.to(device))

plt.figure(figsize=(8,8))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(fmap[0,i].cpu(), cmap="gray")
    plt.axis("off")
plt.show()

mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32*14*14,10)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

mnist_model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)

for _ in range(3):
    for x,y in mnist_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(mnist_model(x),y)
        loss.backward()
        optimizer.step()

print("MNIST training completed")
