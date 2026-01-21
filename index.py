import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#load data(this is coppied btw it is like a famous dataset so it comes with code for it hahah)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)



class Model(nn.Module):
    #nn module tracks all your weights and biases
    #simple to move from cpu to gpu
    #buidl in thingggggssssf wouhhhjh
    def __init__(self, start = 784, h1 = 128, h2 = 64, out = 10):
        super().__init__()
        self.fow1 = nn.Linear(start, h1) #nn.linear does the w times input + bias thingie auto
        self.fow2 = nn.Linear(h1, h2)
        self.fow3 = nn.Linear(h2, out)

    def forward(self, x):
        x = F.relu(self.fow1(x)) #aplly out butifull linear activation function
        x= F.relu(self.fow2(x))
        x= self.fow3(x)

        return x
model = Model()
mistakes= nn.CrossEntropyLoss() #returns losss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #gives adam aacceess to the weights and bisases and tells him how much to mose each time

loop= 31

for i in range(loop):
    model.train()
    for x, y in train_loader:
        x = x.view(x.size(0), -1)  # changes images to  2D tensors
        optimizer.zero_grad()      # clear old gradients
        out = model(x)             # forward pass
        loss = mistakes(out, y)    # compute loss
        loss.backward()            # backpropegation
        optimizer.step()           # update weights and biases

model.eval()
correct = 0
total = 0
#same hting but we are testing now
with torch.no_grad(): #i dont frickin know why we do this but its impoortant cuz we aint training we aint updating weights we just testing
    for x, y in test_loader:
        x = x.view(x.size(0), -1)
        out = model(x)
        predicted = torch.max(out, 1).indices
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(100 * correct / total)
