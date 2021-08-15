#Importing Libraries
import torch
import torchvision
import numpy as np
from tqdm import tqdm_notebook
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

#Importing Data and making Dataloaders 
train_data = datasets.STL10(root='data', split='train', transform=transforms.Lambda(lambda y:transforms.ToTensor()(np.array(y))), download=True)
test_data = datasets.STL10(root='data', split='test', transform=transforms.Lambda(lambda y:transforms.ToTensor()(np.array(y))), download=True)
unlabeled_data = datasets.STL10(root='data', split='unlabeled', transform=transforms.Lambda(lambda y:transforms.ToTensor()(np.array(y))), download=True)

train_dataloader = DataLoader(train_data, batch_size = 256, shuffle=True)
test_Dataloader = DataLoader(test_data, batch_size=64)
unlabeled_dataloadet = DataLoader(unlabeled_data, batch_size=64, shuffle = True)

#Initializing Hyperparameters
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001

#Alpha Rate function and variables
T1 = 100
T2 = 700
af = 3

def alpha_weight(step):
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
         return ((step-T1) / (T2-T1))*af
      
#Creating the basic Block For Residual Deep Learing Network   
class Basic_Block(nn.Module):
    def __init__(self, channels, outchannels, stride = 1, downsample = None):
        super(Basic_Block, self).__init__()

        self.Conv1 = nn.Conv2d(channels, outchannels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()
        self.Conv2 = nn.Conv2d(outchannels, outchannels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.downsample = downsample
        self.stride =stride

    def forward(self, x):

        identity = x

        out = self.Conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.Conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
      
      
#Creating a class for Residual Deep Learning Network
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self._make_layer(block, 64, layers[0])
        self.layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.FC = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None

        if stride!=1 or self.inplanes!=planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride),
                nn.BatchNorm2d(planes)
            )

        layers =[]
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.FC(out)

        return out

#Defining the Model with ResNet using ResBlock and the layer
model = ResNet(Basic_Block, [4, 6, 6, 2], 10).to(device)

#Creating the loss function aka Criterion using CrossEntropyLoss and the Optimization Function -Optimizer using Adam Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Training the model with only supervised learning
for epoch in range(epochs):
  for i , (images, labels) in enumerate(train_dataloader):
    images = images.to(device)
    labels = labels.to(device)

    preds = model(images)
    loss = criterion(preds, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  if (epoch+1)%10 ==0 :
    print("Epoch ", (epoch+1), " Done!")
    
#Setting up parameters for semi-supervised learning    
epochs = 100
step = 100

#Training the model on the semi-supervised learning algorithm
model.train()

for epoch in tqdm_notebook(range(epochs)):
  for batch, (x_unlabeled, y_random) in enumerate(unlabeled_dataloadet):

    
    x_unlabeled = x_unlabeled.to(device)

    model.eval()
    pseudo_labels = torch.argmax(model(x_unlabeled), dim=-1)
    

    model.train()

    output = model(x_unlabeled)
    unlabeled_loss = criterion(output, pseudo_labels)

    optimizer.zero_grad()
    unlabeled_loss.backward()
    optimizer.step()

    if batch %50 == 0 :
      
      for batch_idx, (im, la) in enumerate(train_dataloader):
        im = im.to(device)
        la = la.to(device)

        op = model(im)
        lab_loss = criterion(op, la)

        optimizer.zero_grad()
        lab_loss.backward()
        optimizer.step()
    step+=1
    
# Testing the model    
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_Dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    
