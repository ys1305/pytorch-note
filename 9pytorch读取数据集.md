



```python
# 自定义 Dataset
class MyMNIST(Dataset):
    def __init__(self, csv_file: str, train=False, transform=None):
        self.train = train
        self.transform = transform
        if self.train:
            train_df = pd.read_csv(csv_file)
            self.train_labels = train_df.iloc[:, 0].values
            self.train_data = train_df.iloc[:, 1:].values.reshape((-1, 28, 28))
        else:
            test_df = pd.read_csv(csv_file)
            self.test_data = test_df.values.reshape((-1, 28, 28))
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def __getitem__(self, index):
        if self.train:
            image, label = self.train_data[index], self.train_labels[index]
        else:
            image = self.test_data[index]
        image = Image.fromarray(image.astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        if self.train:
            return image, label
        else:
            return image
```

# 普通神经网络代码

这里需要除以255,而CNN不需要,因为ToTensor()会自动除以255

```python
import os
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
class MyData():
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = pd.read_csv(csv_file,header=None)
        self.xdata = self.csv_data.iloc[:,1:].values/255.
        self.ydata = self.csv_data.iloc[:,0].values

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
    	x = self.xdata[idx]
    	y = self.ydata[idx]

    	return x,y

        # return self.xdata[idx],self.ydata[idx]
# 

# initialize the paths to our training and testing CSV files
TRAIN_CSV = "../data/Mnist/mnist_train.csv"
TEST_CSV = "../data/Mnist/mnist_test.csv"

# initialize the number of epochs to train for and batch size
batch_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 1
learning_rate = 0.001

# initialize both the training and testing image generators
train_loader = DataLoader(dataset=MyData(TRAIN_CSV),
                        batch_size=batch_size, 
                       shuffle=True)
test_loader = DataLoader(dataset=MyData(TEST_CSV), 
                        batch_size=batch_size, 
                       shuffle=True)
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.type(torch.FloatTensor)
        # print(images.max()) 最大值为1

        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.type(torch.FloatTensor)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

```





# CNN完整代码

```python
import torch
import pandas as pd 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers

import matplotlib.pyplot as plt

train_df = pd.read_csv('./data/train.csv')

n_train = len(train_df)
n_pixels = len(train_df.columns) - 1
n_class = len(set(train_df['label']))

# test_df = pd.read_csv('./data/test.csv')

# print(train_df.iloc[:,1:].values.mean(axis=1).mean())

class MNIST_data(Dataset):
    """MNIST dtaa set"""
    # transforms.ToTensor()自动除以255
    
    def __init__(self, file_path, 
                 transform = transforms.Compose([transforms.ToTensor(), 
                     transforms.Normalize(mean=(0.5,), std=(0.5,))])
                ):
        
        df = pd.read_csv(file_path)
        
        if len(df.columns) == n_pixels:
            # test data
            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = None
        else:
            # training data
            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = torch.from_numpy(df.iloc[:,0].values)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])

batch_size = 32

train_dataset = MNIST_data('./data/train.csv', transform= transforms.Compose(
                            [transforms.ToPILImage(), transforms.RandomRotation(degrees=20), 
                            # RandomShift(3),
                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
# test_dataset = MNIST_data('./data/test.csv')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                            batch_size=batch_size, shuffle=False)


# print(next(iter(train_loader))[0].max())
# print(next(iter(train_loader))[0].min())
# tensor(1.)
# tensor(-1.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # 这里论文上写的是conv,官方教程用了线性层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train() #进入训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #梯度归零
        output = model(data)  #输出的维度[N,10] 这里的data是函数的forward参数x
        loss = F.nll_loss(output, target) #这里loss求的是平均数，除以了batch
#F.nll_loss(F.log_softmax(input), target) ：
#单分类交叉熵损失函数，一张图片里只能有一个类别，输入input的需要softmax
#还有一种是多分类损失函数，一张图片有多个类别，输入的input需要sigmoid
        
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}".format(
                epoch, 
                batch_idx * len(data), #100*32
                len(train_loader.dataset), #60000
                100. * batch_idx / len(train_loader), #len(train_loader)=60000/32=1875 计算的不是准确率，而是已训练数据的比例
                loss.item()
            ))
            #print(len(train_loader))

lr = 0.01
momentum = 0.5
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

epochs = 1
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)

# Train Epoch: 1 [0/42000 (0.000000%)]    Loss: 2.316952
# Train Epoch: 1 [3200/42000 (7.616146%)] Loss: 2.290542
# Train Epoch: 1 [6400/42000 (15.232292%)]    Loss: 2.284746
# Train Epoch: 1 [9600/42000 (22.848439%)]    Loss: 2.109172
# Train Epoch: 1 [12800/42000 (30.464585%)]   Loss: 1.203029
# Train Epoch: 1 [16000/42000 (38.080731%)]   Loss: 0.748803
# Train Epoch: 1 [19200/42000 (45.696877%)]   Loss: 0.574634
# Train Epoch: 1 [22400/42000 (53.313024%)]   Loss: 0.399714
# Train Epoch: 1 [25600/42000 (60.929170%)]   Loss: 0.317008
# Train Epoch: 1 [28800/42000 (68.545316%)]   Loss: 0.168170
# Train Epoch: 1 [32000/42000 (76.161462%)]   Loss: 0.153755
# Train Epoch: 1 [35200/42000 (83.777609%)]   Loss: 0.332306
# Train Epoch: 1 [38400/42000 (91.393755%)]   Loss: 0.197081
# Train Epoch: 1 [41600/42000 (99.009901%)]   Loss: 0.208421
```

