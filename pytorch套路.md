```python
# loss.backward()是求梯度的过程,可以通过手动来更新参数，而不用优化器来更新
# optimizer.step()只是使用loss.backward()得到的梯度进行更新参数
# 需要to(device) 只有model,训练集data，标签target

import torch
import torch.nn as nn # 各种层类型的实现
import torch.nn.functional as F
# 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等

# 是否可以用gpu
USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)
    
# 自定义模型类需要继承nn.Module，且你至少要重写__init__和forward两个函数

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们实例化两个nn.Linear模块并将它们指定为成员变量。
        """
        super(TwoLayerNet, self).__init__()
        # 初始化继承
        
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在forward函数中，我们接受输入数据的Tensor，我们必须返回Tensor的输出数据。
        我们可以使用构造函数中定义的模块以及Tensors上的任意（可区分）操作。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# 输入和输出
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# 创建模型
model = TwoLayerNet(D_in, H, D_out)
model = model.to(device)

# 构造我们的损失函数和优化器。 
#在SGD构造函数中对model.parameters（）的调用将包含作为模型成员的两个nn.Linear模块的可学习参数。
# 损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')
# 优化器不用 to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: 喂入数据并前向传播获取输出
    y_pred = model(x)

    # Compute and print loss
    # 调用损失函数计算损失
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    # 清除所有优化的梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 参数更新
    optimizer.step()
    
#测试时不用计算梯度
#with torch.no_grad(): 
# 禁用梯度计算
```




​    

# CNN-LeNet5
```python
import torch.nn as nn
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
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
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

```

    net = LeNet5()
    print(net)
    LeNet5(
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=256, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )

# 完整CNN  

## 定义CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1) 
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)
        #in_channels：输入图像通道数，手写数字图像为1，彩色图像为3
        #out_channels：输出通道数，这个等于卷积核的数量
        #kernel_size：卷积核大小
        #stride：步长
         
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        #上个卷积网络的out_channels，就是下一个网络的in_channels，所以这里是20
        #out_channels：卷积核数量50
        
        
        self.fc1 = nn.Linear(4*4*50, 500)
        #全连接层torch.nn.Linear(in_features, out_features)
        #in_features:输入特征维度，4*4*50是自己算出来的，跟输入图像维度有关
        #out_features；输出特征维度
        
        self.fc2 = nn.Linear(500, 10)
        #输出维度10，10分类

    def forward(self, x):  
        #print(x.shape)  #手写数字的输入维度，(N,1,28,28), N为batch_size
        x = F.relu(self.conv1(x)) # x = (N,50,24,24)
        x = F.max_pool2d(x, 2, 2) # x = (N,50,12,12)
        x = F.relu(self.conv2(x)) # x = (N,50,8,8)
        x = F.max_pool2d(x, 2, 2) # x = (N,50,4,4)
        x = x.view(-1, 4*4*50)    # x = (N,4*4*50)
        x = F.relu(self.fc1(x))   # x = (N,4*4*50)*(4*4*50, 500)=(N,500)
        x = self.fc2(x)           # x = (N,500)*(500, 10)=(N,10)
        return F.log_softmax(x, dim=1)  #带log的softmax分类，每张图片返回10个概率
```



NLL-loss的定义
$$
\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\}
$$

## 定义训练函数
```python
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
                100. * batch_idx / len(train_loader), #len(train_loader)=60000/32=1875
                loss.item()
            ))
            #print(len(train_loader))
```


## 定义测试函数

```python
def test(model, device, test_loader):
    model.eval() #进入测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) 
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # sum up batch loss
            #reduction='sum'代表batch的每个元素loss累加求和，默认是mean求平均
                       
            pred = output.argmax(dim=1, keepdim=True) 
            # get the index of the max log-probability
            
            #print(target.shape) #torch.Size([32])
            #print(pred.shape) #torch.Size([32, 1])
            correct += pred.eq(target.view_as(pred)).sum().item()
            #pred和target的维度不一样
            #pred.eq()相等返回1，不相等返回0，返回的tensor维度(32，1)。

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
## 训练和测试

```python
torch.manual_seed(53113)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = test_batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)


lr = 0.01
momentum = 0.5
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

epochs = 2
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

save_model = True
if (save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")
```

## 损失函数细节

cross_entropy输入的logits是未经过softmax层的输出。

而标签值为一个数字，而不是对应的one-hot向量。
$$
loss(x, class) = -log(\frac{exp(x[class])}{(\sum_j exp(x[j]))})
               = -x[class] + log(\sum_j exp(x[j]))
$$
nll_loss 输入的则是经过softmax和log后的输出

```
out=F.log_softmax(out,dim=1)
```

```python
torch.nn.CrossEntropyLoss
将输入经过 softmax 激活函数之后，再计算其与 target 的交叉熵损失。即该方法将
nn.LogSoftmax()和 nn.NLLLoss()进行了结合

输入的target是标签，而不能是对应的one-hot向量

torch.nn.NLLLoss
loss(input, class) = -input[class]。 举个例，三分类任务， 
input=[-1.233, 2.657, 0.534]， 真实标签为 2（class=2），则 loss 为-0.534
```



| torch.nn         | torch.nn.functional (F) |
| ---------------- | ----------------------- |
| CrossEntropyLoss | cross_entropy           |
| LogSoftmax       | log_softmax             |
| NLLLoss          | nll_loss                |



```python
x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)


torch_dataset = Data.TensorDataset(x, y) # y为target
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training每次训练打乱顺序
    num_workers=2,              # subprocesses for loading data
)
```

