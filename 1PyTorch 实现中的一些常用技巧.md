模型统计数据（Model Statistics)
------------------------

### 统计参数总数量

```python
num_params  =  sum(param.numel()  for  param in  model.parameters())
```

参数正则化（Weight Regularization）
----------------------------

### 以前的方法

#### L2/L1 Regularization

机器学习中几乎都可以看到损失函数后面会添加一个额外项，常用的额外项一般有两种，称作**_L1正则化_**和**_L2正则化_**，或者**_L1范数_**和**_L2范数_**。

L1 正则化和 L2 正则化可以看做是损失函数的惩罚项。所谓 “惩罚” 是指对损失函数中的某些参数做一些限制。

*   L1 正则化是指权值向量 w中各个元素的**_绝对值之和_**，通常表示为 ${||w||}_1$
*   L2 正则化是指权值向量 w中各个元素的**_平方和然后再求平方根_**，通常表示为{||w||}_2$

下面是L1正则化和L2正则化的作用，这些表述可以在很多文章中找到。

*   L1 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
*   L2 正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合

L2 正则化的实现方法：  

```python
reg = 1e-6
l2_loss = Variable(torch.FloatTensor(1), requires_grad=True)
for name, param in model.named_parameters():
    if \'bias\' not in name:
        l2_loss = l2_loss   (0.5 * reg * torch.sum(torch.pow(W, 2)))
```

L1 正则化的实现方法：  

```python
reg = 1e-6
l1_loss = Variable(torch.FloatTensor(1), requires_grad=True)
for name, param in model.named_parameters():
    if \'bias\' not in name:
        l1_loss = l1_loss   (reg * torch.sum(torch.abs(W)))
```



#### Orthogonal Regularization

```python
reg = 1e-6
orth_loss = Variable(torch.FloatTensor(1), requires_grad=True)
for name, param in model.named_parameters():
    if \'bias\' not in name:
        param_flat = param.view(param.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= Variable(torch.eye(param_flat.shape[0]))
        orth_loss = orth_loss   (reg * sym.sum())
```



#### Max Norm Constraint

简单来讲就是对 w 的指直接进行限制。  

```python
def max_norm(model, max_val=3, eps=1e-8):
    for name, param in model.named_parameters():
        if \'bias\' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps   norm))
```

### L2正则

在pytorch中进行L2正则化，最直接的方式可以直接用优化器自带的weight_decay选项指定权值衰减率，相当于L2正则化中的λ

```
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9,weight_decay=1e-5) 
```

```python
lambda = torch.tensor(1.) 
l2_reg = torch.tensor(0.) 
for param in model.parameters():     
	l2_reg += torch.norm(param) 
loss += lambda * l2_reg 
```



此外，优化器还支持一种称之为Per-parameter options的操作，就是对每一个参数进行特定的指定，以满足更为细致的要求。做法也很简单，与上面不同的，我们传入的待优化变量不是一个Variable而是一个可迭代的字典，字典中必须有params的key，用于指定待优化变量，而其他的key需要匹配优化器本身的参数设置。

```python
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```



```python
weight_p, bias_p = [],[]
for name, p in model.named_parameters():
  if 'bias' in name:
     bias_p += [p]
   else:
     weight_p += [p]
# 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias，
# 因此可以通过名字判断属性，这个和tensorflow不同，tensorflow是可以用户自己定义名字的，当然也会系统自己定义。
optim.SGD([
          {'params': weight_p, 'weight_decay':1e-5},
          {'params': bias_p, 'weight_decay':0}
          ], lr=1e-2, momentum=0.9)
```

### L1正则化

```python
criterion= nn.CrossEntropyLoss()

classify_loss = criterion(input=out, target=batch_train_label)

lambda = torch.tensor(1.)
l1_reg = torch.tensor(0.)
for param in model.parameters():
    l1_reg += torch.sum(torch.abs(param))

loss =classify_loss+ lambda * l1_reg
```



### 定义正则化类

```python
# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cuda'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))
 
 
class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")
```

#### 正则化类的使用

```python
# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))
 
weight_decay=100.0 # 正则化参数
 
model = my_net().to(device)
# 初始化正则化
if weight_decay>0:
   reg_loss=Regularization(model, weight_decay, p=2).to(device)
else:
   print("no regularization")
 
 
criterion= nn.CrossEntropyLoss().to(device) # CrossEntropyLoss=softmax+cross entropy
optimizer = optim.Adam(model.parameters(),lr=learning_rate)#不需要指定参数weight_decay
 
# train
batch_train_data=...
batch_train_label=...
 
out = model(batch_train_data)
 
# loss and regularization
loss = criterion(input=out, target=batch_train_label)
if weight_decay > 0:
   loss = loss + reg_loss(model)
total_loss = loss.item()
 
# backprop
optimizer.zero_grad()#清除当前所有的累积梯度
total_loss.backward()
optimizer.step()
```



### **学习率衰减**

torch.optim.lr_scheduler 

#### 根据迭代次数

当epoch每过stop_size时,学习率都变为初始学习率的gamma倍

```python
optimizer = optim.SGD(params=model.parameters(), lr=0.05)

# lr_scheduler.StepLR()
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
plt.figure()
x = list(range(100))
y = []
for epoch in range(100):
    scheduler.step()
    lr = scheduler.get_lr()
    print(epoch, scheduler.get_lr()[0])
    y.append(scheduler.get_lr()[0])
```

#### 根据测试指标

```python
CLASS torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
mode='min', factor=0.1, patience=10, verbose=False, 
threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```



# 查看Pytorch网络的各层输出(feature map)、权重(weight)、偏置(bias)

## weight and bias

```python
# Method 1 查看Parameters的方式多样化，直接访问即可
model = alexnet(pretrained=True).to(device)
conv1_weight = model.features[0].weight

# Method 2 
# 这种方式还适合你想自己参考一个预训练模型写一个网络，各层的参数不变，但网络结构上表述有所不同
# 这样你就可以把param迭代出来，赋给你的网络对应层，避免直接load不能匹配的问题！
for layer,param in model.state_dict().items(): # param is weight or bias(Tensor) 
	print layer,param
```

## feature map
由于pytorch是动态网络，不存储计算数据，查看各层输出的特征图并不是很方便！分下面两种情况讨论：

1、你想查看的层是独立的,那么你在forward时用变量接收并返回即可！！

```python
class Net(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.conv2 = nn.Conv2d(1, 1, 3)
        self.conv3 = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = F.relu(self.conv3(out2))
        return out1, out2, out3
```

2、你的想看的层在nn.Sequential()顺序容器中，这个麻烦些，主要有以下几种思路：

```python
# Method 1 巧用nn.Module.children()
# 在模型实例化之后，利用nn.Module.children()删除你查看的那层的后面层
import torch
import torch.nn as nn
from torchvision import models

model = models.alexnet(pretrained=True)

# remove last fully-connected layer
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
# Third convolutional layer
new_features = nn.Sequential(*list(model.features.children())[:5])
model.features = new_features
```


​    
```python
# Method 2 巧用hook,推荐使用这种方式，不用改变原有模型
# torch.nn.Module.register_forward_hook(hook)
# hook(module, input, output) -> None

model = models.alexnet(pretrained=True)
# 定义
def hook (module,input,output):
    print output.size()
# 注册
handle = model.features[0].register_forward_hook(hook)
# 删除句柄
handle.remove()

# torch.nn.Module.register_backward_hook(hook)
# hook(module, grad_input, grad_output) -> Tensor or None
model = alexnet(pretrained=True).to(device)
outputs = []
def hook (module,input,output):
    outputs.append(output)
    print len(outputs)

handle = model.features[0].register_backward_hook(hook)
```

注：还可以通过定义一个提取特征的类，甚至是重构成各层独立相同模型将问题转化成第一种

## 计算模型参数数量
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```
# 自定义Operation(Function)
class torch.autograd.Function能为微分操作定义公式并记录操作历史，在Tensor上执行的每个操作都会创建一个新的函数对象，它执行计算，并记录它发生的事件。历史记录以函数的DAG形式保留，边表示数据依赖关系（输入< - 输出）。 然后，当backward被调用时，通过调用每个Function对象的backward()方法并将返回的梯度传递给下一个Function，以拓扑顺序处理图。

一般来说，用户与函数交互的唯一方法是通过创建子类并定义新的操作。这是拓展torch.autograd的推荐方法。

## 创建子类的注意事项

- 子类必须重写forward()，backward()方法，且为静态方法，定义时需加@staticmethod装饰器。
- forward()必须接受一个contextctx作为第一个参数，context可用于存储可在反向传播期间检索的张量。后面可接任意个数的参数(张量或者其他类型)。
- backward()必须接受一个contextctx作为第一个参数，context可用于检索前向传播期间保存的张量。
- 其参数是forward()给定输出的梯度，数量与forward()返回值个数一致。其返回值是forward()对应输入的梯度，数量与forward()的输入个数一致。
  使用class_name.apply(arg)的方式即可调用该操作

### 示例1：自定义ReLU激活函数

```python
class MyReLU(torch.autograd.Function):
"""
We can implement our own custom autograd Functions by subclassing
torch.autograd.Function and implementing the forward and backward passes
which operate on Tensors.
"""

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
```


### 示例2：自定义OHEMHingeLoss损失函数

```python
# from the https://github.com/yjxiong/action-detection
class OHEMHingeLoss(torch.autograd.Function):
    """
    This class is the core implementation for the completeness loss in paper.
    It compute class-wise hinge loss and performs online hard negative mining (OHEM).
    """

    @staticmethod
    def forward(ctx, pred, labels, is_positive, ohem_ratio, group_size):
        n_sample = pred.size()[0]
        assert n_sample == len(labels), "mismatch between sample size and label size"
        losses = torch.zeros(n_sample)
        slopes = torch.zeros(n_sample)
        for i in range(n_sample):
            losses[i] = max(0, 1 - is_positive * pred[i, labels[i] - 1])
            slopes[i] = -is_positive if losses[i] != 0 else 0

        losses = losses.view(-1, group_size).contiguous()
        sorted_losses, indices = torch.sort(losses, dim=1, descending=True)
        keep_num = int(group_size * ohem_ratio)
        loss = torch.zeros(1).cuda()
        for i in range(losses.size(0)):
            loss += sorted_losses[i, :keep_num].sum()
        ctx.loss_ind = indices[:, :keep_num]
        ctx.labels = labels
        ctx.slopes = slopes
        ctx.shape = pred.size()
        ctx.group_size = group_size
        ctx.num_group = losses.size(0)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        labels = ctx.labels
        slopes = ctx.slopes

        grad_in = torch.zeros(ctx.shape)
        for group in range(ctx.num_group):
            for idx in ctx.loss_ind[group]:
                loc = idx + group * ctx.group_size
                grad_in[loc, labels[loc] - 1] = slopes[loc] * grad_output.data[0]
        return torch.autograd.Variable(grad_in.cuda()), None, None, None, None
```