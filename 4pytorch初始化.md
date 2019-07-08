

## 参数初始化（Weight Initialization）

PyTorch 中参数的默认初始化在各个层的 `reset_parameters()` 方法中。例如：`nn.Linear` 和 `nn.Conv2D`，都是在 \[-limit, limit\] 之间的均匀分布（Uniform distribution），其中 limit 是 `1. / sqrt(fan_in)` ，`fan_in` 是指参数张量（tensor）的输入单元的数量

下面是几种常见的初始化方式。

### Xavier Initialization

Xavier初始化的基本思想是保持输入和输出的方差一致，这样就避免了所有输出值都趋向于0。这是通用的方法，适用于任何激活函数。

```python
# 默认方法
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
```

也可以使用 `gain` 参数来自定义初始化的标准差来匹配特定的激活函数：

```python
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight(), gain=nn.init.calculate_gain('relu'))
```

参考资料：

- [Understanding the difficulty of training deep feedforward neural networks](https://www.pytorchtutorial.com/goto/http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

### He et. al Initialization

```
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

```

He initialization的思想是：在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0。推荐在ReLU网络中使用。

```python
# he initialization
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
```



### 正交初始化（Orthogonal Initialization）

主要用以解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用的参数初始化方法。

```python
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal(m.weight)
```



### Batchnorm Initialization

在非线性激活函数之前，我们想让输出值有比较好的分布（例如高斯分布），以便于计算梯度和更新参数。Batch Normalization 将输出值强行做一次 Gaussian Normalization 和线性变换：

![](https://www.pytorchtutorial.com/wp-content/uploads/2019/02/v2-2b14851823a6ec035cc16147eb5e04b0_hd.png)

实现方法：

```python
for m in model:
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
```









## 单层初始化

```python
conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
nn.init.xavier_uniform(conv1.weight)
nn.init.constant(conv1.bias, 0.1)
```

## 模型初始化

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
net = Net()
net.apply(weights_init) #apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上。
```

不建议访问以下划线为前缀的成员，他们是内部的，如果有改变不会通知用户。更推荐的一种方法是检查某个module是否是某种类型：

```
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
```



```python
import torch
import torch.nn as nn

w = torch.empty(2, 3)

# 1. 均匀分布 - u(a,b)
# torch.nn.init.uniform_(tensor, a=0, b=1)
nn.init.uniform_(w)
# tensor([[ 0.0578,  0.3402,  0.5034],
#         [ 0.7865,  0.7280,  0.6269]])

# 2. 正态分布 - N(mean, std)
# torch.nn.init.normal_(tensor, mean=0, std=1)
nn.init.normal_(w)
# tensor([[ 0.3326,  0.0171, -0.6745],
#        [ 0.1669,  0.1747,  0.0472]])

# 3. 常数 - 固定值 val
# torch.nn.init.constant_(tensor, val)
nn.init.constant_(w, 0.3)
# tensor([[ 0.3000,  0.3000,  0.3000],
#         [ 0.3000,  0.3000,  0.3000]])

# 4. 对角线为 1，其它为 0
# torch.nn.init.eye_(tensor)
nn.init.eye_(w)
# tensor([[ 1.,  0.,  0.],
#         [ 0.,  1.,  0.]])

# 5. Dirac delta 函数初始化，仅适用于 {3, 4, 5}-维的 torch.Tensor
# torch.nn.init.dirac_(tensor)
w1 = torch.empty(3, 16, 5, 5)
nn.init.dirac_(w1)

# 6. xavier_uniform 初始化
# torch.nn.init.xavier_uniform_(tensor, gain=1)
# From - Understanding the difficulty of training deep feedforward neural networks - Bengio 2010
nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
# tensor([[ 1.3374,  0.7932, -0.0891],
#         [-1.3363, -0.0206, -0.9346]])

# 7. xavier_normal 初始化
# torch.nn.init.xavier_normal_(tensor, gain=1)
nn.init.xavier_normal_(w)
# tensor([[-0.1777,  0.6740,  0.1139],
#         [ 0.3018, -0.2443,  0.6824]])

# 8. kaiming_uniform 初始化
# From - Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - HeKaiming 2015
# torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
# tensor([[ 0.6426, -0.9582, -1.1783],
#         [-0.0515, -0.4975,  1.3237]])

# 9. kaiming_normal 初始化
# torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
# tensor([[ 0.2530, -0.4382,  1.5995],
#         [ 0.0544,  1.6392, -2.0752]])

# 10. 正交矩阵 - (semi)orthogonal matrix
# From - Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe 2013
# torch.nn.init.orthogonal_(tensor, gain=1)
nn.init.orthogonal_(w)
# tensor([[ 0.5786, -0.5642, -0.5890],
#         [-0.7517, -0.0886, -0.6536]])

# 11. 稀疏矩阵 - sparse matrix 
# 非零元素采用正态分布 N(0, 0.01) 初始化.
# From - Deep learning via Hessian-free optimization - Martens 2010
# torch.nn.init.sparse_(tensor, sparsity, std=0.01)
nn.init.sparse_(w, sparsity=0.1)
# tensor(1.00000e-03 *
#        [[-0.3382,  1.9501, -1.7761],
#         [ 0.0000,  0.0000,  0.0000]])
```



### Xavier均匀分布
```python
torch.nn.init.xavier_uniform_(tensor, gain=1)
xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，
这里有一个gain，增益的大小是依据激活函数类型来设定
eg：nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain(‘relu’))
PS：上述初始化方法，也称为Glorot initialization

"""
torch.nn.init.xavier_uniform_(tensor, gain=1)
根据Glorot, X.和Bengio, Y.在“Understanding the dif×culty of training deep feedforward neural
networks”中描述的方法，用一个均匀分布生成值，填充输入的张量或变量。结果张量中的值
采样自U(-a, a)，其中a= gain * sqrt( 2/(fan_in + fan_out))* sqrt(3). 该方法也被称为Glorot initialisat

参数：
tensor – n维的torch.Tensor
gain - 可选的缩放因子
"""
import torch
from torch import nn
w=torch.Tensor(3,5)
nn.init.xavier_uniform_(w,gain=1)
print(w)
```



### Xavier正态分布

```python
torch.nn.init.xavier_normal_(tensor, gain=1)
xavier初始化方法中服从正态分布，
mean=0,std = gain * sqrt(2/fan_in + fan_out)

kaiming初始化方法，论文在《 Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification》，公式推导同样从“方差一致性”出法，kaiming是针对xavier初始化方法在relu这一类激活函数表现不佳而提出的改进，详细可以参看论文。

"""
根据Glorot, X.和Bengio, Y. 于2010年在“Understanding the dif×culty of training deep
feedforward neural networks”中描述的方法，用一个正态分布生成值，填充输入的张量或变
量。结果张量中的值采样自均值为0，标准差为gain * sqrt(2/(fan_in + fan_out))的正态分布。
也被称为Glorot initialisation.
参数：
tensor – n维的torch.Tensor
gain - 可选的缩放因子
"""
    
b=torch.Tensor(3,4)
nn.init.xavier_normal_(b, gain=1)
print(b)
```



### kaiming均匀分布

```python
torch.nn.init.kaiming_uniform_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’)
此为均匀分布，U～（-bound, bound）, bound = sqrt(6/(1+a^2)*fan_in)
其中，a为激活函数的负半轴的斜率，relu是0
mode- 可选为fan_in 或 fan_out, fan_in使正向传播时，方差一致; fan_out使反向传播时，方差一致
nonlinearity- 可选 relu 和 leaky_relu ，默认值为 。 leaky_relu
nn.init.kaiming_uniform_(w, mode=‘fan_in’, nonlinearity=‘relu’)

w=torch.Tensor(3,5)
nn.init.kaiming_normal_(w,a=0,mode='fan_in')
print(w)
```



### kaiming正态分布

```
torch.nn.init.kaiming_normal_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’)
此为0均值的正态分布，N～ (0,std)，其中std = sqrt(2/(1+a^2)*fan_in)
其中，a为激活函数的负半轴的斜率，relu是0
mode- 可选为fan_in 或 fan_out, fan_in使正向传播时，方差一致;fan_out使反向传播时，方差一致
nonlinearity- 可选 relu 和 leaky_relu ，默认值为 。 leaky_relu
nn.init.kaiming_normal_(w, mode=‘fan_out’, nonlinearity=‘relu’)
```


2.其他

### 均匀分布初始化

torch.nn.init.uniform_(tensor, a=0, b=1)
使值服从均匀分布U(a,b)

tensor - n维的torch.Tensor
a - 均匀分布的下界
b - 均匀分布的上界



### 正态分布初始化

torch.nn.init.normal_(tensor, mean=0, std=1)
使值服从正态分布N(mean, std)，默认值为0，1

tensor – n维的torch.Tensor
mean – 正态分布的均值
std – 正态分布的标准差



### 常数初始化

torch.nn.init.constant_(tensor, val)
使值为常数val nn.init.constant_(w, 0.3)

```python
"""
torch.nn.init.constant(tensor, val)
用val的值填充输入的张量或变量
参数：
tensor – n维的torch.Tensor或autograd.Variable
val – 用来填充张量的值
"""
w=torch.Tensor(3,5)
nn.init.constant_(w,1.2)
print(w)
tensor([[1.2000, 1.2000, 1.2000, 1.2000, 1.2000],
        [1.2000, 1.2000, 1.2000, 1.2000, 1.2000],
        [1.2000, 1.2000, 1.2000, 1.2000, 1.2000]])
```



### 单位矩阵初始化

torch.nn.init.eye_(tensor)
将二维tensor初始化为单位矩阵（the identity matrix）

```python

"""
torch.nn.init.eye(tensor)
用单位矩阵来填充2维输入张量或变量。在线性层尽可能多的保存输入特性。
参数：
tensor – 2维的torch.Tensor或autograd.Variable
"""
w=torch.Tensor(3,5)
nn.init.eye_(w)
print(w)
tensor([[1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.]])

```



### 正交初始化

torch.nn.init.orthogonal_(tensor, gain=1)
使得tensor是正交的，论文:Exact solutions to the nonlinear dynamics of learning in deep linear neural networks” - Saxe, A. et al. (2013)

```python
"""
torch.nn.init.orthogonal_(tensor, gain=1)
25 torch.nn.init - PyTorch中文文档
https://pytorch-cn.readthedocs.io/zh/latest/package_references/nn_init/ 5/5
用（半）正交矩阵填充输入的张量或变量。输入张量必须至少是2维的，对于更高维度的张
量，超出的维度会被展平，视作行等于第一个维度，列等于稀疏矩阵乘积的2维表示。其中非
零元素生成自均值为0，标准差为std的正态分布。

参数：
tensor – n维的torch.Tensor或 autograd.Variable，其中n>=2
gain -可选
"""
w = torch.Tensor(3, 5)
nn.init.orthogonal_(w)
print(w)
```



### 稀疏初始化

torch.nn.init.sparse_(tensor, sparsity, std=0.01)
从正态分布N～（0. std）中进行稀疏化，使每一个column有一部分为0
sparsity- 每一个column稀疏的比例，即为0的比例_

sparsity - 每列中需要被设置成零的元素比例
std - 用于生成非零值的正态分布的标准差
nn.init.sparse_(w, sparsity=0.1)

```python
w = torch.Tensor(3, 5)
nn.init.sparse_(w, sparsity=0.1)
print(w)

tensor([[-0.0042,  0.0000,  0.0000, -0.0016,  0.0000],
        [ 0.0000,  0.0050,  0.0082,  0.0000,  0.0003],
        [ 0.0018, -0.0016, -0.0003, -0.0068,  0.0103]])
```



### dirac

```python
"""
torch.nn.init.dirac(tensor)
用Dirac 函数来填充{3, 4, 5}维输入张量或变量。在卷积层尽可能多的保存输入通道特性
参数：
tensor – {3, 4, 5}维的torch.Tensor或autograd.Variable
"""
w=torch.Tensor(3,16,5,5)
nn.init.dirac_(w)
print(w)

w.sum()
tensor(3.)
```





### 计算增益calculate_gain

torch.nn.init.calculate_gain(nonlinearity, param=None)

```python
torch.nn.init.calculate_gain(nonlinearity,param=None)
对于给定的非线性函数，返回推荐的增益值.
参数：
nonlinearity - 非线性函数（ nn.functional 名称）
param - 非线性函数的可选参数

from torch import nn
import torch
gain = nn.init.calculate_gain('leaky_relu')
print(gain)

1.4141428569978354
```



|nonlinearity|	gain|
| ---- | ---- |
|Linear / Identity|	1|
|Conv{1,2,3}D|	1|
|Sigmoid|	1|
|Tanh	|5/3|
|ReLU	|sqrt(2)|
||         |

