把最常用的记住就行了

# 1交叉熵损失函数CrossEntropyLoss

cross_entropy输入的logits是未经过softmax层的输出。

而标签值为一个数字，而不是对应的one-hot向量。
$$
loss(x, class) = -log(\frac{exp(x[class])}{(\sum_j exp(x[j]))})
               = -x[class] + log(\sum_j exp(x[j]))
$$


![img](F:/%E7%AC%94%E8%AE%B0%E6%95%B4%E7%90%86/%E6%9C%89%E9%81%93%E4%BA%91%E7%AC%94%E8%AE%B0/yangsenupc@163.com/32f7540b9310445d879401203e4e0881/clipboard.png)

```python
class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
# 将输入经过 softmax 激活函数之后，再计算其与 target 的交叉熵损失。即该方法将
# nn.LogSoftmax()和 nn.NLLLoss()进行了结合
# 输入的target是标签，而不能是对应的one-hot向量

#weight:a manual rescaling weight given to each class. If given, has to be a Tensor of size  C
```

## 模板

```python
criteon = nn.CrossEntropyLoss().to(device)

for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        logits = model(data)
        loss = criteon(logits, target)
        # logits.shape:[batch,C] ,target.shape:[batch]
        # C为类别总数
        # 手写数字识别的例子batch=50
        # torch.Size([50, 10])
		# torch.Size([50])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```



| torch.nn         | torch.nn.functional (F) |
| ---------------- | ----------------------- |
| CrossEntropyLoss | cross_entropy           |
| LogSoftmax       | log_softmax             |
| NLLLoss          | nll_loss                |

# 2NLLLoss

negative log likelihood loss：最大似然 / log似然代价函数

```python
torch.nn.NLLLoss
loss(input, class) = -input[class]。 举个例，三分类任务， 
input=[-1.233, 2.657, 0.534]， 真实标签为 2（class=2），则 loss 为-0.534
```

nll-loss 输入的则是经过softmax和log后的输出

## 模板

```python
out=F.log_softmax(out,dim=1)
# #带log的softmax分类，每个样本返回N个概率,N为类别总数
```

```python
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

```



# 3L1loss 

## 功能

计算 output 和 target 之差的绝对值，可选返回同维度的 tensor 或者是一个标量 

```python
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

reduce(bool)-返回值是否为标量，默认为 True
size_average(bool)-当 reduce=True 时有效。为 True 时，返回的 loss 为平均值；为 False
时，返回的各样本的 loss 之和。
size_average and reduce are in the process of being deprecated

reduction='mean':输出为标量,求均值
reduction='sum':输出为标量,求和
reduction='none':输出为张量,不降维
```

![img](https://pytorch.apachecn.org/docs/1.0/img/415564bfa6c89ba182a02fe2a3d0ca49.jpg)

where $N$ is the batch size. If reduce is `True`, then:
$$
\ell(x, y)=\left\{\begin{array}{ll}{\operatorname{mean}(L),} & {\text { if reduction }=\text { 'mean' }} \\ {\operatorname{sum}(L),} & {\text { if reduction }=\text { 'sum' }}\end{array}\right.
$$

$$
\begin{array}{l}{\text { Input }(N, *) \text { where } * \text { means, any number of additional dimensions }} \\ {\text { Target: }(N, *) \text { , same shape as the input }} \\ {\text { Output: scalar. If reduce is Falue, then }(N, *), \text { same shape as the input }}\end{array}
$$


# 4MSELoss

## 功能

计算 output 和 target 之差的平方，可选返回同维度的 tensor 或者是一个标量 



```python
torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
reduction='mean':输出为标量,求均值
reduction='sum':输出为标量,求和
reduction='none':输出为张量,不降维
```

The unreduced (i.e. with `reduction` set to `'none'`) loss can be described as:
$$
\ell(x, y)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top}, \quad l_{n}=\left(x_{n}-y_{n}\right)^{2}
$$
where $N$ is the batch size. If `reduction` is not `'none'` (default `'mean'`), then:
$$
\ell(x, y)=\left\{\begin{array}{ll}{\operatorname{mean}(L),} & {\text { if reduction }=\text { 'mean' }} \\ {\operatorname{sum}(L),} & {\text { if reduction }=\text { 'sum' }}\end{array}\right.
$$


# 5BCELoss 二分类任务时的交叉熵

## 功能

二分类任务时的交叉熵计算函数。此函数可以认为是 nn.CrossEntropyLoss 函数的特例。其分类限定为二分类， y 必须是{0,1}。还需要注意的是， input 应该为概率分布的形式，这样才符合交叉熵的应用。所以在 BCELoss 之前， input 一般为 sigmoid 激活层的输出

```python
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
# weight(Tensor)- 为每个类别的loss设置权值，常用于类别不均衡问题。
```

The unreduced (i.e. with `reduction` set to `'none'`) loss can be described as
$$
\ell(x, y)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{T}, \quad l_{n}=-w_{n}\left[y_{n} \cdot \log x_{n}+\left(1-y_{n}\right) \cdot \log \left(1-x_{n}\right)\right]\\
\ell(x, y)=\left\{\begin{array}{ll}{\operatorname{mean}(L),} & {\text { if reduction }=\text { 'mean' }} \\ {\operatorname{sum}(L),} & {\text { if reduction }=\text { 'sum' }}\end{array}\right.
$$


# 6BCEWithLogitsLoss 

## 功能

将 Sigmoid 与 BCELoss 结合，类似于 CrossEntropyLoss(将 nn.LogSoftmax()和 nn.NLLLoss()进行结合）。即 input 会经过 Sigmoid 激活函数，将 input 变成概率分布的形式。 

```python
torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
# weight(Tensor): 为batch中单个样本设置权值，If given, has to be a Tensor of size “nbatch”.
# pos_weight: 正样本的权重, 当p>1，提高召回率，当P<1，提高精确度。可达到权衡召回率(Recall)和精确度(Precision)的作用。 Must be a vector with length equal to the number of classes.
```


$$
\ell(x, y)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top}, \quad l_{n}=-w_{n}\left[y_{n} \cdot \log \sigma\left(x_{n}\right)+\left(1-y_{n}\right) \cdot \log \left(1-\sigma\left(x_{n}\right)\right)\right]
$$





# 7.PoissonNLLLoss
```
torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
```

功能：
用于target服从泊松分布的分类任务。
计算公式：
$$
\text{target} \sim \mathrm{Poisson}(\text{input})\\
\text{loss}(\text{input}, \text{target}) = \text{input} - \text{target} * \log(\text{input}) + \log(\text{target!})
$$
参数：

- log_input(bool)- 为True时，计算公式为：$loss(input,target)=exp(input) - target * input$;
    为False时，$loss(input,target)=input - target * log(input+eps)$

- full(bool)- 是否计算全部的loss。例如，当采用斯特林公式近似阶乘项时，此为 target*log(target) - target+0.5∗log(2πtarget)

- eps(float)- 当log_input = False时，用来防止计算log(0)，而增加的一个修正项。即 $loss(input,target)=input - target * log(input+eps)$

- reduction (*string*,*optional*) – Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

    ### shape

    Input: $(N,∗)$ where $∗ $means, any number of additional dimensions

    Target: $(N,∗)$, same shape as the input

    Output: scalar by default. If `reduction` is `'none'`, then$(N,∗)$ the same shape as the input

# 8.KLDivLoss
```python
torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean')

```

功能：
计算input和target之间的KL散度( Kullback–Leibler divergence) 。
计算公式：
$$
l(x, y)=L=\left\{l_{1}, \ldots, l_{N}\right\}, \quad l_{n}=y_{n} \cdot\left(\log y_{n}-x_{n}\right)
$$
（后面有代码手动计算，证明计算公式确实是这个，但是为什么没有对x_n计算对数呢？）

 If `reduction` is not `'none'`(default `'mean'`), then:
$$
\ell(x, y) = \begin{cases} \operatorname{mean}(L); \text{if reduction} = \text{'mean';} \\ \operatorname{sum}(L); \text{if reduction} = \text{'sum';} \end{cases}
$$
补充：KL散度
KL散度( Kullback–Leibler divergence) 又称为相对熵(Relative Entropy)，用于描述两个概率分布之间的差异。计算公式(离散时)：

其中p表示真实分布，q表示p的拟合分布， D(P||Q)表示当用概率分布q来拟合真实分布p时，产生的信息损耗。这里的信息损耗，可以理解为损失，损失越低，拟合分布q越接近真实分布p。同时也可以从另外一个角度上观察这个公式，即计算的是 p 与 q 之间的对数差在 p 上的期望值。
特别注意，D(p||q) ≠ D(q||p)， 其不具有对称性，因此不能称为K-L距离。
信息熵 = 交叉熵 - 相对熵
从信息论角度观察三者，其关系为信息熵 = 交叉熵 - 相对熵。在机器学习中，当训练数据固定，最小化相对熵 D(p||q) 等价于最小化交叉熵 H(p,q) 。

参数：
**reduction** (*string*, *optional*) – Specifies the reduction to apply to the output: `'none'` | `'batchmean'` | `'sum'` | `'mean'`. `'none'`: no reduction will be applied. `'batchmean'`: the sum of the output will be divided by batchsize. `'sum'`: the output will be summed. `'mean'`: the output will be divided by the number of elements in the output. Default: `'mean'`

使用注意事项：
要想获得真正的KL散度，需要如下操作：

`reduction` = `'mean'` doesn’t return the `true kl divergence` value, please use `reduction` = `'batchmean'` which aligns with KL math definition. 在下一个主要版本中, `'mean'` will be changed to be the same as `'batchmean'`.

## shape

Input: $(N,∗)$ where $∗ $means, any number of additional dimensions

Target: $(N,∗)$, same shape as the input

Output: scalar by default. If `reduction` is `'none'`, then$(N,∗)$ the same shape as the input



# 9.MarginRankingLoss
```
torch.nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```

功能：
计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0。
计算公式：
$$
\operatorname{loss}(x, y)=\max (0,-y *(x 1-x 2)+\operatorname{margin})
$$
y = 1时，x1要比x2大，才不会有loss，反之，y = -1 时，x1要比x2小，才不会有loss。
参数：
margin(float):x1和x2之间的差异。
**reduction** (*string*, *optional*) – Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

## shape

$$
\begin{array}{l}{\text { Input: }(N, D) \text { where } N \text { is the batch size and } D \text { is the size of a sample. }} \\ {\text { Target: }(N)} \\ {\text { Output: scalar. If reduction is 'none', then }(N) .}\end{array}
$$

# 10.HingeEmbeddingLoss
```
torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')
```

功能：
未知。为折页损失的拓展，主要用于衡量两个输入是否相似。 used for learning nonlinear embeddings or semi-supervised 。
计算公式：

The loss function for n*n*-th sample in the mini-batch is

$l_n = \begin{cases} x_n,  \text{if}\; y_n = 1,\\ \max \{0, \Delta - x_n\}, \text{if}\; y_n = -1, \end{cases}$

and the total loss functions is

$\ell(x, y) = \begin{cases} \operatorname{mean}(L), \text{if reduction} = \text{mean;}\\ \operatorname{sum}(L), \text{if reduction} = \text{sum.} \end{cases}$

where $L = \{l_1,\dots,l_N\}^\top.$

参数：
margin(float)- 默认值为1，容忍的差距。

## shape

$$
\begin{array}{l}{\text { input: } : \text { (*) where * means, any number of dimensions.
The sum operation operates over all the elements. }} \\ {\text { Target: }(*), \text { same shape as the input }} \\ {\text { Output: scalar. If reduction is 'none', then same shape as the input }}\end{array}
$$

# 11.MultiLabelMarginLoss
class torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction=‘elementwise_mean’)
功能：
用于一个样本属于多个类别时的分类任务。例如一个四分类任务，样本x属于第0类，第1类，不属于第2类，第3类。
计算公式：

x[y[j]] 表示 样本x所属类的输出值，x[i]表示不等于该类的输出值。

参数：
size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
reduce(bool)- 返回值是否为标量，默认为True。
Input: © or (N,C) where N is the batch size and C is the number of classes.
Target: © or (N,C), same shape as the input.

# 12.SmoothL1Loss
class torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction=‘elementwise_mean’)
功能：
计算平滑L1损失，属于 Huber Loss中的一种(因为参数δ固定为1了)。
补充：
Huber Loss常用于回归问题，其最大的特点是对离群点（outliers）、噪声不敏感，具有较强的鲁棒性。
公式为：

理解为，当误差绝对值小于δ，采用L2损失；若大于δ，采用L1损失。
回到SmoothL1Loss，这是δ=1时的Huber Loss。
计算公式为：

对应下图红色线：

参数：
size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
reduce(bool)- 返回值是否为标量，默认为True。

# 13.SoftMarginLoss
class torch.nn.SoftMarginLoss(size_average=None, reduce=None, reduction=‘elementwise_mean’)
功能：
Creates a criterion that optimizes a two-class classification logistic loss between input tensor xand target tensor y (containing 1 or -1). （暂时看不懂怎么用，有了解的朋友欢迎补充！）
计算公式：

参数：
size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
reduce(bool)- 返回值是否为标量，默认为True。

# 14.MultiLabelSoftMarginLoss
class torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction=‘elementwise_mean’)
功能：
SoftMarginLoss多标签版本，a multi-label one-versus-all loss based on max-entropy,
计算公式：

参数：
weight(Tensor)- 为每个类别的loss设置权值。weight必须是float类型的tensor，其长度要于类别C一致，即每一个类别都要设置有weight。

# 15.CosineEmbeddingLoss
class torch.nn.CosineEmbeddingLoss(margin=0, size_average=None, reduce=None, reduction=‘elementwise_mean’)
功能：
用Cosine函数来衡量两个输入是否相似。 used for learning nonlinear embeddings or semi-supervised 。
计算公式：

参数：
margin(float)- ： 取值范围[-1,1]， 推荐设置范围 [0, 0.5]
size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
reduce(bool)- 返回值是否为标量，默认为True。

# 16.MultiMarginLoss
class torch.nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=None, reduce=None, reduction=‘elementwise_mean’)
功能：
计算多分类的折页损失。
计算公式：

其中，0≤y≤x.size(1) ; i == 0 to x.size(0) and i≠y; p==1 or p ==2; w[y]为各类别的weight。
参数：
p(int)- 默认值为1，仅可选1或者2。
margin(float)- 默认值为1
weight(Tensor)- 为每个类别的loss设置权值。weight必须是float类型的tensor，其长度要于类别C一致，即每一个类别都要设置有weight。
size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
reduce(bool)- 返回值是否为标量，默认为True。

# 17.TripletMarginLoss
class torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction=‘elementwise_mean’)
功能：
计算三元组损失，人脸验证中常用。
如下图Anchor、Negative、Positive，目标是让Positive元和Anchor元之间的距离尽可能的小，Positive元和Negative元之间的距离尽可能的大。

从公式上看，Anchor元和Positive元之间的距离加上一个threshold之后，要小于Anchor元与Negative元之间的距离。

计算公式：


参数：
margin(float)- 默认值为1
p(int)- The norm degree ，默认值为2
swap(float)– The distance swap is described in detail in the paper Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al. Default: False
size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。

reduce(bool)- 返回值是否为标量，默认为True。



# 18CTCLoss



