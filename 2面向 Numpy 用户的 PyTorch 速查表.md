# 类型（Types）

| Numpy      | PyTorch                     |
| ---------- | --------------------------- |
| np.ndarray | torch.Tensor                |
| np.float32 | torch.float32; torch.float  |
| np.float64 | torch.float64; torch.double |
| np.float16 | torch.float16; torch.half   |
| np.int8    | torch.int8                  |
| np.uint8   | torch.uint8                 |
| np.int16   | torch.int16; torch.short    |
| np.int32   | torch.int32; torch.int      |
| np.int64   | torch.int64; torch.long     |

# 构造器（Constructor）

## 零和一（Ones and zeros）

| Numpy            | PyTorch             |
| ---------------- | ------------------- |
| np.empty((2, 3)) | torch.empty(2, 3)   |
| np.empty_like(x) | torch.empty_like(x) |
| np.eye           | torch.eye           |
| np.identity      | torch.eye           |
| np.ones          | torch.ones          |
| np.ones_like     | torch.ones_like     |
| np.zeros         | torch.zeros         |
| np.zeros_like    | torch.zeros_like    |

## 从已知数据构造

| Numpy                                                        | PyTorch                                       |
| ------------------------------------------------------------ | --------------------------------------------- |
| np.array([[1, 2], [3, 4]])                                   | torch.tensor([[1, 2], [3, 4]])                |
| np.array([3.2, 4.3], dtype=np.float16)np.float16([3.2, 4.3]) | torch.tensor([3.2, 4.3], dtype=torch.float16) |
| x.copy()                                                     | x.clone()                                     |
| np.fromfile(file)                                            | torch.tensor(torch.Storage(file))             |
| np.frombuffer                                                |                                               |
| np.fromfunction                                              |                                               |
| np.fromiter                                                  |                                               |
| np.fromstring                                                |                                               |
| np.load                                                      | torch.load                                    |
| np.loadtxt                                                   |                                               |
| np.concatenate                                               | torch.cat                                     |

## 数值范围

| Numpy                | PyTorch                 |
| -------------------- | ----------------------- |
| np.arange(10)        | torch.arange(10)        |
| np.arange(2, 3, 0.1) | torch.arange(2, 3, 0.1) |
| np.linspace          | torch.linspace          |
| np.logspace          | torch.logspace          |

## 构造矩阵

| Numpy   | PyTorch    |
| ------- | ---------- |
| np.diag | torch.diag |
| np.tril | torch.tril |
| np.triu | torch.triu |

## 参数

| Numpy     | PyTorch      |
| --------- | ------------ |
| x.shape   | x.shape      |
| x.strides | x.stride()   |
| x.ndim    | x.dim()      |
| x.data    | x.data       |
| x.size    | x.nelement() |
| x.dtype   | x.dtype      |

## 索引

| Numpy               | PyTorch                                  |
| ------------------- | ---------------------------------------- |
| x[0]                | x[0]                                     |
| x[:, 0]             | x[:, 0]                                  |
| x[indices]          | x[indices]                               |
| np.take(x, indices) | torch.take(x, torch.LongTensor(indices)) |
| x[x != 0]           | x[x != 0]                                |

## 形状（Shape）变换

| Numpy                                  | PyTorch                  |
| -------------------------------------- | ------------------------ |
| x.reshape                              | x.reshape; x.view        |
| x.resize()                             | x.resize_                |
|                                        | x.resize_as_             |
| x.transpose                            | x.transpose or x.permute |
| x.flatten                              | x.view(-1)               |
| x.squeeze()                            | x.squeeze()              |
| x[:, np.newaxis]; np.expand_dims(x, 1) | x.unsqueeze(1)           |

## 数据选择

| Numpy                                                   | PyTorch                                                      |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| np.put                                                  |                                                              |
| x.put                                                   | x.put_                                                       |
| x = np.array([1, 2, 3])x.repeat(2) # [1, 1, 2, 2, 3, 3] | x = torch.tensor([1, 2, 3])x.repeat(2) # [1, 2, 3, 1, 2, 3]x.repeat(2).reshape(2, -1).transpose(1, 0).reshape(-1) # [1, 1, 2, 2, 3, 3] |
| np.tile(x, (3, 2))                                      | x.repeat(3, 2)                                               |
| np.choose                                               |                                                              |
| np.sort                                                 | sorted, indices = torch.sort(x, [dim])                       |
| np.argsort                                              | sorted, indices = torch.sort(x, [dim])                       |
| np.nonzero                                              | torch.nonzero                                                |
| np.where                                                | torch.where                                                  |
| x[::-1]                                                 |                                                              |

## 数值计算

| Numpy       | PyTorch                        |
| ----------- | ------------------------------ |
| x.min       | x.min                          |
| x.argmin    | x.argmin                       |
| x.max       | x.max                          |
| x.argmax    | x.argmax                       |
| x.clip      | x.clamp                        |
| x.round     | x.round                        |
| np.floor(x) | torch.floor(x); x.floor()      |
| np.ceil(x)  | torch.ceil(x); x.ceil()        |
| x.trace     | x.trace                        |
| x.sum       | x.sum                          |
| x.cumsum    | x.cumsum                       |
| x.mean      | x.mean                         |
| x.std       | x.std                          |
| x.prod      | x.prod                         |
| x.cumprod   | x.cumprod                      |
| x.all       | (x == 1).sum() == x.nelement() |
| x.any       | (x == 1).sum() > 0             |

## 数值比较

| Numpy            | PyTorch |
| ---------------- | ------- |
| np.less          | x.lt    |
| np.less_equal    | x.le    |
| np.greater       | x.gt    |
| np.greater_equal | x.ge    |
| np.equal         | x.eq    |
| np.not_equal     | x.ne    |

 



pytorch与tensorflow API速查表
|方法名称	|pytroch	|tensorflow	|numpy|
| ---------------- | ------- | ------- | ------- |
|裁剪	|torch.clamp(x, min, max)	|tf.clip_by_value(x, min, max)	|np.clip(x, min, max)|
|取最小值|	torch.min(x, dim)[0]|	tf.min(x, axis)|	np.min(x , axis)|
|取两个tensor的最大值|	torch.max(x, y)|	tf.maximum(x, y)|	np.maximum(x, y)|
|取两个tensor的最小值|	torch.min(x, y)	|torch.minimum(x, y)|	np.minmum(x, y)|
|取最大值索引|	torch.max(x, dim)[1]|	tf.argmax(x, axis)|	np.argmax(x, axis)|
|取最小值索引|	torch.min(x, dim)[1]|	tf.argmin(x, axis)|	np.argmin(x, axis)|
|比较(x > y)|	torch.gt(x, y)|	tf.greater(x, y)|	np.greater(x, y)|
|比较(x < y)	|torch.le(x, y)|	tf.less(x, y)|	np.less(x, y)|
|比较(x==y)|	torch.eq(x, y)|	tf.equal(x, y)|	np.equal(x, y)|
|比较(x!=y)	|torch.ne(x, y)	|tf.not_equal(x, y)|	np.not_queal(x , y)|
|取符合条件值的索引|	torch.nonzero(cond)|	tf.where(cond)	|np.where(cond)|
|多个tensor聚合	|torch.cat([x, y], dim)|	tf.concat([x,y], axis)	|np.concatenate([x,y], axis)|
|堆叠成一个tensor|	torch.stack([x1, x2], dim)	|tf.stack([x1, x2], axis)|	np.stack([x, y], axis) |
|tensor切成多个tensor|	torch.split(x1, split_size_or_sections, dim)|	tf.split(x1, num_or_size_splits, axis)	|np.split(x1, indices_or_sections, axis)	|
|-|torch.unbind(x1, dim)|	tf.unstack(x1,axis)|	NULL|
|随机扰乱| torch.randperm(n)    1 |	tf.random_shuffle(x)| np.random.shuffle(x)   2 np.random.permutation(x )  3 |
|前k个值|	torch.topk(x, n, sorted, dim)|	tf.nn.top_k(x, n, sorted)|	NULL|

1. 该方法只能对0~n-1自然数随机扰乱，所以先对索引随机扰乱，然后再根据扰乱后的索引取相应的数据得到扰乱后的数据 
2. 该方法会修改原值，没有返回值
3. 该方法不会修改原值，返回扰乱后的值