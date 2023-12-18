#<center>numpy学习笔记</center>#

###1. 利用numpy生成随机数据
```python
x, y = numpy.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], n) #生成x,y，其中[[1, 0], [0, 1]]是协方差

z = numpy.random.normal(mean, var, n) #生成n个服从正态分布的点
```

###2. numpy中矩阵乘法
```python
a * b #矩阵a和b对应位置相乘

a.dot(b) #线性代数中矩阵的乘法
```

###3. numpy中矩阵的保存和加载
```python
# 将一个numpy.array保存为一个.npy的二进制文件
numpy.save(filename, array, allow_pickle=True, fix_imports=True)

np.save('data/x_trn_%d.npy' % N_Cls, x)
```

```python
img = np.load('data/x_trn_%d.npy' % N_Cls)
```

###4. np.zeros_like(array)
生成一个全0矩阵,生成的矩阵的shape和array相同

###5. np.rollaxis(a, axis, start=0)
将指定的轴向后移动到指定的位置(start),其他轴的相对位置保持不变
```python
a = np.ones((3,4,5,6))
np.rollaxis(a, 3, 1).shape # (3,6,4,5)

np.rollaxis(a, 2).shape # (5，3，4，6)

np.rollaxis(a, 1, 4) #(3,5,6,4)
```

###6. np.stack(arrays, axis=0)
将一个数组序列按照一个新的axis组织成一个新的数组
```python
>>> arrays = [np.random.randn(3, 4) for _ in range(10)]
>>> np.stack(arrays, axis=0).shape
(10, 3, 4)
>>> np.stack(arrays, axis=1).shape
(3, 10, 4)
>>> np.stack(arrays, axis=2).shape
(3, 4, 10)
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.stack((a, b))
array([[1, 2, 3],
       [2, 3, 4]])
>>> np.stack((a, b), axis=-1)
array([[1, 2],
       [2, 3],
       [3, 4]])
```

###7. np.unravel_index(indices, dims, order="C")
返回indices中的下标在shape为dims中的下表是多少。如果indices是整数：
```python
np.unravel_index(1621, (6,7,8,9))
# 输出(3,1,4,1)

np.unravel_index(22, (7,6))
# 输出(3,4) 3*6+4=22
```
如果indices是数组:
```python
np.unravel_index([6,7,8], (3,3))
# (array([2, 2, 2]), array([0, 1, 2]))
# 2 2 2
# 0 1 2

In [23]: np.unravel_index([6], (3,3))
# Out[23]: (array([2]), array([0]))
```
1个$3\times 3$的矩阵，flattem后，对应的索引图为：

||||
|:-:|:-:|:-:|
|0|1|2|
|3|4|5|
|6|7|8|
那么indice6,7,8对应的原坐标:
col在前,order参数为"F"为 (0,2)(1,2)(2,2)
row在前,order参数为默认的"C"为(2,0)(2,1)(2,2)

###8. np.ceil()