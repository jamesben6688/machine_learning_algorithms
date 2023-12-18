#<center>deep learning.ai学习笔记</center>#

###1. padding层的作用

+ 避免图像的边缘信息被丢弃
+ 避免每一层filter过快的shrink

###2. pooling层的作用
+ 降维，加快计算
+ 让检测到的feature鲁棒性更强

###3. 对数据的看法
数据是驱动力。在以前数据贫乏的时代，手工设计特征用的非常多。例如SIFT等。现在的深度学习完全就是从数据中学习，只需要定义好网络，然后不断学习即可。学习出来的函数通常及其复杂。

###4. Bias和Variance
Bias和Variance是相对模型的泛化性能。

