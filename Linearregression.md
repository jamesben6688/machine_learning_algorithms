#<center>回归分析学习笔记</center>#

回归主要分为线性回归和逻辑回归。线性回归主要解决连续值预测问题，逻辑回归主要解决分类问题。但逻辑回归输出的是属于某一类的概率，因此常被用来进行排序。

##1线性回归的原理
假定输入$\chi$和输出$y$之间有线性相关关系，线性回归就是学习一个映射
$$f: \chi \to y$$
然后对于给你的样本$x$，预测其输出：
$$
\hat y=f(x)
$$

现假定$x=(x_0,x_1\dots x_n)$，则预测值为：

$$
h_\theta(x)=\sum_{i=0}^n\theta_ix_i=\theta^Tx
$$
在特征$x$中加上一维$x_0=1$表示截距，即：
$$
f(x)=\theta_0+\theta_1x_1+\theta_2x_2+\dots+\theta_nx_n
$$

##2 损失函数
为了找到组好的权重参数$\theta$，令$X$到$y$的映射函数记为
$$f(x)=h_\theta(x)$$
其中
$$
\theta=(\theta_0, \theta_1\dots\theta_n)
$$
为了评价模型拟合的效果，对于一个特定的数据集$(X,y)$定义一个损失函数来计算预测值与真实值之间的误差：
$$
J_\theta(X)=J_{(\theta_0, \theta_1\dots\theta_n)}(X)=\frac 1{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
$$
即总体误差是所有样本点误差平方和的均值，其中$(x^{(i)},y^{(i)})$表示的是第$i$个样本点。现在给定数据集$(X, y)$，要求解的目标为使得$J_\theta(X)$最小的$\theta$，即：
$$
\theta = \arg\min_\theta \{\frac 1{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2\}
$$

##3 梯度下降
假设有一堆样本点$(x_1, y_1)(x_2,y_2)\dots (x_n,y_n)$，定义函数$h_\theta(x)$来模拟$y$。假设最后的拟合函数为$J_\theta(X)=h_\theta(X)$。则损失函数为：
$$
J(\theta)=\frac 1{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
$$

+ 首先随机初始化$\theta$，例如令$\vec\theta=\vec 0$。
+ 不断变化$\vec \theta$的值来改变$J(\theta)$的值，使其越来越小。改变的规则为：
$$
\theta_i:=\theta_i-\alpha\frac {\partial J(\theta)}{\partial \theta_i}
$$
$$
\frac {\partial J(\theta)}{\partial \theta_i}=\sum_{j=1}^m(h_\theta(x^{(j)})-y^{(j)})x_i^{(j)}
$$
因此对于所有的$m$个样本点求和，有：
$$
\theta_i:=\theta_i-\alpha\sum_{j=1}^m[(h_\theta(x^{(j)})-y^{(j)})\cdot x_i^{(j)}]
$$
其中$x^{(j)}，y^{(j)}$表示第$j$个样本点，$x^{(j)}$是一个向量，$x_i^{(j)}$表示第$j$个样本点$x^{(j)}$的第$i$个分量，是一个标量。

+ 不断重复上述过程，直到最后收敛(例如最后发现损失函数$J_\theta(X)$基本不再变化)。

整个过程当中，$\theta, h_\theta(x), J_\theta(X)$都会不断变化，但是$h_\theta(x)$会越来越接近$y$，因此$J_\theta(x)$会变得越来越小，最后接近0。

###1.4 利用最小二乘拟合的方法来计算$\theta$
$$
X=
\begin{bmatrix}
(x^{(1)})^T\\
(x^{(2)})^T\\
 \vdots \\
(x^{(n)})^T\\
\end{bmatrix}
$$
$$
X\cdot \theta=
\begin{bmatrix}
(x^{(1)})^T \theta\\
(x^{(2)})^T \theta\\
 \vdots \\
(x^{(n)})^T\theta\\
\end{bmatrix}=
\begin{bmatrix}
h_\theta(x^{(1)})^T\\
h_\theta(x^{(2)})^T\\
 \vdots \\
h_\theta(x^{(n)})^T\\
\end{bmatrix}
$$
$$
y=
\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
\vdots \\
y^{(n)}\\
\end{bmatrix}
$$

$$
X\cdot \theta-y=
\begin{bmatrix}
h_\theta(x^{(1)})^T-y^{(1)}\\
h_\theta(x^{(2)})^T-y^{(2)}\\
 \vdots \\
h_\theta(x^{(n)})^T-y^{(n)}\\
\end{bmatrix}
$$
为了计算函数$J_\theta(x)$在指定的计算步骤内达到的最小值，每次我们都沿当前点下降最快的方向移动。最快的方向即梯度方向：
$$
(\frac {\partial J_\theta(x^{(i)})}{\partial \theta_0}, \frac {\partial J_\theta(x^{(i)})}{\partial \theta_1}\dots \frac {\partial J_\theta(x^{(i)})}{\partial \theta_n})
$$

假设$z$是一个向量，$z=
\begin{pmatrix}
z_1\\
z_2\\
 \vdots \\
z_n\\
\end{pmatrix}$。则:$z^Tz=\sum_{i=0}^nz_i^2$。

故
$$
(X\theta -y)^T(X\theta -y)=\frac 12\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
$$
则
$$
J(\theta)=\frac 12(X\theta -y)^T(X\theta -y)
$$
要求梯度，令
$$
\nabla_\theta J(\theta)= \vec 0
$$

$$
\nabla_\theta J(\theta)= \nabla_\theta \frac 12(x\theta-y)^T(x\theta-y)=x^Tx\theta-x^Ty=\vec 0
$$
求得
$$
\vec \theta=(x^Tx)^{-1}x^Ty
$$
最终$\vec \theta$是一个$m\times 1$的向量。这样对于简单的线性回归问题，就不需要用前面的迭代方法啦。
>如果$x^Tx$是不可逆的，说明$x$当中有特征冗余，需要去掉某些特征。