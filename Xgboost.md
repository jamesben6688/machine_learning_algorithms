##<center>Xgboost</center>##
###1. xgboost原理
首先明确下我们的目标，希望建立K个回归树，使得树群的预测值尽量接近真实值（准确率）而且有尽量大的泛化能力（更为本质的东西），从数学角度看这是一个泛函最优化，多目标，看下目标函数：
$$
L(\phi)=\sum_i{L(\hat y_i-y_i)}+\sum_k\Omega(f_k)
$$

其中$i$表示第i个样本。$L$表示第$i$个样本的预测误差，误差越小越好。后面$\sum_k\Omega(f_k)$表示树的复杂度，复杂度越小越低，泛化能力越强。其表达式为:
$$
\Omega(f_k)=\gamma T+\frac 12\lambda||w||^2
$$
其中$T$表示叶子节点数目，$w$表示叶子节点数值(这是回归树的东西，分类树对应的是类别)。

直观上看，目标要求预测误差尽量小，叶子节点尽量少，节点数值尽量不极端。例如：某个样本$label$为4，那么第一个回归树预测3，第二个预测为1；另外一组回归树，一个预测2，一个预测2，那么倾向后一种，为什么呢？前一种情况，第一棵树学的太多，太接近4，也就意味着有较大的过拟合的风险。

ok，听起来很美好，可是怎么实现呢，上面这个目标函数跟实际的参数怎么联系起来，记得我们说过，回归树的参数:
+ 选取哪个feature分裂点
+ 节点的预测值

上述形而上的公式并没有“直接”解决这两个，那么是如何间接解决的呢？

XGboost优化策略: **贪心策略+最优化(二次最优化)**

通俗解释贪心策略：就是决策时刻按照当前目标最优化决定，说白了就是眼前利益最大化决定，“目光短浅”策略，他的优缺点细节大家自己去了解，经典背包问题等等。

这里是怎么用贪心策略的呢，刚开始你有一群样本，放在第一个节点，这时候$T=1$，$w$多少呢，不知道，是求出来的，这时候所有样本的预测值都是$w$（这个地方自己好好理解，决策树的节点表示类别，回归树的节点表示预测值）,带入样本的$label$数值，此时$loss\hspace{0.3cm}function$变为:
$$
L(\phi)=\sum_i{L(w-y_i)}+\gamma+\frac 12\lambda||w||^2
$$
这里$L(w-y_i)$误差表示用的是平方误差，那么上述函数就是一个关于$w$的二次函数求最小值，取最小值的点就是这个节点的预测值，最小的函数值为最小损失函数。

暂停下，这里你发现了没，二次函数最优化！
要是损失函数不是二次函数咋办，哦，泰勒展开式会否？，不是二次的想办法近似为二次。

着来，接下来要选个feature分裂成两个节点，变成一棵弱小的树苗，那么需要：
+ 确定分裂用的feature，how？最简单的是粗暴的枚举，选择loss function效果最好的那个（关于粗暴枚举，Xgboost的改良并行方式咱们后面看）；
+ 如何确立节点的w以及最小的loss function，大声告诉我怎么做？对，二次函数的求最值（细节的会注意到，计算二次最值是不是有固定套路，导数=0的点，ok）

那么节奏是，选择一个feature分裂，计算loss function最小值，然后再选一个feature分裂，又得到一个loss function最小值…你枚举完，找一个效果最好的，把树给分裂，就得到了小树苗。

在分裂的时候，你可以注意到，每次节点分裂，loss function被影响的只有这个节点的样本，因而每次分裂，计算分裂的增益（loss function的降低量）只需要关注打算分裂的那个节点的样本。

接下来，继续分裂，按照上述的方式，形成一棵树，再形成一棵树，每次在上一次的预测基础上取最优进一步分裂/建树，是不是贪心策略？！

凡是这种循环迭代的方式必定有停止条件，什么时候停止呢：
+ 当引入的分裂带来的增益小于一个阀值的时候，我们可以剪掉这个分裂，所以并不是每一次分裂loss function整体都会增加的，有点预剪枝的意思（其实我这里有点疑问的，一般后剪枝效果比预剪枝要好点吧，只不过复杂麻烦些，这里大神请指教，为啥这里使用的是预剪枝的思想，当然Xgboost支持后剪枝），阈值参数为γ正则项里叶子节点数T的系数（大神请确认下）；
+ 当树达到最大深度时则停止建立决策树，设置一个超参数max_depth，这个好理解吧，树太深很容易出现的情况学习局部样本，过拟合；
+ 当样本权重和小于设定阈值时则停止建树，这个解释一下，涉及到一个超参数-最小的样本权重和min_child_weight，和GBM的 min_child_leaf 参数类似，但不完全一样，大意就是一个叶子节点样本太少了，也终止同样是过拟合；
+ 貌似看到过有树的最大数量的…这个不确定


节点分裂的时候是按照哪个顺序来的，比如第一次分裂后有两个叶子节点，先裂哪一个？
答案：呃，同一层级的（多机）并行，确立如何分裂或者不分裂成为叶子节点

**看下Xgboost的一些重点**：
+ $w$是最优化求出来的，不是啥平均值或规则指定的，这个算是一个思路上的新颖吧；
+ 正则化防止过拟合的技术，上述看到了，直接loss function里面就有；
+ 支持自定义loss function，哈哈，不用我多说，只要能泰勒展开（能求一阶导和二阶导）就行，你开心就好；
+ 支持并行化，这个地方有必要说明下，因为这是xgboost的闪光点，直接的效果是训练速度快，boosting技术中下一棵树依赖上述树的训练和预测，所以树与树之间应该是只能串行！那么大家想想，哪里可以并行？！
	+ **没错，在选择最佳分裂点，进行枚举的时候并行！（据说恰好这个也是树形成最耗时的阶段）**
	+ **同层级节点可并行**。具体的对于某个节点，节点内选择最佳分裂点，候选分裂点计算增益用多线程并行。

较少的离散值作为分割点倒是很简单，比如“是否是单身”来分裂节点计算增益是很easy，但是“月收入”这种feature，取值很多，从5k~50k都有，总不可能每个分割点都来试一下计算分裂增益吧？（比如月收入feature有1000个取值，难道你把这1000个用作分割候选？缺点1：计算量，缺点2：出现叶子节点样本过少，过拟合）我们常用的习惯就是划分区间，那么问题来了，这个区间分割点如何确定（难道平均分割），作者是这么做的：

+ XGBoost还特别设计了针对稀疏数据的算法
假设样本的第i个特征缺失时，无法利用该特征对样本进行划分，这里的做法是将该样本默认地分到指定的子节点，至于具体地分到哪个节点还需要某算法来计算，

算法的主要思想是，分别假设特征缺失的样本属于右子树和左子树，而且只在不缺失的样本上迭代，分别计算缺失样本属于右子树和左子树的增益，选择增益最大的方向为缺失数据的默认方向（咋一看如果缺失情况为3个样本，那么划分的组合方式岂不是有8种？指数级可能性啊，仔细一看，应该是在不缺失样本情况下分裂后（有大神的请确认或者修正），把第一个缺失样本放左边计算下loss function和放右边进行比较，同样对付第二个、第三个…缺失样本，这么看来又是可以并行的？？）；
+ 可实现后剪枝
+ 交叉验证，方便选择最好的参数，early stop，比如你发现30棵树预测已经很好了，不用进一步学习残差了，那么停止建树。
+ 行采样、列采样，随机森林的套路（防止过拟合）
+ **Shrinkage**，你可以是几个回归树的叶子节点之和为预测值，也可以是加权，比如第一棵树预测值为3.3，label为4.0，第二棵树才学0.7，….再后面的树还学个鬼，所以给他打个折扣，比如3折，那么第二棵树训练的残差为4.0-3.3*0.3=3.01，这就可以发挥了啦，以此类推，作用是啥，防止过拟合，如果对于“伪残差”学习，那更像梯度下降里面的学习率；
+ xgboost还支持设置样本权重，这个权重体现在梯度g和二阶梯度h上，是不是有点adaboost的意思，重点关注某些样本。

Xgboost和深度学习的关系，陈天奇在Quora上的解答如下：
不同的机器学习模型适用于不同类型的任务。深度神经网络通过对时空位置建模，能够很好地捕获图像、语音、文本等高维数据。而基于树模型的XGBoost则能很好地处理表格数据，同时还拥有一些深度神经网络所没有的特性（如：模型的可解释性、输入数据的不变性、更易于调参等）。
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a
a