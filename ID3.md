#<div align="center">ID3（Iterative Dichotomiser 3）算法详解</div>#

##1.信息熵
   熵这个概念最早起源于物理学，在物理学中是用来度量一个热力学系统的无序程度，而在信息学里面，熵是对不确定性的度量。在1948年，香农引入了信息熵，将其定义为离散随机事件出现的概率，一个系统越是有序，信息熵就越低，反之一个系统越是混乱，它的信息熵就越高。所以信息熵可以被认为是系统有序化程度的一个度量。

假设变量$X$的随机取值为$X = ${$x_1, x_2,..., x_n$},每一种取到的概率分别是{$p_1, p_2, p_3,...p_n$},则变量$X$的熵为:
$$H(X) = -\sum_{i=1}^{n}p_ilog_2p_i$$
>意思就是一个变量的变化情况越多，那么信息熵越大越不稳定。

##2.信息增益
信息增益针对单个特征而言,即看一个特征t,系统有它和没有它时信息熵之差。下面是weka中的一个数据集,关于不同天气是否打球的例子。特征是天气,label是是否打球。

|outlook |temperature|humidity|windy|play|
|:------:|:---------:|:------:|:---:|:--:|
|sunny   |hot        |high    |FALSE|no  |
|sunny   |hot        |high    |TRUE |no  |
|overcast|hot        |high    |FALSE|yes |
|rainy   |mild       |high    |FALSE|yes |
|rainy   |cool       |normal  |FALSE|yes |
|rainy   |cool       |normal  |TRUE |no  |
|overcast|cool       |normal  |TRUE |yes |
|sunny   |mild       |high    |FALSE|no  |
|sunny   |cool       |normal  |FALSE|yes |
|rainy   |mild       |normal  |FALSE|yes |
|sunny   |mild       |normal  |TRUE |yes |
|overcast|mild       |high    |TRUE |yes |
|overcast|hot        |normal  |FALSE|yes |
|rainy   |mild       |high    |TRUE |no  |

共有14个样本，9个正样本(yes)5个负样本(no)，信息熵为:
$$
Entropy(S) = -\frac 9{14}log_2 \frac 9{14}-\frac 5{14}log_2 \frac 5{14}=0.940286
$$
接下来会遍历outlook, temperature, humidity, windy四个属性，求出用每个属性划分以后的信息熵假设以outlook来划分,此时只关心outlook这个属性，而不再关心其他属性:

~~~mermaid
graph TD
A(outlook)--sunny-->b[yes<br>yes<br>no<br>no<br>no]
A(outlook)-- overcast-->c[yes<br>yes<br>yes<br>yes]
A(outlook)--rainy-->d[yes<br>yes<br>yes<br>no<br>no]
~~~
此时的信息熵为:
$$
Entropy(sunny) = -\frac 2{5}log_2 \frac 2{5}-\frac 3{5}log_2 \frac 3{5}=0.970951
$$

$$
Entropy(overcast) = -\frac 4{4}log_2 \frac 4{4}-0\times log_2 0=0
$$

$$
Entropy(rainy) = -\frac 2{5}log_2 \frac 2{5}-\frac 3{5}log_2 \frac 3{5}=0.970951
$$

总的信息熵为
$$
Entropy = \sum_{t_i=t_0}^{t_n}P(t=t_i)Entropy(T=t_i)
$$

即
$$
Entropy(S|outlook) = P(outlook=sunny)\times Entropy(sunny)+
P(outlook=overcast)\times Entropy(overcast) + P(outlook=rainy)\times Entropy(rainy)=0.693536
$$

$Entropy(S|outlook)$指的是选择属性$Outlook$作为分类条件的信息熵,最终属性$Outlook$的信息增益为:
$$
IG(outlook) = Entropy(S) - Entropy(S|outlook) = 0.24675
$$
>IG：Information Gain(信息增益)

同理可以计算选择其他分类属性的信息增益，选择信息增益最大的属性作为分类属性。分类完成之后，样本被分配到3个叶子叶子节点：

|outlook |temperature|humidity|windy|play|
|:------:|:---------:|:------:|:---:|:--:|
|sunny   |hot        |high    |FALSE|no  |
|sunny   |hot        |high    |TRUE |no  |
|sunny   |mild       |high    |FALSE|no  |
|sunny   |cool       |normal  |FALSE|yes |
|sunny   |mild       |normal  |TRUE |yes |

|outlook |temperature|humidity|windy|play|
|:------:|:---------:|:------:|:---:|:--:|
|overcast|mild       |high    |TRUE |yes |
|overcast|hot        |normal  |FALSE|yes |
|overcast|cool       |normal  |TRUE |yes |
|overcast|hot        |high    |FALSE|yes |

|outlook |temperature|humidity|windy|play|
|:------:|:---------:|:------:|:---:|:--:|
|rainy   |mild       |high    |TRUE |no  |
|rainy   |mild       |normal  |FALSE|yes |
|rainy   |mild       |high    |FALSE|yes |
|rainy   |cool       |normal  |FALSE|yes |
|rainy   |cool       |normal  |TRUE |no  |
当子节点只有一种$label$时分类结束。若子节点不止一种$label$，此时再按上面的方法选用其他的属性继续分类，直至结束。

##3.ID3算法总结
$$
IG(S|t) = Entropy(S)-\sum_{value(T)}\frac {|S_v|}{S}Entropy(S_v)
$$
>IG: Information Gain(信息增益)

其中$S$为全部样本集合，$value(T)$属性$T$的所有取值集合，$v$是$T$的其中一个属性值，$S_v$是$S$中属性$T$的值为v的样例集合，$|S_v|$为$S_v$中所含样例数。在决策树的每一个非叶子结点划分之前，先计算每一个属性所带来的信息增益，选择最大信息增益的属性来划分，因为信息增益越大，区分样本的能力就越强。

**注意: ID3只能正对nominal attribute，即标称属性**






































