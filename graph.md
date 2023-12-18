```mermaid
graph TD
a(root)-- physician fee freeze=n -->b(b)
a(root)-- physician fee freeze=y -->c(c)
a(root)-- physician fee freeze=u -->d(d)
b-- adoption of the budget resolution=y -->e(democrat 151)
b-- adoption of the budget resolution=u -->f(democrat 1)
b-- adoption of the budget resolution=n -->g(g)
g-- education spending=n -->h(democrat 6)
g-- education spending=y -->i(democrat 9)
g-- education spending=u -->j(republican 1)
c-- synfuels corporation cutback=n -->k(republican 97/3)
c-- synfuels corporation cutback=n -->l(republican 4)
c-- synfuels corporation cutback=n -->m(m)
m-- duty free exports=y -->n(democrat 2)
m-- duty free exports=u -->o(republican 1)
m-- duty free exports=n -->p(p)
p-- education spending=n -->r(democrat 5/2)
p-- education spending=y -->s(republican 13/2)
p-- education spending=u -->t(democrat 1)
d-- water project cost sharing=n -->u1(democrat 0)
d-- water project cost sharing=y -->v1(democrat 4)
d-- water project cost sharing=u -->w1(u)
w1-- mx missile=n -->x(republican 0)
w1-- mx missile=y -->y(democrat 3/1)
w1-- mx missile=u -->z(republican 2)
```
u-- mx missile=n -->v(republican 0)
u-- mx missile=y -->w(democrat 3/1)
u-- mx missile=u -->x(republican 2)


graph TD
a(root)-- attr1=v1 p=0.4-->b(node1)
a(root)-- attr1=v2 p=0.3-->c(node2)
a(root)-- attr1=v3 p=0.3-->d(node3)
b--attr2=v4 p=0.6-->e(node4<br>label=yes<br>p=0.24)
b--attr2=v5 p=0.2-->f(node5<br>label=no<br>p=0.0.08)
b--attr2=v6 p=0.2-->g(node6<br>label=no<br>p=0.0.08)
c--attr2=v7 p=0.3-->h(node7<br>label=no<br>p=0.09)
c--attr2=v8 p=0.3-->i(node8<br>label=no<br>p=0.09)
c--attr2=v9 p=0.4-->j(node9<br>label=no<br>p=0.12)
d--attr2=v10 p=0.1-->k(node10<br>label=yes<br>p=0.03)
d--attr2=v11 p=0.4-->l(node11<br>label=yes<br>p=0.12)
d--attr2=v12 p=0.5-->m(node12<br>label=yes<br>p=0.15)

后面再继续分裂时,计算信息增益和增益率与上面的方法相同,只有样本个数中含有小数,计算过程此处省略。

graph TD
a(root)--sunny-->b(D1 	1	no<br>D2	0.125	no<br>D3	0.125	yes<br>D8	0.125	no<br>D9	0.125	yes<br>D10	0.125	yes<br>D11	0.125	yes)
a(root)-- overcast-->c(D7 	1	yes<br>D12 	1	yes<br>D13	1	yes<br>D2	0.375	no<br>D3	0.375	yes<br>D8	0.125	no<br>D9	0.375	yes<br>D10	0.125	yes<br>D11	0.375	yes)
a(root)--rain-->d(D4 	1	yes<br>D5 	1	yes<br>D6	1	no<br>D14	1	no<br>D2	0.5	no<br>D3	0.5	yes<br>D8	0.5	no<br>D9	0.5	yes<br>D10	0.5	yes<br>D11	0.5	yes)


graph TD
a(N个样本)-- attr1 < vi -->b(a个样本)
a-- attr1 >= attri -->c(b个样本)
b-- attr2=male-->d(c个样本)
b-- attr2=female-->e(d个样本)
d-- attr1 < vj -->f(e个样本)
d-- attr1 >= vj -->g(f个样本)