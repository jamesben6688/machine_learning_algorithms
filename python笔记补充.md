#<center>python补充</center>#
1. 偏函数
```python
from functools import partial
funcB = partial(funcA, para1 = para1)
```
有些时候为了方便调用，函数又没法设置一个默认参数，因此就可以使用偏函数：
	```python
	def funcA(para1, para2):
		pass

	funcA(para1, para2)
	funcA(para1, para3)
	funcA(para1, para4)
	...
	```
这时为了方便调用，就可以设置一个偏函数:
	```python
	funcB = partial(funcA, para1=para1)

	funcB(para2)
	funcB(para3)
	funcB(para4)
	```
2. zip函数
zip函数接受任意多个（包括0个和1个）序列作为参数，返回一个tuple列表。

	```python
	x = [1,2,3]
	y = [4,5,6]
	z = [7,8,9]
	xyz = zip(x,y,z)

	# xyz的值为: [(1,4,7),(2,5,8),(3,6,9)]

    x = [1, 2, 3]
	y = [4, 5, 6, 7]
	xy = zip(x, y)
    # xy的值为: [(1, 4), (2, 5), (3, 6)]

    x = [1, 2, 3]
	x = zip(x)
    # x的值为: [(1,),(2,),(3,)]
    ```

3. zip（*）函数
一般认为zip(*)是一个unzip的过程。在函数调用中使用*list/tuple的方式表示将list/tuple分开，作为位置参数传递给对应函数（前提是对应函数支持不定个数的位置参数）。
```python
	x = [1,2,3]
	y = [4,5,6]
	z = [7,8,9]
	xyz = zip(x,y,z)

    u=zip(*xyz)
    # u的值为: [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
```