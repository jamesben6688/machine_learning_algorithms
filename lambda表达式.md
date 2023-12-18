Lambda表达式
lambda args : return_val

lambda语句中，冒号前是参数，可以有多个，用逗号隔开，冒号右边是返回值，lambda语句构建的其实是一个函数对象。
```python
g = lambda x, y: x**y

g(2,3)
```