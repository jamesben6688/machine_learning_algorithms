#python opencv学习笔记#
1. 安装:
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
```

2. 打开并显示图片
	```python
	import cv2
	import matplotlib.pyplot as plt
	import numpy as np
	%matplotlib inline

	# 读取方式有cv2.IMREAD_COLOR(忽略透明度alpha通道), cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED(包含alpha通道)
	img = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)

	b, g, r = img[:,:0], img[:,:,1], img[:,:,2]  # opencv读入通道顺序是b, g, r
    img = np.stack([r,g,b], axis=-1)
    plt.show()
	```

3. 图片缩放
	```python
	img = cv2.resize(src_img, dst, dst_size=(), fx, fy, interpolation)
	```
参数:
+ src: 要缩放的图片
+ dst: 缩放后的图片(矩阵)
+ dst_size: 目标图像的大小，必须为tuple类型，如果是(0,0)，则计算方式为：(src_heigth*fx, width*fy)
+ interpolation：插值方式
	+ cv2.INTER_NEAREST: 最邻近插值
	+ cv2.INTER_LINEAR: 线性插值
	+ cv2.INTER_AREA: 
	+ cv2.INTER_CUBIC:
	+ cv2.INTER_LANCZOS4:

4. 