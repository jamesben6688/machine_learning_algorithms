#<center>scikit-image学习笔记</center>#
1. 导包

```python
import skimage.io as io
import matplotlib.pyplot as plt

img = io.imread("lena.jpg")

io.imshow(img)
plt.show()
```
io.imread_collection()：读取一批图片
```python
imgs = io.imread_collection(""../input/*.jpg) # 读入一批图像

for i, img in enumerate(imgs):
	pass
    # 取出每个图像
    io.imshow(img)

plt.imshow()
```

2. 缩放

```python
from skimage import io, transform

img = io.imread("lean.jpg")
img_1 = transform.rescale(img, scale=(0.5, 0.5))
io.imshow(img_1)
plt.show()
```

函数原型：

```python
skimage.transform.rescale(image, scale, order=1, mode=None, cval=0, clip=True, preserve_range=False)
```

参数：
+ image:要缩放的图片
+ scale：float或者float的tuple(row_scale, col_scale)，缩放的比例
+ order：插值顺序,默认为1，范围为(0, 5)。0表示Nearest-neighbor, 1表示Bi-linear插值法,2表示Bi-quadratic插值法, 3表示Bi-cubic插值法, 4表示Bi-quartic插值法, 5表示Bi-quintic插值法。
+ mode: 对于边界外的点的填充模式。有{"constant", "edge", "symmetric", "reflect", "wrap"}几种方式。默认为"constant"。
+ cval: 浮点型，与mode="constant"结合使用，指定边界外的点的值。
+ clip: True/False,是否将输出图像弄到输入图像的范围。默认True,因为高阶插值法可能会产生输入范围以外的值。
+ preserve_range: 是否保持原来的价值范围。 否则，根据img_as_float的约定转换输入图像。

返回值:
scaled_img: ndarray类型，缩放后的图像

```python
from skimage import io, transform

img = io.imread("lean.jpg")
img_1 = transform.rescale(img, outputshape=(256, 256))
io.imshow(img_1)
plt.show()
```
将图像缩放到指定的大小，功能类似于rescale函数，参数也类似。

3. 旋转

```python
from skimage import io, transform

img = io.imread("lean.jpg")
img_1 = transform.rotate(img, angle=90, resize=True)
io.imshow(img_1)
plt.show()
```
函数原型：
transform.rotate(image, angle, resize=False, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=False)
参数：
+ image: 要旋转的图像
+ angle：旋转的角度，逆时针方向
+ resize: 确定输出图像的形状是否被自动计算，因此完全旋转的图像是完全适合的。 默认值为False，即旋转后的图像所有的像素都不会丢失。
+ center:旋转中心。如果center为None,则默认为图像中心,center=(rows/2-0.5, cols/2-0.5)
+ order, mode, cval, clip, preserve_range参数和上面相同。
数据增强变换（Data Augmentation Transformation）
不同的任务背景下, 我们可以通过图像的几何变换, 使用以下一种或多种组合数据增强变换来增加输入数据的量. 这里具体的方法都来自数字图像处理的内容, 相关的知识点介绍, 网上都有, 就不一一介绍了．

4. 翻转
```python
from skimage import io, transform

img = io.imread("lean.jpg")
img_1 = img[:,::-1] # 左右翻转, -1表示从右往左，步长为1进行切片
img_2 = img[::-1,:]
```

5. 
旋转 | 反射变换(Rotation/reflection): 随机旋转图像一定角度; 改变图像内容的朝向;
翻转变换(flip): 沿着水平或者垂直方向翻转图像;
缩放变换(zoom): 按照一定的比例放大或者缩小图像;
平移变换(shift): 在图像平面上对图像以一定方式进行平移;
可以采用随机或人为定义的方式指定平移范围和平移步长, 沿水平或竖直方向进行平移. 改变图像内容的位置;
尺度变换(scale): 对图像按照指定的尺度因子, 进行放大或缩小; 或者参照SIFT特征提取思想, 利用指定的尺度因子对图像滤波构造尺度空间. 改变图像内容的大小或模糊程度;
对比度变换(contrast): 在图像的HSV颜色空间，改变饱和度S和V亮度分量，保持色调H不变. 对每个像素的S和V分量进行指数运算(指数因子在0.25到4之间), 增加光照变化;
噪声扰动(noise): 对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声;
颜色变换(color): 在训练集像素值的RGB颜色空间进行PCA, 得到RGB空间的3个主方向向量,3个特征值, p1, p2, p3, λ1, λ2, λ3. 对每幅图像的每个像素

$$
{I_x}y = {[I^R{xy}, I^G{xy}, I^B{xy}]}^T
$$
进行加上如下的变化:

PCA
Auto-encoding
Transform's such as log, powers, etc.
Binning continuous variables into discrete categories (i.e., continuous variable is 1 SD above mean, 1 below mean, etc.)
Composite variables (for example, see here)


```python
def random_zoom(img, mask, zoom_range=(0.8, 1), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = zoom(img, zx, zy)
        mask = zoom(mask, zx, zy)
    return img, mask


def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


# 图片扭曲
def random_shear(img, mask, intensity_range=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        img = shear(img, sh)
        mask = shear(mask, sh)
    return img, mask
```

饱和度变化
```python
def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
    return img
```