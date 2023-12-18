#<center>常见问题<center>#

1. yum损坏重装，安装下面这些文件
python-2.6.6-64.el6.x86_64.rpm
 python-iniparse-0.3.1-2.1.el6.noarch.rpm
 python-libs-2.6.6-64.el6.x86_64.rpm
 python-pycurl-7.19.0-9.el6.x86_64.rpm
 python-urlgrabber-3.9.1-11.el6.noarch.rpm
 rpm-python-4.8.0-55.el6.x86_64.rpm
```
rpm -ivh *.rpm
```
yum-3.2.29-73.el6.centos.noarch.rpm
yum-metadata-parser-1.1.2-16.el6.x86_64.rpm
yum-plugin-fastestmirror-1.1.30-37.el6.noarch.rpm
```
rpm -ivh yum-*
rpm -ivh yum-* --force --nodeps #如果提示缺少依赖的话就用这句
```
2. 安装软件时指定安装目录
```
./configure --prefix=path
```
3. 建立快捷方式
```
ln -s des src
```
4. centos升级Python找不到_tkinter
> 详情见[_tkinter安装]("http://www.tkdocs.com/tutorial/install.html")以及[_tkinter配置]("http://www.jb51.net/article/54153.htm")
+ 下载[ActiveTCL](https://www.activestate.com/activetcl/downloads)
+ 解压后进入目录 ./install安装tkl，注意设置好安装目录
+ 修改python安装目录下的 Modules/Setup.dist，添加下面内容
```
_tkinter _tkinter.c tkappinit.c -DWITH_APPINIT \
-L/usr/local/lib \
-I/usr/local/include \
-ltk8.5 -ltcl8.5 \
-lX11
```
其中lib和include要与安装目录对应，否则可能报错，后面的版本也要对应

+ 重新安装python3
```
./configure --prefix=path
make && make install
```