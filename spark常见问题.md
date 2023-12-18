###1. spark的worker无法启动

![](./pics/177.png)

需要在conf/spark-env.sh中添加
```
export SPARK_MASTER_HOST=master_ip
```

###2. pyspark与jupyter结合
编辑~/.bashrc，加入：
```
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook'
```