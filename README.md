# 3D U-net For RibFrac Challenge

Author： 郭奕玮、郭子奥

## Guidelines

![img](https://github.com/cantabile-kwok/3D-U-net-for-RibFrac/blob/main/flowchart.png)

首先将数据集都按照`dataset`中的路径存放，csv文件和label一起放。

然后运行`newpreprocess.py`（train、val、test均要运行），这会生成`dataset/fixed_data`，存放处理过的数据。
