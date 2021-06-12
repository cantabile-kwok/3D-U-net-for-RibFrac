# 3D U-net For RibFrac Challenge

Author： 郭奕玮、郭子奥

## Guidelines

![img](https://github.com/cantabile-kwok/3D-U-net-for-RibFrac/blob/main/flowchart.png)

首先将数据集都按照`dataset`中的路径存放，csv文件和label一起放。

然后运行`newpreprocess.py`（train、val、test均要运行），这会生成`dataset/fixed_data`，存放处理过的数据。

随后运行`newtrain.py`，其中参数`upper, lower`是对图像进行上下截断的值，`save`指实验名称；通过调整`start_epoch`来设置是否热启动。

随后运行`make_prediction`，其中需要指定`exp_path`即实验目录，以及`raw_data_path`

随后运行`make_label.py`，也要指定`exp_path`。

最后运行`evaluate/ribfrac/evaluation.py`，其中`gt_dir` `pred_dir`为上一个脚本产生的label目录和原始数据集的label目录。

其中`FracNet`目录clone自https://github.com/M3DV/FracNet， `evaluate`目录clone自https://github.com/M3DV/RibFrac-Challenge
