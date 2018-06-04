# FashionAI服饰关键点定位全球挑战赛 *AILAB-ZJU* 源码（持续更新最新实验）
* 比赛名称：天池FashionAI服饰关键点定位全球挑战赛
* 队伍名称：AILAB-ZJU
* 第一赛季排名：70/2322
* 第二赛季排名：41/2322
* 最好成绩：第二赛季B榜NE=4.45%

## 结果展示
![blouse](https://github.com/shaoniangu/Realize_Convolutional_Pose_Machines_On_FashionAI/raw/master/Readme_images/blouse.jpg)
![dress](https://github.com/shaoniangu/Realize_Convolutional_Pose_Machines_On_FashionAI/raw/master/Readme_images/dress.jpg)
![outwear](https://github.com/shaoniangu/Realize_Convolutional_Pose_Machines_On_FashionAI/raw/master/Readme_images/outwear.jpg)
![skirt](https://github.com/shaoniangu/Realize_Convolutional_Pose_Machines_On_FashionAI/raw/master/Readme_images/skirt.jpg)
![trousers](https://github.com/shaoniangu/Realize_Convolutional_Pose_Machines_On_FashionAI/raw/master/Readme_images/trousers.jpg)

## 代码环境说明
我们的代码运行环境为：
* 操作系统：Ubuntu16.04 LTS
* Python版本：3.5.1
* Opencv版本：3.3.0.10
* Tensorflow版本：1.3.0
* 必要的库函数：pickle、pandas、numpy、math、os、sys、matplotlib、random、time、skimage、scipy、PIL、importlib、configparser、imageio
* 使用的Baseline模型：Convolutional Pose Machines
* Baseline模型原作者版权：Apache License 2.0
* Baseline模型原作者代码地址：https://github.com/timctho/convolutional-pose-machines-tensorflow
* 使用时请遵循Apache License 2.0规定，在此仅供分享学习。

## 代码结构说明
整套代码文件由Results、Train、Test三部分构成，其中Results为存放gt结果和评测代码的文件夹、Train为训练代码文件、Test为测试代码文件，结构展示如下：
```
|--data
|--Train
   |--no occlusion
	 |--models
	 |--preprocess
	 |--config.py
	 |--run_training.py
   |--其它实验文件类同
|--Test
   |--normal_test
	 |--checkpoints documents
	 |--models
	 |--preprocess
	 |--config.py
	 |--test_config.py
	 |--run_test.py
   |--test_add_aug
	 |--checkpoints documents
	 |--models
	 |--preprocess
	 |--config.py
	 |--test_config.py
	 |--run_test.py
   |--test_add_aug_with_IOU
	 |--checkpoints documents
	 |--models
	 |--preprocess
	 |--config.py
	 |--test_config.py
	 |--run_test.py
|--README.md
```
**详细说明**：
* **Train**：包括多个实验文件，每个文件夹包含一种实验代码，文件名即实验名称。每个文件均由*models*、*preprocess*、*config.py*和*run_training*四个部分构成，其中*models*中存放的是模型文件，*preprocess*中存放的是数据生成器的py文件，*config.py*是训练的配置文件，*run_training*为训练的执行文件。
* **Test**：包含三种test模式，分别是常规test、带增广的test、结合衣服面积占比增广的test，文件名称分别为*normal_test*、*test_add_aug*、*test_add_aug_with_IOU*。测试文件均由*checkpoints*、*models*、*preprocess*、*config.py*、*test_config.py*和*run_test*六个部分构成，其中“checkpoints”中存放的是训练生成的模型参数的checkpoint文件夹，需从Train中生成并拷贝过来，models中存放的是模型文件，preprocess中存放的是测试图片预处理的py文件，*config.py*是训练的配置文件（测试时需要搭建训练时的模型结构），*test_config.py*是测试的配置文件，*run_test.py*为测试的执行文件。

## 训练流程
* 修改每个实验文件中的*preprocess/config.cfg*中的两个路径，要修改的路径为```train_data_file```和```img_directory```，分别为训练图片的标签csv文件路径和训练图片路径。
* 配置*config.py*中的训练参数。
* 修改*run_training.py*中一些关于路径保存的代码，自定义自己的路径结构。
* 格外注意：因为一些原因暂不能公开我们用的学习率，其他学习率例如衰减学习率均可，请在要调用的模型文件中完善build_loss2()、build_loss3()、build_loss4()中的学习率，或在run_training.py中直接打开注释，使用build_loss()。
* 运行*run_training.py*训练模型，每个模型运行生成的checkpoint将会保存在新生成包含有checkpoints的文件。

## 测试流程
* 请将训练生成的保存有checkpoint的文件拷贝至Test中文件夹下。
* 同样需要修改路径，一处为/preprocess/test_preprocess_config.py，一处为*Valid_config.py*，两个文件中要修改的路径均是```test_img_directory```和```test_data_file```，分别为测试图片的文件夹路径和测试图片标签csv文件的路径。
* 运行测试文件中的/preprocess/test_preprocess.py，进行测试数据的预处理，将测试图片中心padding到网络输入的大小。
* 此外还要修改*Valid_config.py*中预处理好的图片路径地址以及checkpoint文件地址。
* 修改*run_test.py*中一些关于路径保存的代码，自定义自己的路径结构。
* 运行*run_test.py*进行测试,测试后会生成一个新的csv的文件，此文件中保存的便是预测的标签。