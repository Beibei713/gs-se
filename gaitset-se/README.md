# GaitSet

[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

- GaitSet is a **flexible**, **effective** and **fast** network for cross-view gait recognition. The [paper](https://ieeexplore.ieee.org/document/9351667) has been published on IEEE TPAMI. 
- GaitSet是一种灵活、有效、快速的跨视图步态识别网络。该论文已在IEEE TPAMI上发表。

#### Flexible  灵活
- The input of GaitSet is a set of silhouettes. GaitSet的输入是一组轮廓。

- There are **NOT ANY constrains** on an input,
which means it can contain **any number** of **non-consecutive** silhouettes filmed under **different viewpoints**
with **different walking conditions**.
- 输入没有任何限制，这意味着它可以包含在不同视点和不同行走条件下拍摄的任何数量的非连续轮廓。
- As the input is a set, the **permutation** of the elements in the input
will **NOT change** the output at all.
- 由于输入是一个集合，输入中元素的排列根本不会改变输出。

#### Effective  有效
It achieves **Rank@1=95.0%** on [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) 
and  **Rank@1=87.1%** on [OU-MVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html),
excluding  identical-view cases.

#### Fast   快速
With 8 NVIDIA 1080TI GPUs, it only takes **7 minutes** to conduct an evaluation on
[OU-MVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html) which contains 133,780 sequences
and average 70 frames per sequence.

## What's new
- The code and checkpoint for OUMVLP dataset have been released.
See [OUMVLP](#oumvlp) for details.
- OUMVLP数据集的代码和检查点已经发布。详见OUMVLP。

## Prerequisites

- Python 3.6
- PyTorch 0.4+
- GPU


## Getting started  开始使用
### Installation

- (Not necessary) Install [Anaconda3](https://www.anaconda.com/download/)
- Install [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive)
- install [cuDNN7.0](https://developer.nvidia.com/cudnn)
- Install [PyTorch](http://pytorch.org/)

Noted that our code is tested based on [PyTorch 0.4](http://pytorch.org/)

### Dataset & Preparation  数据集和准备
Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

Before training or test, please make sure you have prepared the dataset
by this two steps:
   在训练或测试之前，请确保您已经通过以下两个步骤准备了数据集：
- **Step1:** Organize the directory as:
`your_dataset_path/subject_ids/walking_conditions/views`.
E.g. `CASIA-B/001/nm-01/000/`.
- 将目录组织为：
- **Step2:** Cut and align the raw silhouettes with `pretreatment.py`.
(See [pretreatment](#pretreatment) for details.)
Welcome to try different ways of pretreatment but note that
the silhouettes after pretreatment **MUST have a size of 64x64**.
- 使用pretreatment.py剪切并对齐原始轮廓。（有关详细信息，请参阅预处理。）
欢迎尝试不同的预处理方法，但请注意，预处理后的轮廓大小必须为64x64。

- Futhermore, you also can test our code on [OU-MVLP Dataset](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html).
The number of channels and the training batchsize is slightly different for this dataset.
For more detail, please refer to [our paper](https://arxiv.org/abs/1811.06186).
- 此外，您还可以在OU-MVLP数据集上测试我们的代码。
此数据集的通道数量和训练批大小略有不同。有关更多详细信息，请参阅我们的论文。

#### Pretreatment   预处理
`pretreatment.py` uses the alignment method in
[this paper](https://ipsjcva.springeropen.com/articles/10.1186/s41074-018-0039-6).
Pretreatment your dataset by
  -  pretreatment.py使用了本文中的对齐方法。通过以下方式预处理数据集
```
python pretreatment.py --input_path='root_path_of_raw_dataset' --output_path='root_path_for_output'
```
- `--input_path` **(NECESSARY)** Root path of raw dataset.原始数据集的根路径。
- `--output_path` **(NECESSARY)** Root path for output.输出的根路径。
- `--log_file` Log file path. #Default: './pretreatment.log'log_file日志文件路径#默认值：'/预处理.log
- `--log` If set as True, all logs will be saved. 
Otherwise, only warnings and errors will be saved. #Default: False
- --log如果设置为True，则将保存所有日志。否则，只会保存警告和错误#默认值：False
- `--worker_num` How many subprocesses to use for data pretreatment. Default: 1
- --worker_num用于数据预处理的子流程数量。默认值：1 配置

### Configuration   配置

- In `config.py`, you might want to change the following settings:
- 在config.py中，您可能需要更改以下设置：
  - `dataset_path` **(NECESSARY)** root path of the dataset 
(for the above example, it is "gaitdata")
  - dataset_path（NECESSARY）数据集的根路径（对于上面的示例，它是“gaitdata”）
  - `WORK_PATH` path to save/load checkpoints
  -保存/加载检查点的WORK_PATH路径
  - `CUDA_VISIBLE_DEVICES` indices of GPUs
  - GPU的CUDA_VSIBLE_DEVICES索引

### Train
Train a model by
```bash
python train.py 
```
- `--cache` if set as TRUE all the training data will be loaded at once before the training start.
This will accelerate the training.
**Note that** if this arg is set as FALSE, samples will NOT be kept in the memory
even they have been used in the former iterations. #Default: TRUE
- --如果设置为TRUE，则缓存将在训练开始前一次加载所有训练数据。这将加速训练。
请注意，如果此参数设置为FALSE，则即使在前面的迭代中使用了样本，也不会将其保存在内存中#默认值：TRUE

### Evaluation
Evaluate the trained model by
```bash
python test.py
```
- `--iter` iteration of the checkpoint to load. #Default: 80000  迭代要加载的检查点#默认值：80000
- `--batch_size` batch size of the parallel test. #Default: 1   并行测试的批量大小#默认值：1
- `--cache` if set as TRUE all the test data will be loaded at once before the transforming start.
This might accelerate the testing. #Default: FALSE
- 如果设置为TRUE，则缓存将在转换开始前一次加载所有测试数据。这可能会加速测试#默认值：FALSE

It will output Rank@1 of all three walking conditions. 
Note that the test is **parallelizable**. 
To conduct a faster evaluation, you could use `--batch_size` to change the batch size for test.
它将输出Rank@1在所有三种行走条件下。请注意，该测试是可并行的。
为了进行更快的评估，您可以使用--batch_size更改测试的批大小。

#### OUMVLP
Since the huge differences between OUMVLP and CASIA-B, the network setting on OUMVLP is slightly different.
由于OUMVLP和CASIA-B之间的巨大差异，OUMVLP上的网络设置略有不同。
- The alternated network's code can be found at `./work/OUMVLP_network`. Use them to replace the corresponding files in `./model/network`.
- The checkpoint can be found [here](https://1drv.ms/u/s!AurT2TsSKdxQuWN8drzIv_phTR5m?e=Gfbl3m).
- 替换网络的代码可以在上找到/work/OUVLP_network。使用它们替换中的相应文件/模型/网络。
- In `./config.py`, modify `'batch_size': (8, 16)` into `'batch_size': (32,16)`.
- Prepare your OUMVLP dataset according to the instructions in [Dataset & Preparation](#dataset--preparation).
- 在/config.py，将“batch_size”修改为“batch-size”：（32,16）。
- 根据数据集和准备中的说明准备OUMVLP数据集。


## To Do List   待办事项列表 
- Transformation: The script for transforming a set of silhouettes into a discriminative representation.

## Authors & Contributors   作者和贡献者
GaitSet is authored by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/), 
[Yiwei He](https://www.linkedin.com/in/yiwei-he-4a6a6bbb/),
[Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/)
and JianFeng Feng from Fudan Universiy.
[Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/)
is the corresponding author.
The code is developed by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/)
and [Yiwei He](https://www.linkedin.com/in/yiwei-he-4a6a6bbb/).
Currently, it is being maintained by
[Hanqing Chao](https://www.linkedin.com/in/hanqing-chao-9aa42412b/)
and Kun Wang.


## Citation
Please cite these papers in your publications if it helps your research:
```
@ARTICLE{chao2019gaitset,
  author={Chao, Hanqing and Wang, Kun and He, Yiwei and Zhang, Junping and Feng, Jianfeng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={GaitSet: Cross-view Gait Recognition through Utilizing Gait as a Deep Set}, 
  year={2021},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3057879}}
```
Link to paper:
- [GaitSet: Cross-view Gait Recognition through Utilizing Gait as a Deep Set](https://ieeexplore.ieee.org/document/9351667)


## License
GaitSet is freely available for free non-commercial use, and may be redistributed under these conditions.
For commercial queries, contact [Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/).
