<center>
  <h1>Readme</h1>
</center>

[toc]

本次实验为 **中-英机器翻译**

任务为采用 $\text{Seq2Seq}$ 模型 + $\text{Attention}$ 机制实现中文语句到英文语句的翻译



## Install

本次实验支持在 $GPU$ 上进行模型的训练与预测，若想使用 $GPU$ 进行任务执行，请确保您已经安装好正确的显卡驱动与对应版本的 $CUDA$ 和 $cudnn$

本次项目所依赖的第三方库及其版本信息存储在 `requirements.txt` 文件中

可在终端使用命令

```bash
pip install -r requirements.txt
```

在新的环境中部署项目时安装对应的第三方库



## Usage

项目代码由三个 $py$ 文件构成

- `datapreprocessing`：数据预处理，包含了
  - 数据清洗
  - 分词
  - 中英词典构建
  - 文本编码
  - 数据加载器构建
- `model`：网络模型架构的具体实现
  - $\text{Seq2Seq}$ 模型：以 $\text{GRU}$ 为基础构建的 $\text{Encoder}$ 和 $\text{Decoder}$
  - 手动的 $\text{Attention}$ 机制的实现（包含三种不同的对齐函数）
  - 两种不同的训练策略（$\text{Teacher Forcing}$ 和 $\text{Free Running}$ ）
  - 两种不同的解码策略（$\text{Greedy Search}$ 和 $\text{Beam Search}$）
- `FER`：模型训练过程、验证过程、测试过程与 $\text{BLEU}$ 评价指标



由于本次实验采用的数据集为 `./data_short/nmt/en-cn`

- 训练集：`train.txt`
- 验证集：`dev.txt`
- 测试集：`test.txt`



**代码执行方法**

```bash
python3 main.py
```



## Documents instruction

`./Checkpoint/translate_model_FR.pt`：使用训练策略 $\text{Free Running}$ 训练好的模型 $\text{checkpoint}$

`./Checkpoint/translate_model_TF.pt`：使用训练策略 $\text{Teacher Forcing}$ 训练好的模型 $\text{checkpoint}$

`./Image/FR_50.png`：使用训练策略 $\text{Free Running}$ 模型每一迭代轮次的训练损失和验证损失构成的曲线图

`./Image/TF_50.png`：使用训练策略 $\text{Teacher Forcing}$ 模型每一迭代轮次的训练损失和验证损失构成的曲线图

`./Image/attention_1.png`：$\text{Example 1}$ 中为进行归一化的 $\text{Attention}$ 热力图  

`./Image/attention_1_scaling`：$\text{Example 1}$ 中进行归一化的 $\text{Attention}$ 热力图  

`./Image/attention_2_scaling.png`：$\text{Example 2}$ 中为进行归一化的 $\text{Attention}$ 热力图  