## 用于光合能力估计的全局光谱特征自动挖掘网络

Indexfindnet: 

基于全局注意力与光谱指数计算先验的光合能力估计模型。


以一维反射率输入，搭建深度学习模型，基准模型是采用一个一维卷积OnedCNN，它主要的结构是先对光谱维度池化：平滑，以防数据冗余，再利用两层空洞卷积扩大感受野。以这个基准模型进行稍微改进，搭建IndiceCNN，首先将平均池化分成三次进行，更加充分提取特征，然后，将第一层卷积替换为门控卷积选择特征，将第二层卷积替换为三层光谱指数计算层。

![图片2](https://github.com/ddxxz/hyperspec_one_alltry/blob/main/pic/图片2.png)

Indexfindnet的主要思想是想通过深度学习方法找出光合能力敏感的波段与植被指数特征。为了实现这一目标，我们将常见的光谱指数形式的计算公式嵌入模型当中M5，接着为了找出指数对应位置的敏感波段，我们结合注意力的思想，构造了NonlocalBandAttention模块M4，将对应的敏感波段找到，对于M4的输入，是经过M2和M3模块特征提取和特征筛选的反射率特征。


![图片4](https://github.com/ddxxz/hyperspec_one_alltry/blob/main/pic/图片4.png)

将得到的指数特征用于光合能力Vcmax,Jmax估计。

## 安装

```
git clone git@github.com:ddxxz/Indexfindnet.git
cd Indexfindnet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  
```

## 设置

### 配置

我们使用[Hydra](https://github.com/facebookresearch/hydra)来控制所有训练配置。如果您不熟悉 Hydra，我们建议您访问 Hydra[网站](https://hydra.cc/)。一般来说，Hydra 是一个开源框架，它通过提供动态创建分层配置的能力来简化研究应用程序的开发。

可以在该`conf`文件夹下找到包含训练模型的所有相关参数的配置文件。请注意，在该`conf`文件夹下，该`dset`文件夹包含不同数据集的配置文件。`config.yaml`您应该会看到一个以调试示例集的相关配置命名的文件。

您可以通过命令行传递选项，例如`python train.py lr=1e-4`. 请参阅[conf/config.yaml]以获取可能选项的参考。您也可以直接编辑该`config.yaml`文件，但不建议这样做，因为实验是自动命名的，如下文所述。

#### 划分数据集

```
python data_proceed/get_testdata.py 0 data/all_data.csv
python data_proceed/split-train-val-data.py 0
```

#### 训练

```
python main.py name=${crop}_IndexfindNet_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndexfindNet
```

