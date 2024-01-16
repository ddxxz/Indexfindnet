# Indexfindnet
## A network for automatic mining of global spectral features used for photosynthetic capacity estimation.

Indexfindnet: 

Photosynthetic capacity Estimation Model Based on Global Attention and Spectral Index Calculation Prior.


Building upon the one-dimensional reflectance input, a deep learning model called OnedCNN serves as the baseline model. It primarily consists of spectral dimension pooling, such as smoothing to prevent data redundancy, followed by two layers of dilated convolutions for expanded receptive field. Based on this baseline model, an improved version named IndiceCNN is constructed. It involves dividing average pooling into three rounds for more comprehensive feature extraction. Additionally, the first convolution layer is replaced with gated convolutions for feature selection, while the second convolution layer is substituted with three layers of spectral index calculation.

![图片2](https://github.com/ddxxz/hyperspec_one_alltry/blob/main/pic/图片2.png)

The main idea behind Indexfindnet is to use deep learning methods to identify the spectral bands and vegetation index features that are sensitive to photosynthetic capacity. To achieve this goal, we embed the commonly used computation formulas for spectral indices into the model, referred to as M5. To identify the sensitive bands corresponding to the indices, we combine the concept of attention and construct the NonlocalBandAttention module, denoted as M4. This module helps locate the corresponding sensitive bands. The input for M4 is the reflectance features that have undergone feature extraction and selection through modules M2 and M3.

![图片4](https://github.com/ddxxz/hyperspec_one_alltry/blob/main/pic/图片4.png)

The obtained index features are utilized for the estimation of photosynthetic parameters Vcmax and Jmax.

## Install

```
git clone git@github.com:ddxxz/Indexfindnet.git
cd Indexfindnet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  
```

## Configuration

### Setup

We utilize Hydra to manage all training configurations. If you are not familiar with Hydra, we recommend visiting the Hydra website. In general, Hydra is an open-source framework that simplifies the development of research applications by providing the capability to dynamically create hierarchical configurations.

You can find the configuration files containing all relevant parameters for training the models in the conf folder. Please note that within the conf folder, the dset folder contains configuration files for different datasets. In the config.yaml, you should see a file named after the corresponding configuration for a debugging sample set.

You can pass options through the command line, for example, python train.py lr=1e-4. Refer to [conf/config.yaml] for a reference of possible options. You can also directly edit the config.yaml file, but it is not recommended as experiments are automatically named, as mentioned below.

#### Data spliting

```
python data_proceed/get_testdata.py 0 data/all_data.csv
python data_proceed/split-train-val-data.py 0
```

#### Training

```
python main.py name=${crop}_IndexfindNet_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndexfindNet
```
