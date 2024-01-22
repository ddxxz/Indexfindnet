# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/13 9:20
@Auth ： dengxz
@File ：weight_indexfind_single.py
@IDE ：PyCharm
"""
# ...include code from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
import sys
from pathlib import Path
sys.path.append("/mnt/e/deep_learning/hyperspec_band_find/")
#import shap
import numpy as np
import torch
import os
import csv
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from data_proceed.data import BasicDataset
import matplotlib.pyplot as plt
from models.Model_indice import *
from models.Model_indexfind import *
from omegaconf import OmegaConf
from scipy import stats
from types import SimpleNamespace
base_path = '/mnt/e/deep_learning/outputs/outputs20231020/att235_resample_compress_mask_gumbel_attloss/'
model_results_path = '/mnt/e/deep_learning/outputs/outputs20231020/att235_resample_compress_mask_gumbel_attloss/model_results/Indexfind_onehot_adacos_ gumbel/base'
class RepeatedDataset(Dataset):
    def __init__(self, dataset, repeat=10000):
        self.dataset = dataset
        self.repeat = repeat

    def __getitem__(self, index):
        repeated_index = index % len(self.dataset)  # 通过取余操作来循环索引数据集
        return self.dataset[repeated_index]

    def __len__(self):
        return len(self.dataset) * self.repeat
# model = indice_CNNformer(torch.device('cpu'),204,indice_max,indice_min)
files=os.listdir(f'{model_results_path}')
for file in files:
    if Path(base_path).joinpath(file).is_file():
        continue
    conf_path = f'{model_results_path}/{file}/.hydra/config.yaml'
    args = OmegaConf.load(conf_path)
    args = SimpleNamespace(**args)
    print(file)
    print("resample:",args.resample,"compres:",args.compress_num)
    plt.rcParams['font.sans-serif']=['Times New Roman']#'SimHei',
    plt.rcParams['axes.unicode_minus'] = False
    #args.resample =60
    train_data = BasicDataset(args=args,datapath='data/wheat_rice_train.csv')
    # if args.task=='hyperimg':
    #     train_data.allhyperimg = (train_data.allhyperimg - train_data.allhyperimg_mean) / train_data.allhyperimg_std
    train_data.input_x_norm = (train_data.input_x_norm - train_data.input_x_mean) / train_data.input_x_std
    train_data.label_norm = (train_data.label_norm - train_data.label_mean) / train_data.label_std
    #train_data = RepeatedDataset(train_data, repeat=10000)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    val_data = BasicDataset(args=args,datapath='data/wheat_rice_val.csv')
    # if args.task == 'hyperimg':
    #     val_data.allhyperimg = (val_data.allhyperimg - val_data.allhyperimg_mean) / val_data.allhyperimg_std
    val_data.input_x_norm = (val_data.input_x_norm - train_data.input_x_mean) / train_data.input_x_std
    val_data.label_norm = (val_data.label_norm - train_data.label_mean) / train_data.label_std
    #val_data = RepeatedDataset(val_data, repeat=10000)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    #print(len(train_loader))
    test_data = BasicDataset(args=args,datapath='data/wheat_rice_test.csv')
    # if args.task == 'hyperimg':
    #     test_data.allhyperimg = (test_data.allhyperimg - test_data.allhyperimg_mean) / test_data.allhyperimg_std
    test_data.input_x_norm = (test_data.input_x_norm - train_data.input_x_mean) / train_data.input_x_std
    test_data.label_norm = (test_data.label_norm - train_data.label_mean) / train_data.label_std
    #test_data = RepeatedDataset(test_data, repeat=10000)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    inputs=[]
    for data in train_loader:
        # print(data)
        input = data['Spectral']
        inputs.append(input)
    x_train = torch.concat(inputs,dim=0)

    test_inputs=[]
    for data in test_loader:
        with torch.no_grad():
            # print(data)
            test_input = data['Spectral']
            test_inputs.append(test_input)
    val_inputs=[]
    for data in val_loader:
        with torch.no_grad():
            # print(data)
            val_input = data['Spectral']
            val_inputs.append(val_input)
            test_inputs.append(val_input)
    x_test = torch.concat(test_inputs,dim=0)#torch.concat(test_inputs,dim=0)x_train


    #model0 = str(file).split('_')[3:-2]
    #models = "_".join(model0)
    models='IndexfindNet_onehot'
    header_name = str(file).split('_')[3:]
    model_name = "_".join(header_name)
    print(models)
    # for i,m in enumerate(model0):
    #     mod =

   # print(models)
    path = base_path + '/' +'band_results/Indexfind_onehot_adacos_ gumbel/sorted_softmax/resampleplot_right' + '/' +model_name
    # 创建文件夹
    device = torch.device('cuda:0')
    os.makedirs(path, exist_ok=True)
    model = eval(models)(torch.device('cuda:0'),args.resample)
    model = model.to(device)
    checkpoint = torch.load(
        f'{model_results_path}/{file}/best_model_both.pt',
        map_location='cpu')

    model.load_state_dict(checkpoint['state'])
    model.eval()
    print(x_test.shape)
    maskeds=[]
    for i in range(x_test.shape[0]):#
        with torch.no_grad():
            x_test = x_test.to(device)
            _,weight,att,masked=model(x_test[i:i+1,:,:],return_weight=True)
        # _,weights = model(x_train,return_weight=True)
        # print(weights.shape)
        #print(x_test.shape)
        # plot_two('att_weight_two_cnnformer_wheat_rice1.png',x_test[0,:,:,0],weights.T,bands)
        # plot_one('att_weight_cnnformer_wheat_rice1.png',weights.T,bands)
        #print('weight num',len(weight))
        #weight = np.array(weight)
        #print('weight_shape',np.array(np.array(weight)[0]).shape)
        #names = model.get_index_find_name()
        names = ['Index_Add_mul','Index_Sub_sub','Index_Div']
        #print('att',att.cpu().numpy().shape)
        #print('masked',masked.shape)
        masked = masked.cpu().numpy()
        maskeds.append(masked)
        att = att.cpu().numpy()
        max_values = np.max(att, axis=1)  # 沿第二维度计算最大值
        
        #att_idx = np.argmax(att.cpu().numpy(),axis=1) # 1,64
        if len(np.unique(max_values)) == 1:
            print("第二维中的三个值都相等")
        else:
            att_idx = np.argmax(att, axis=1)  # 沿第二维度计算最大值索引
            all_index = []
            for c in range(64):#通道数
                att_idx_c = att_idx[0,c]
                weight_func = weight[att_idx_c]#哪个指数
                header = [f'channel{c} index is {names[att_idx_c]}']
                func_idxs = []
                for n in range(len(weight_func)):#4 , 2
                    weight_index = weight_func[n]
                    weight_index_channel = weight_index[...,c]
                    #print('weight_index_channel',weight_index_channel)
                    idx_c_n = np.argmax(np.array(weight_index_channel.cpu()),axis=1)
                    func_idxs.append(int(idx_c_n))
                if len(func_idxs) == 2 :
                    func_idxs.append(np.nan)
                    func_idxs.append(np.nan)
                func_idxs = header + func_idxs+[model_name]+ [names[att_idx_c]]
                #print(func_idxs)
                all_index.append(np.array(func_idxs))
            all_index = np.stack(all_index,axis=0)
            #print(all_index)
            #print(all_index.shape)
            data = pd.DataFrame(all_index)
    maskeds = np.squeeze(maskeds,axis=1)
    maskeds = np.squeeze(maskeds,axis=1)
    maskeds = np.transpose(maskeds,(1,0))
    maskeds = pd.DataFrame(maskeds)
    #data.to_csv(f'{path}/{model_name}_sample{i}.csv')
    maskeds.to_csv(f'/mnt/e/deep_learning/outputs/outputs20231020/att235_resample_compress_mask_gumbel_attloss/model_results/Indexfind_onehot_adacos_ gumbel/maskeds.csv')
    