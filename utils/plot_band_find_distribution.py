import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
config = {
    "font.family":'Times New Roman',
    "font.size": 18,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}

#plt.figure(dpi=500)
plt.rcParams.update(config)
# 定义文件夹路径

folder_path = f"/mnt/e/deep_learning/outputs/outputs20231020/att235_resample_compress_mask_gumbel_attloss/band_results/Indexfind_onehot_adacos_ gumbel/softmax/base"

# 获取所有文件夹
folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
folders[0], folders[1]= folders[1],folders[0]
fig, ax = plt.subplots(2,2,figsize=(12, 10.5),dpi=500)
# 遍历每个文件夹
n=0
for i in range(2):
    for j,folder in enumerate(folders):
        print('folder',folder)
        # 提取波段数
        waveband = 204#folder.split('_')[-3]
        # 定义存储数字的列表
        numbers = []
        indices = []
        # 获取当前文件夹路径
        folder_full_path = os.path.join(folder_path, folder)
        
        # 获取文件夹中所有csv文件
        csv_files = glob.glob(os.path.join(folder_full_path, "*.csv"))
        
        # 遍历每个csv文件
        for csv_file in csv_files:
            # 读取csv文件
            df = pd.read_csv(csv_file)
            
            # 提取3-6列的数字
            for col in df.columns[2:6]:
                for num in df[col]:
                    # 如果数字不是nan，则加入列表
                    if pd.notnull(num):
                        band = 397+num/int(waveband)*606
                        numbers.append(band)
            for col in df.columns[7:]:
                for index in df[col]:
                    indices.append(index)
        # 绘制波段分布图
        #print('numbers',numbers)
        if i == 0:
            print('shape',np.array(numbers).shape,waveband)
            ax[i,j].hist(numbers, bins=int(waveband),alpha=0.5,color='darkblue',density=True)
            ax[i,j].set_xlabel("Wavelength(nm)",fontweight='bold',fontsize=25)
            ax[i,0].set_ylabel("Frequency of bands for $\mathregular{V_{cmax}}$",fontweight='bold',fontsize=25)
            ax[i,1].set_ylabel("Frequency of bands for $\mathregular{J_{max}}$",fontweight='bold',fontsize=25)
            # 修改坐标轴字体及大小
            xticks = [400,500,600,700,800,900,1000]#ax[i,j].get_xticks()
            xticklabels = [400,500,600,700,800,900,1000]#ax[i,j].get_xticklabels()
            yticks = ax[i,j].get_yticks()
            yticklabels = ax[i,j].get_yticklabels()
            ax[i,j].set_yticks(yticks,labels=yticklabels,fontproperties='Times New Roman', size=18,weight='bold')#设置大小及加粗
            ax[i,j].set_xticks(xticks,labels=xticklabels,fontproperties='Times New Roman', size=18,weight='bold')
            #ax[i,j].tick_params(axis='x', labelfont={'family': 'Times New Roman', 'size': 18, 'weight': 'bold'})
            #ax[i,j].set_title(f"({chr(97+n)}) Distribution of Bands for " + folder.split('_')[-1],fontweight='bold',fontsize=25,pad=15)

        
        #plt.savefig(f'/mnt/e/deep_learning/outputs/outputs20231020/att235_resample_compress_mask_gumbel_attloss/plot_map/{folder}.png',bbox_inches='tight')
        ax[i,j].spines['bottom'].set_linewidth(2)  # 设置边框线宽为2.0
        ax[i,j].spines['right'].set_linewidth(2)
        ax[i,j].spines['left'].set_linewidth(2)
        ax[i,j].spines['top'].set_linewidth(2)
        #ax.set_xtitle([])
        ax[i,j].tick_params(bottom=True, top=False, left=True, right=False,width=1)
        # 绘制指数形式分布图
        #print(pd.Series(indices).unique())

        #print(indices)
        replacement_dict = {'Index_Sub_sub': 'Index-Sub-sub', 'Index_Div': 'Index-Div', 'Index_Add_mul': 'Index-Add-mul'}  # 替换映射字典
        # 替换列表中的字符
        colors = ['#2771A8','palevioletred','seagreen']
        indices = [replacement_dict.get(char, char) for char in indices]
        #print(indices)
        if i ==1:
            #print(indices)
            #counts, bins, _ = ax[i,j].hist(sorted(indices), bins=range(0, 4), width=0.5,edgecolor='black',alpha=0.8,density=True,align='right')
            #plt.bar(bins,counts,color=plt.get_cmap('Blues')(np.linspace(0, 2, 4)))
            # 计算频率分布的中心点
            #bin_centers = (bins[:-1] + bins[1:]) / 2
            counter = Counter(sorted(indices))
            indices = list(counter.keys())
            frequencies = list(counter.values())
            total_sum = sum(frequencies)
            result = [num / total_sum for num in frequencies]
            print(frequencies)

            ax[i,j].bar(indices, result,width=0.5,color=colors,alpha=0.25,edgecolor='black',align='center')


            #print(bin_centers)
            # 绘制折线图
            #ax[i,j].plot(bin_centers, counts, linestyle='-', marker='o', color='darkred',linewidth=3)
            #plt.hist(sorted(indices), bins=3,histtype='step')
            ax[i,j].set_xlabel("Vegetation Indices",fontweight='bold',fontsize=25)
            ax[i,0].set_ylabel("Frequency of Indices for $\mathregular{V_{cmax}}$",fontweight='bold',fontsize=25)
            ax[i,1].set_ylabel("Frequency of Indices for $\mathregular{J_{max}}$",fontweight='bold',fontsize=25)
            ax[i,j].spines['bottom'].set_linewidth(2)  # 设置边框线宽为2.0
            ax[i,j].spines['right'].set_linewidth(2)
            ax[i,j].spines['left'].set_linewidth(2)
            ax[i,j].spines['top'].set_linewidth(2)
            xticks = ax[i,j].get_xticks()
            #ax[i,j].set_xlim(0,3)
            #xticks=[0.5,1.5,2.5]
            xticklabels = ax[i,j].get_xticklabels()
            yticks = ax[i,j].get_yticks()
            yticklabels = ax[i,j].get_yticklabels()
            ax[i,j].set_yticks(yticks,labels=yticklabels,fontproperties='Times New Roman', size=18,weight='bold')#设置大小及加粗
            ax[i,j].set_xticks(xticks,labels=xticklabels,fontproperties='Times New Roman', size=18,weight='bold')
            #ax[i,j].set_yticks(ticks=indices,fontproperties='Times New Roman', size=18,weight='bold')#设置大小及加粗
            #ax[i,j].set_xticks(fontproperties='Times New Roman', size=18,weight='bold')
            ax[i,j].tick_params(bottom=True, top=False, left=True, right=False,width=1)
    
            #ax[i,j].set_title(f'({chr(97+n)}) Distribution of Index for ' + folder.split('_')[-1],fontweight='bold',fontsize=25,pad=15)
            
ax[0,0].text(400,0.032,f"({chr(65+0)})",fontweight='bold',fontsize=25)
ax[0,1].text(400,0.023,f"({chr(65+1)})",fontweight='bold',fontsize=25)
ax[1,0].text(-0.2,0.635,f"({chr(65+2)})",fontweight='bold',fontsize=25)
ax[1,1].text(-0.2,0.55,f"({chr(65+3)})",fontweight='bold',fontsize=25)

plt.tight_layout()
plt.savefig(f'base_index.tiff',bbox_inches='tight')
