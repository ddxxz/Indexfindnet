import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2
from scipy import signal
#from librosa import filters
from torch.utils.data import DataLoader
#from librosa import filters,mel_frequencies,fft_frequencies,power_to_db
#import librosa
import scipy
#from utils.RandomErasing import RandomErasing
import h5py


def Grunwald_Letnikov(n,ref):
    import differint.differint as df
    import numpy as np
    def f(idx):
        idx = int(idx)
        return ref[idx]
    x = list(range(len(ref)))
    GL_differint = df.GL(n, f, x[0], x[-1], len(x))
    return GL_differint


class BasicDataset(Dataset):
    def __init__(self,args,datapath,train_flag=False):
        self.train_flag = train_flag
        self.datapath = datapath
        self.task = args.task
        self.label_name = args.label_name
        self.gl = args.gl
        self.resample = args.resample
        self.compress_num = args.compress_num
        # print(datapath)
        data = pd.read_csv(datapath)
        self.embedding = args.embedding
        #print('embedding',self.embedding)
        if self.task != 'hyper':
            self.rgb_path = data['rgb_path']
            self.hyperimg_path = data['hyperimg_path']
            data = data.drop('rgb_path', axis=1)
        #print(self.rgb_path)
        if self.embedding:
            self.embed_path = data['hyperimg_feature_path']
            data = data.drop('hyperimg_feature_path', axis=1)
            # self.embed_path = data['ref_feature_path']
            # data = data.drop('ref_feature_path',axis=1)

        self.input_x = data.loc[:,'397.32':'1003.58']
        input_x = self.input_x.values

        #单分数阶光谱输入
        if self.gl!=0:
            input_x = [Grunwald_Letnikov(self.gl,x) for x in input_x]
        input_x = np.array(input_x)
        if self.resample!=204:
            input_x = signal.resample(input_x,num=self.resample,axis=-1)#重采样光谱
        if self.compress_num!=1:
            input_x = np.power(input_x,self.compress_num)#np.sqrt(input_x)#对光谱压缩
        #input_x = np.sqrt(input_x)
        print(input_x.shape)


        self.input_x_mean = np.mean(input_x)
        self.input_x_std = np.std(input_x)
        self.input_x_norm = input_x

        
        self.label = data.loc[:,[self.label_name]]#,,,'Vcmax','Jmax',
        #label = self.label.values
        self.label_mean = self.label.mean(axis=0).values.astype(np.float32)#np.mean(label,axis=0,keepdims=True)
        self.label_mean = np.array(self.label_mean)
        self.label_std =  self.label.std(axis=0).values.astype(np.float32)          #np.std(label,axis=0,keepdims=True)
        self.label_std = np.array(self.label_std)
        self.label_norm = self.label
        #self.label_norm = (label-label_mean)/label_std
        # print(self.label)
        self.Vcmax_max = np.max(np.array(self.label)[:, 0])
        self.Vcmax_min = np.min(np.array(self.label)[:, 0])

        self.preload = args.preload


    def __len__(self):
        return len(self.label_norm)
    def norm(self,rgb):
        rgb = rgb / 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        rgb = (rgb - mean) / std
        return rgb

    def paddle(self,x, w=512, h=80):
        if x.shape[1] % 2 == 0:
            pad_w = [(w - x.shape[1]) // 2, (w - x.shape[1]) // 2]
        else:
            pad_w = [(w - x.shape[1]) // 2 + 1, (w - x.shape[1]) // 2]
        if x.shape[2] % 2 == 0:
            pad_h = [(h - x.shape[2]) // 2, (h - x.shape[2]) // 2]
        else:
            pad_h = [(h - x.shape[2]) // 2 + 1, (h - x.shape[2]) // 2]
        #[1,1],[2,2]
        #[ [0, 0], [1,1],[2,2]]
        x = np.pad(x, [ [0, 0], pad_w, [0, 0]])
        return x

    def __getitem__(self, item):
        input_x = self.input_x_norm[item]

        label = self.label_norm.iloc[item]
        input_x = input_x[None, :]   #用一维反射率只用这一行
        #print(input_x.shape)


        #print(str(self.rgb_path[item]))
        if self.task == 'hyper':
            rgb = np.zeros([3,64,64])

        elif self.task == 'xiaobo':
            rgb = np.load(self.hyperimg_path[item])
        elif self.task == 'hyperimg':
            #
            if self.preload:
                rgb=self.allhyperimg[item]
                #print(rgb.shape)

            else:
                rgb = h5py.File(self.hyperimg_path[item], 'r')
                rgb = rgb['data'][:]  # 取出主键为data的所有的键值
                #rgb.close()
                rgb = rgb.transpose((2, 0, 1))
                # print('rgb',rgb.shape)
                c, w, h = rgb.shape
                rgb = np.reshape(rgb, [c // 4, 4, w, h])
                rgb = np.mean(rgb, axis=1)
                rgb = self.paddle(rgb)
                # print('pad',rgb.shape)


        else:
            if self.embedding:
                embedding = np.load(self.embed_path[item])
                rgb = cv2.imread(str(self.rgb_path[item]))
                #print(np.array(rgb).sum())
            else:
                #print(self.rgb_path[item])
                rgb = cv2.imread(str(self.rgb_path[item]))
                #RE = RandomErasing(p=1)
                #RE_rgb = RE(rgb.copy())
                #print(rgb.shape)
                #rgb = cv2.resize(rgb,(64,64))
                #rgb = self.norm(rgb)
                #RE_rgb = self.norm(RE_rgb)
                rgb = rgb.transpose((2, 0, 1))
                #RE_rgb = RE_rgb.transpose((2, 0, 1))

        if self.embedding:
            dataset = {'Spectral':torch.from_numpy(np.array(input_x,dtype='float32')),
                       #'Indice': torch.from_numpy(np.array(indice, dtype='float32')),
                       #'Embedding':torch.from_numpy(np.array(embedding,dtype='float32')),
                       #'himg': torch.from_numpy(np.array(rgb,dtype='float32')),
                       #'RE_RGB':torch.from_numpy(np.array(RE_rgb)),
                       'Label':torch.from_numpy(np.array(label,dtype='float32'))
            }
        else:
            dataset = {'Spectral': torch.from_numpy(np.array(input_x, dtype='float32')),
                       #'Indice':torch.from_numpy(np.array(indice, dtype='float32')),
                       #'himg': torch.from_numpy(np.array(rgb, dtype='float32')),
                       #'RE_RGB': torch.from_numpy(np.array(RE_rgb)),
                       'Label': torch.from_numpy(np.array(label, dtype='float32'))
                       }
        return dataset

if __name__ == '__main__':
    data = BasicDataset('data/rgb_read_noseg.csv')#实例化
    dataloader = DataLoader(data,batch_size=1)
    #for data in dataloader:
    generator = iter(dataloader)
    data = next(generator)
    spectral = data['Spectral']
    rgb = data['RGB']
    label = data['Chl']
    print(spectral.shape)
    print(rgb.shape)
    print(label.shape)
    print(label)
        #break
    data = next(generator)

