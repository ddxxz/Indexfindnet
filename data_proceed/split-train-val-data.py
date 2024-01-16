
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
clean_data_flag = int(sys.argv[1])
# random_state = sys.argv[2]
# random_state = int(random_state)
path = r'data/dataset_rgb_train_val.csv'
# with open(path,'r') as f:
#      lines = f.readlines()
#      line1 = lines[0]
#      temp = lines[1:]
temp = pd.read_csv(path)

print(len(temp))
if clean_data_flag:
    #train, test = train_test_split(temp, test_size=0.001, random_state=42, shuffle=True)
    val=temp[0:2]
    train=temp[2:]
else:
    # index_val = [i * 9 for i in range(len(temp) // 9)]
    # index_all = [i for i in range(len(temp))]
    # index_train = [i for i in range(len(temp)) if i not in index_val]
    # temp = pd.DataFrame(temp)
    # train = temp.values[index_train]
    # val = temp.values[index_val]

    train, val = train_test_split(temp, test_size=0.1, random_state=42, shuffle=True)
    # before_0429 = temp[:595]
    # other = temp[595:]
    # #print("tmp",len(temp))
    # train, val = train_test_split(before_0429, test_size=0.1, random_state=42, shuffle=True)
    # print(train.shape,other.shape)
    # train = np.concatenate([train, other])
    # print(train.shape)


train = pd.DataFrame(train,columns=temp.columns)
val = pd.DataFrame(val,columns=temp.columns)
train.to_csv('data/train.csv',index=False)
val.to_csv('data/val.csv',index=False)
# with open('data/wheat_rice_train_rgb.csv', 'w+') as f:
#     f.writelines(line1)
#     f.writelines(train)
#
# with open('data/wheat_rice_val_rgb.csv', 'w+') as f:
#     f.writelines(line1)
#     f.writelines(val)