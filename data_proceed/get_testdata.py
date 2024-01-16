import pandas as pd
import sys
from sklearn.model_selection import train_test_split
clean_data_flag = int(sys.argv[1])
input_csv = sys.argv[2]
# random_state = sys.argv[3]
# random_state = int(random_state)
if str(input_csv).endswith('csv'):
    data = pd.read_csv(input_csv)
    # if crop == 'wheat' or crop == 'wheat_rice':
    #     data = data[52:]
else:
    data= pd.read_excel(input_csv)
#test = pd.read_csv('data/test.csv')
if clean_data_flag:
    index_test = [0]
else:
    # index_id = test['id']
    # data_id = list(data['id'])
    # #print(data_id.index(2465))
    # index_test = [data_id.index(x) for x in index_id]
    index_test = [i * 10 for i in range(len(data) // 10)]
    #index_test = [i*5 for i in range(len(data)//5)]
print(len(index_test))
print(len(data))

index_all = [i for i in range(len(data))]
index_train_val = [i for i in range(len(data)) if i not in index_test]
print(len(index_train_val))
data_test = data.values[index_test]
data_train_val = data.values[index_train_val]

# data_train_val, data_test = train_test_split(data, test_size=0.1, random_state=random_state, shuffle=True)


data_col=data.columns
test = pd.DataFrame(data=data_test,columns=data_col)
train_val = pd.DataFrame(data=data_train_val,columns=data_col)
test.to_csv('data/test.csv',index=False)
train_val.to_csv('data/dataset_rgb_train_val.csv',index=False)