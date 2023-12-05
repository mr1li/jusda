import os
import warnings
import torch
import numpy as np
from processing import make_predict_data
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
def predict(data,model,mode,con_length,pre_length,save_model_path,sku):
    name = save_model_path + model + '_' + mode + '_' + str(con_length) + '_' + str(pre_length) + '.pkl'
    data['quantity'] = data['quantity'].astype(float)
    print(data)
    with open(name,'rb') as f:
        pre_model=pickle.load(f)
    predictions = pre_model.predict(data, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    print(predictions)
    pre = torch.tensor(predictions.output)
    pre = pre.cpu().numpy()
    pre=np.transpose(pre)
    # 确保 new_data 有 m 列




    grouped = data.groupby(sku)
    global i
    i=0
    def sum_by_week(group):
        global  i
        print(group.iloc[-pre_len:,group.columns.get_loc('quantity')])
        print(pre[:,i])
        group.iloc[-pre_len:,group.columns.get_loc('quantity')]=pre[:,i]
        i=i+1
        return group

    data = grouped.apply(sum_by_week).reset_index(drop=True)
    selected_columns = [ 'eta']+sku+[ 'quantity','time_idx']
    # # 创建一个新的DataFrame，只包含选定的列
    new_data = data[selected_columns]
    return new_data
if __name__=='__main__':
    mode='Day'
    con_len=8
    pre_len=4
    model='tft'
    save_model_path='E:\\model\\'
    sku=['customer_name', 'customer_part_no']
    data=pd.read_csv(r'F:/集中数据1.csv')
    data=make_predict_data(data,mode,sku,pre_len,con_len)
    pre_data=predict(data,model,mode,con_len,pre_len,save_model_path,sku)