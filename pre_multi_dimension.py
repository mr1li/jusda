import os
import warnings
import torch
from datetime import datetime, timedelta
import numpy as np
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from processing import make_predict_data,make_traindata,make_yanzheng
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
def predict(data,model,mode,con_len,pre_len,save_model_path,sku,yanzheng,end):
    if yanzheng==True:
        name = save_model_path + model + '_' + mode + '_' + str(con_len) + '_' + str(pre_len) + '.pkl'
        name2=save_model_path + model + '_' + mode + '_' + str(con_len) + '_' + str(pre_len) + '_smape'+'.csv'
        data['quantity'] = data['quantity'].astype(float)

        # def shai(group):
        #     group= group.iloc[-60:,:]
        #     # group['quantity'].iloc[0:10]=0
        #     return group
        # grouped=data.groupby(sku)
        # data=grouped.apply(shai).reset_index(drop=True)
        # print(data)
        def filter_by_training_cutoff(group):
            training_cutoff = group["time_idx"].max() - pre_len
            return group[group['time_idx'] <= training_cutoff]

        # 对每个分组应用操作，并使用 concat 将结果拼接成一个新的 DataFrame
        result_df = pd.concat(
            [filter_by_training_cutoff(group) for name, group in data.groupby(sku)],
            ignore_index=True)

        # 重置索引
        training = TimeSeriesDataSet(
            result_df,
            time_idx="time_idx",
            target="quantity",
            group_ids=sku,
            min_encoder_length=con_len,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=con_len,
            min_prediction_length=pre_len,
            max_prediction_length=pre_len,
            static_categoricals=sku,
            target_normalizer=GroupNormalizer(
                groups=sku, transformation="softplus"
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True

        )
        validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
        print(validation)
        batch_size = 4  # set this between 32 to 128
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        grouped = data.groupby(sku)
        val_batch_size = len(grouped)
        print(val_batch_size)
        val_dataloader = validation.to_dataloader(train=False, batch_size=val_batch_size, num_workers=0)

        with open(name, 'rb') as f:
            pre_model = pickle.load(f)




        core_df=val_dataloader.dataset.index
        print(len(core_df))
        con_matrix=np.zeros((len(core_df),int(core_df['sequence_length'].iloc[0]/2)))

        for n in range(len(core_df)):
            con_matrix[n,:]=data['quantity'].iloc[core_df['index_start'].iloc[n]:core_df['index_end'].iloc[n]-13]
        con_matrix = con_matrix.reshape(len(con_matrix), int(len(con_matrix[0]) / 7), 7).sum(axis=2)
        print(con_matrix)

        predictions = pre_model.predict(val_dataloader,  return_y=True,return_index = True)
        pre = np.array(predictions.output)

        real = np.array(predictions.y[0])

        real = real.reshape(len(pre), int(len(pre[0])/7), 7).sum(axis=2)
        pre = pre.reshape(len(real),int(len(pre[0])/7), 7).sum(axis=2)
        for n in range(len(pre)):
            for m in range(len(pre[0])):
                # pre[n, m]=int(pre[n,m])
                ave=(con_matrix[n, m] + con_matrix[n, 1 - m])/2
                pre[n, m] = (pre[n, m]+ con_matrix[n, m] + con_matrix[n, 1 - m]) / 3
                if pre[n,m]>1.1*ave and con_matrix[n,m]!=0 :
                    pre[n,m]=ave*(1+(pre[n,m]-ave)/pre[n,m])
                if pre[n,m]<0.9*ave :
                    pre[n,m]=ave*(1-(-pre[n,m]+ave)/pre[n,m])

        print(len(pre))
        smape= []
        for k in range(len(pre)):
            a = np.mean(2 * abs(pre[k] - real[k]) / (pre[k] + real[k]))
            smape.append(a)

        df=predictions.index
        grouped=df.groupby(sku)
        global t
        t=0
        def pinjie(group,pre,end,smape):
            global t

            group = group.loc[group.index.repeat(len(pre[0]))].reset_index(drop=True)
            group['pre']=pre[t]
            group['real'] = real[t]
            group['smape']=smape[t]
            end=pd.to_datetime(end)
            group['eta']=[end - timedelta(days=7*(len(pre[0])-i)) for i in range(len(pre[0]))]

            t=t+1
            return group
        df=grouped.apply(lambda x: pinjie(x, pre,end,smape))
        # 存储为 CSV 文件
        df.to_csv(name2, index=False)
        return core_df

    else:
        name = save_model_path + model + '_' + mode + '_' + str(con_len) + '_' + str(pre_len) + '.pkl'
        data['quantity'] = data['quantity'].astype(float)
        print(data)

        def filter_by_training_cutoff(group):
            training_cutoff = group["time_idx"].max() - pre_len
            return group[group['time_idx'] <= training_cutoff]

            # 对每个分组应用操作，并使用 concat 将结果拼接成一个新的 DataFrame

        result_df = pd.concat(
            [filter_by_training_cutoff(group) for name, group in data.groupby(sku)],
            ignore_index=True)

        # 重置索引
        training = TimeSeriesDataSet(
            result_df,
            time_idx="time_idx",
            target="quantity",
            group_ids=sku,
            min_encoder_length=con_len,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=con_len,
            min_prediction_length=pre_len,
            max_prediction_length=pre_len,
            static_categoricals=sku,
            target_normalizer=GroupNormalizer(
                groups=sku, transformation="softplus"
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True

        )
        validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
        batch_size = 4  # set this between 32 to 128
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        grouped = data.groupby(sku)
        val_batch_size = len(grouped)
        print(val_batch_size)
        val_dataloader = validation.to_dataloader(train=False, batch_size=val_batch_size, num_workers=0)
        with open(name, 'rb') as f:
            pre_model = pickle.load(f)
        predictions = pre_model.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)






        pre = torch.tensor(predictions.output)
        pre = pre.cpu().numpy()
        pre=np.transpose(pre)
        # 确保 new_data 有 m 列
        grouped = data.groupby(sku)
        global i
        i=0
        def sum_by_week(group):
            global  i
            group.iloc[-pre_len:,group.columns.get_loc('quantity')]=pre[:,i]
            i=i+1
            return group

        data = grouped.apply(sum_by_week).reset_index(drop=True)
        selected_columns = [ 'eta']+sku+[ 'quantity','time_idx']
        # # 创建一个新的DataFrame，只包含选定的列
        new_data = data[selected_columns]

if __name__=='__main__':
    mode='Day'
    con_len=14
    pre_len=14
    model='tft'
    yanzheng=True
    save_model_path='E:\\model\\'
    sku=['customer_name', 'customer_part_no']
    data=pd.read_csv('f:\\day_shaixuan_jinyibu_大于20_0.7_0.7.csv')
    start = '2022-8-01'
    end = '2022-11-07'
    # data = make_yanzheng(data, mode, sku, 14, 14, start, end)
    global t
    t=0
    f=pre_data=predict(data,model,mode,con_len,pre_len,save_model_path,sku,yanzheng,end)