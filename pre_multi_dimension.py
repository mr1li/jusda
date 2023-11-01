import os
import warnings
import torch
import  matplotlib
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import pickle
from pytorch_forecasting import Baseline, NHiTS, TimeSeriesDataSet
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
def predict(data,filename2,filename3,filename4,filename5,pre_length):
    # data=pd.read_csv(filename1,index_col=0)
    # data['14_mean'] = data['14_mean'].astype(float)
    # data['28_mean'] = data['28_mean'].astype(float)
    # data['4_max'] = data['4_max'].astype(float)
    # data['4_min'] = data['4_min'].astype(float)
    # data['lag'] = data['lag'].astype(float)
    # data['2_mean'] = data['2_mean'].astype(float)
    # data['3_mean'] = data['3_mean'].astype(float)
    # data['4_mean'] = data['4_mean'].astype(float)
    # data['5_mean'] = data['5_mean'].astype(float)
    # data['2_min'] = data['2_min'].astype(float)
    # data['2_max'] = data['2_max'].astype(float)
    # data['7_min'] = data['7_min'].astype(float)
    # data['7_max'] = data['7_max'].astype(float)
    # data['quantity'] = data['quantity'].astype(float)
    # max_prediction_length = 7
    # max_encoder_length = 7
    # def filter_by_training_cutoff(group):
    #     training_cutoff = group["time_idx"].max() - max_prediction_length
    #     return group[group['time_idx'] <= training_cutoff]
    # # 对每个分组应用操作，并使用 concat 将结果拼接成一个新的 DataFrame
    # result_df = pd.concat([filter_by_training_cutoff(group) for name, group in data.groupby(['customer_name', 'customer_part_no'])], ignore_index=True)
    # # 重置索引
    # result_df.reset_index(drop=True, inplace=True)



    # nbeatspre#######################################
    # train_nbeats = TimeSeriesDataSet(
    #     result_df,
    #     time_idx="time_idx",
    #     target="quantity",
    #     group_ids=['customer_name', 'customer_part_no'],
    #     min_encoder_length=max_encoder_length,  # keep encoder length long (as it is in the validation set)
    #     max_encoder_length=max_encoder_length,
    #     min_prediction_length=max_prediction_length,
    #     max_prediction_length=max_prediction_length,
    #     time_varying_unknown_reals=["quantity"],
    # )
    # # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # # for each series
    # val_nbeats = TimeSeriesDataSet.from_dataset(train_nbeats, data, predict=True, stop_randomization=True)
    # # # create dataloaders for model
    # batch_size = 4 # set this between 32 to 128
    # train_dataloader_nbeats = train_nbeats.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    # val_dataloader_nbeats = val_nbeats.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    # with open("nbeats.pkl",'rb') as f:
    #     best_model=pickle.load(f)
    # prediction_nbeats = best_model.predict(val_dataloader_nbeats, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    # MAE()(prediction_nbeats.output, prediction_nbeats.y)
    # raw_predictions_nbeats = best_model.predict(val_dataloader_nbeats, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
    # for idx in range(18):  # plot 10 examples
    #     best_model.plot_prediction(raw_predictions_nbeats.x, raw_predictions_nbeats.output, idx=idx, add_loss_to_title=True)
    # print(prediction_nbeats)
    #
    #
    #
    #
    #
    #
    #
    #
    # # #nhitsarpre###################
    # training_nhits = TimeSeriesDataSet(
    #     result_df,
    #     time_idx="time_idx",
    #     target="quantity",
    #     group_ids=['customer_name', 'customer_part_no'],
    #     min_encoder_length=max_encoder_length,  # keep encoder length long (as it is in the validation set)
    #     max_encoder_length=max_encoder_length,
    #     min_prediction_length=max_prediction_length,
    #     max_prediction_length=max_prediction_length,
    #     time_varying_unknown_reals=[
    #         "7_mean",
    #         # "28_mean",
    #         # "14_mean",
    #         "4_max",
    #         "4_min",
    #         "lag",
    #         '2_mean',
    #         # '3_mean',
    #         # '4_mean',
    #         '5_mean',
    #         '2_max',
    #         '2_min',
    #         '7_max',
    #         '7_min',
    #         "quantity"
    #
    #     ],
    # )
    # with open("nhits.pkl",'rb') as f:
    #     nhits_model=pickle.load(f)
    # validation = TimeSeriesDataSet.from_dataset(training_nhits, data, predict=True, stop_randomization=True)
    # # # create dataloaders for model
    # batch_size = 4 # set this between 32 to 128
    # train_dataloader = training_nhits.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    # val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    # predictions_nhits = nhits_model.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    # print(MAE()(predictions_nhits.output, predictions_nhits.y))
    # raw_predictions_nhits = nhits_model.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
    # # for idx in range(18):  # plot 10 examples
    # #     nhits_model.plot_prediction(raw_predictions_nhits.x, raw_predictions_nhits.output, idx=idx, add_loss_to_title=True)
    # # print(predictions_nhits)
    # # #
    # # #
    # # #
    # #deeparpre###########
    # training_deepar = TimeSeriesDataSet(
    #     result_df,
    #     time_idx="time_idx",
    #     target="quantity",
    #     group_ids=['customer_name', 'customer_part_no'],
    #     max_encoder_length=max_encoder_length,
    #     max_prediction_length=max_prediction_length,
    #     static_categoricals=['customer_name', 'customer_part_no'],
    #     time_varying_unknown_reals=[
    #          "quantity",
    #     ],
    # )
    # #
    # with open("deepar.pkl",'rb') as f:
    #     deepar_model=pickle.load(f)
    # validation_deepar = TimeSeriesDataSet.from_dataset(training_deepar, data, predict=True, stop_randomization=True)
    # # # create dataloaders for model
    # batch_size = 4 # set this between 32 to 128
    # train_dataloader_deepar = training_deepar.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    # val_dataloader_deepar = validation_deepar.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    # predictions_deepar = deepar_model.predict(val_dataloader_deepar, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    # print(MAE()(predictions_deepar.output, predictions_deepar.y))
    # raw_predictions_deepar = deepar_model.predict(val_dataloader_deepar, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
    # # for idx in range(18):  # plot 10 examples
    # #     deepar_model.plot_prediction(raw_predictions_deepar.x, raw_predictions_deepar.output, idx=idx, add_loss_to_title=True)
    # # print(predictions_deepar)
    #
    #
    #
    #
    #
    #
    #
    # ####tftpre###############
    # training_tft = TimeSeriesDataSet(
    #     result_df,
    #     time_idx="time_idx",
    #     target="quantity",
    #     group_ids=['customer_name', 'customer_part_no'],
    #     min_encoder_length=max_encoder_length,  # keep encoder length long (as it is in the validation set)
    #     max_encoder_length=max_encoder_length,
    #     min_prediction_length=max_prediction_length,
    #     max_prediction_length=max_prediction_length,
    #     static_categoricals=['customer_name', 'customer_part_no'],
    #
    #     time_varying_unknown_reals=[
    #         "7_mean",
    #         "28_mean",
    #         "14_mean",
    #         "4_max",
    #         "4_min",
    #         "lag",
    #         '2_mean',
    #         '3_mean',
    #         '4_mean',
    #         '5_mean',
    #         '2_max',
    #         '2_min',
    #         '7_max',
    #         '7_min',
    #     ],
    #     target_normalizer=GroupNormalizer(
    #         groups=['customer_name', 'customer_part_no'], transformation="softplus"
    #     ),  # use softplus and normalize by group
    #     add_relative_time_idx=True,
    #     add_target_scales=True,
    #     add_encoder_length=True,
    # )
    # with open("tft.pkl",'rb') as f:
    #     tft_model=pickle.load(f)
    # validation_tft = TimeSeriesDataSet.from_dataset(training_tft, data, predict=True, stop_randomization=True)
    # # # create dataloaders for model
    # batch_size = 4 # set this between 32 to 128
    # train_dataloader_tft = training_tft.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    # val_dataloader_tft = validation_tft.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    # predictions_tft = tft_model.predict(val_dataloader_tft, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    # print(MAE()(predictions_tft.output, predictions_tft.y))
    # raw_predictions_tft = tft_model.predict(val_dataloader_tft, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
    # # for idx in range(18):  # plot 10 examples
    # #     tft_model.plot_prediction(raw_predictions_tft.x, raw_predictions_tft.output, idx=idx, add_loss_to_title=True)
    # import torch
    # real=predictions_tft.y[0]
    # # pre=(prediction_nbeats.output+predictions_nhits.output+predictions_deepar.output)/3
    # pre=(predictions_nhits.output+predictions_deepar.output+predictions_tft.output)/3
    # # q1=torch.sum(abs(prediction_nbeats.output-real)/real)/126
    # # q2=torch.sum(abs(predictions_nhits.output-real)/real)/126
    # # q3=torch.sum(abs(predictions_deepar.output-real)/real)/126
    # # q4=torch.sum(abs(predictions_tft.output-real)/real)/126
    # data['14_mean'] = data['14_mean'].astype(float)
    # data['28_mean'] = data['28_mean'].astype(float)
    # data['4_max'] = data['4_max'].astype(float)
    # data['4_min'] = data['4_min'].astype(float)
    # data['lag'] = data['lag'].astype(float)
    # data['2_mean'] = data['2_mean'].astype(float)
    # data['3_mean'] = data['3_mean'].astype(float)
    # data['4_mean'] = data['4_mean'].astype(float)
    # data['5_mean'] = data['5_mean'].astype(float)
    # data['2_min'] = data['2_min'].astype(float)
    # data['2_max'] = data['2_max'].astype(float)
    # data['7_min'] = data['7_min'].astype(float)
    # data['7_max'] = data['7_max'].astype(float)
    data['quantity'] = data['quantity'].astype(float)
    with open(filename2,'rb') as f:
        nbeats_model=pickle.load(f)
    predictions_nbeats = nbeats_model.predict(data, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    with open(filename3,'rb') as f:
        deepar_model=pickle.load(f)
    predictions_deepar = deepar_model.predict(data, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    with open(filename4,'rb') as f:
        tft_model=pickle.load(f)
    predictions_tft = tft_model.predict(data, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    with open(filename5,'rb') as f:
        nhits_model=pickle.load(f)
    predictions_nhits = nhits_model.predict(data, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    pre = torch.tensor((predictions_nbeats.output+predictions_nhits.output+predictions_tft.output+predictions_deepar.output ) / 4)
    pre = pre.cpu().numpy()
    # 确保 new_data 有 m 列
    print(pre)
    data['quantity'][-pre_length:]=pre.squeeze()
    selected_columns = [ 'eta','customer_name', 'customer_part_no', 'quantity']
    # 创建一个新的DataFrame，只包含选定的列
    new_data = data[selected_columns]
    print(new_data)
    return new_data
if __name__=='__main__':
    filename1=r'E:/model/nbeats_14_14.pkl'
    filename2=r'E:/model/deepar_14_14_multi_MultivariateNormal.pkl'
    filename3=r'E:/model/nhits_14_14_multi_MASE.pkl'
    filename4=r'E:/model/tft_14_14_MULTI.pkl'
    pre_length=14
    data=pd.read_csv(r'E:/pre.csv')
    predict(data, filename1 ,filename2,filename3,filename4,pre_length=pre_length)