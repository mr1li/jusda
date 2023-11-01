import os
import warnings
warnings.filterwarnings("ignore")
import pickle
os.chdir("../../..")
from pytorch_forecasting import Baseline, NHiTS, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MQF2DistributionLoss, QuantileLoss,MAE, SMAPE, PoissonLoss, QuantileLoss,MAPE,MASE
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd

def NHITS(data,con_length,pre_length,filename2):
    # 构造数据集：
    # df['eta'] = pd.to_datetime(df['eta'])
    # df['eta'] = df['eta'].dt.floor('D')
    # df = df.groupby(['customer_name', 'customer_part_no','eta'])['quantity'].sum().reset_index()
    # # df=pd.read_csv(r'E:/qqq.csv')
    # grouped=df.groupby(['customer_name', 'customer_part_no'])
    # def operation(group):
    #     date_range = pd.date_range(group['eta'].min(), group['eta'].max(), freq='D')
    #     # # # # 使用 merge 将原始数据与完整的日期范围合并，并使用 fillna 填充缺失的值
    #     fill_values = {'quantity': 0,'customer_name':group['customer_name'].iloc[0], 'customer_part_no':group['customer_part_no'].iloc[0]}
    #     group = date_range.to_frame(index=False, name='eta').merge(group, on='eta', how='left').fillna(fill_values)
    #     group['time_idx'] = group['eta'].apply(lambda x: int(x.toordinal()))
    #     group["time_idx"] -= group["time_idx"].min()
    #     group = group.sort_values(by="time_idx")
    #     group['2_mean'] = group['quantity'].rolling(window=2, min_periods=1).mean()
    #     group['3_mean'] = group['quantity'].rolling(window=3, min_periods=1).mean()
    #     group['4_mean'] = group['quantity'].rolling(window=4, min_periods=1).mean()
    #     group['5_mean'] = group['quantity'].rolling(window=5, min_periods=1).mean()
    #     group['7_mean'] = group['quantity'].rolling(window=7, min_periods=1).mean()
    #     group['14_mean'] = group['quantity'].rolling(window=14, min_periods=1).mean()
    #     group['28_mean'] = group['quantity'].rolling(window=28, min_periods=1).mean()
    #     group['2_max'] = group['quantity'].rolling(window=2, min_periods=1).max()
    #     group['2_min'] = group['quantity'].rolling(window=2, min_periods=1).min()
    #     group['4_max'] = group['quantity'].rolling(window=4, min_periods=1).max()
    #     group['4_min'] = group['quantity'].rolling(window=4, min_periods=1).min()
    #     group['7_max'] = group['quantity'].rolling(window=7, min_periods=1).max()
    #     group['7_min'] = group['quantity'].rolling(window=7, min_periods=1).min()
    #     group['lag'] = group['quantity'].shift().fillna(0)
    #     return group
    # # 对每个分组应用操作，并使用 concat 将结果拼接成一个新的DataFrame
    # new_df = pd.concat([operation(group) for name, group in grouped  if len(operation(group)) > 30],ignore_index=True)
    # new_df.to_csv('qqq.csv')
    ########################
    data['quantity'] = data['quantity'].astype(float)
    max_prediction_length = pre_length
    max_encoder_length =con_length
    def filter_by_training_cutoff(group):
        training_cutoff = group["time_idx"].max() - max_prediction_length
        return group[group['time_idx'] <= training_cutoff]

    # 对每个分组应用操作，并使用 concat 将结果拼接成一个新的 DataFrame
    result_df = pd.concat(
        [filter_by_training_cutoff(group) for name, group in data.groupby(['customer_name', 'customer_part_no'])],
        ignore_index=True)
    result_df.reset_index(drop=True, inplace=True)
    training = TimeSeriesDataSet(
        result_df,
        time_idx="time_idx",
        target="quantity",
        group_ids=['customer_name', 'customer_part_no'],
        min_encoder_length=max_encoder_length,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[
            "quantity"
        ],
    )
    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    batch_size = 4  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    import tensorboard as tb
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=200, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback],
        limit_train_batches=30,
        enable_checkpointing=True,
    )

    net = NHiTS.from_dataset(
        training,
        learning_rate=0.007,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        backcast_loss_ratio=0.0,
        hidden_size=64,
        optimizer="AdamW",
        loss=MASE(),
    )
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NHiTS.load_from_checkpoint(best_model_path)
    with open(filename2, 'wb') as f:
        pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(r"E:/model/tft_14_14.pkl", 'rb') as f:
    # predictions = best_tft.predict(pre, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    # print(predictions.output[0])
    # pre['quantity'].iloc[-max_prediction_length:]=np.array(predictions.output[0])

    predictions = best_model.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    # MAE()(predictions.output, predictions.y)
    raw_predictions = best_model.predict(val_dataloader, mode="raw", return_x=True,
                                         trainer_kwargs=dict(accelerator="cpu"))
    for idx in range(15):  # plot 10 examples
        best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
if __name__=='__main__':
    filename1=r'E:/qqq.csv'
    con_length=14
    pre_length=14
    filename2=r'E:/model/nhits_14_14_multi_MASE.pkl'
    data = pd.read_csv(filename1, index_col=0)
    NHITS(data,con_length,pre_length,filename2)
