import os
import warnings
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss, MQF2DistributionLoss, QuantileLoss,NormalDistributionLoss
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import pickle
os.chdir("../../..")
import pandas as pd
from pytorch_forecasting.data import NaNLabelEncoder
def deepar(data, mode,con_length, pre_length, filename2):

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

    #
    data['quantity'] = data['quantity'].astype(float)
    max_prediction_length = pre_length
    max_encoder_length = con_length
    def filter_by_training_cutoff(group):
        training_cutoff = group["time_idx"].max() - max_prediction_length
        return group[group['time_idx'] <= training_cutoff]
    # 对每个分组应用操作，并使用 concat 将结果拼接成一个新的 DataFrame
    result_df = pd.concat(
        [filter_by_training_cutoff(group) for name, group in data.groupby(['customer_name', 'customer_part_no'])],
        ignore_index=True)
    # 重置索引
    result_df.reset_index(drop=True, inplace=True)
    training = TimeSeriesDataSet(
        result_df,
        time_idx="time_idx",
        target="quantity",
        group_ids=['customer_name', 'customer_part_no'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=['customer_name', 'customer_part_no'],
        time_varying_unknown_reals=[
            "quantity",
        ],
    )
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    batch_size = 4  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    grouped = data.groupby(['customer_name', 'customer_part_no'])
    val_batch_size = len(grouped)
    print(val_batch_size)
    val_dataloader = validation.to_dataloader(train=False, batch_size=val_batch_size, num_workers=0)

    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=500, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        limit_train_batches=50,
        enable_checkpointing=True,
    )
    net = DeepAR.from_dataset(
        training,
        learning_rate=0.001,
        log_interval=10,
        log_val_interval=1,
        hidden_size=30,
        rnn_layers=2,
        optimizer="Adam",
        loss=MultivariateNormalDistributionLoss(rank=pre_length),
    )
    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = DeepAR.load_from_checkpoint(best_model_path)
    name = filename2 + 'deepar' + '_' + mode + '_' + str(con_length) + '_' + str(pre_length) + '.pkl'
    with open(name, 'wb') as f:
        pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    prediction_deepar = best_model.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
    MAE()(prediction_deepar.output, prediction_deepar.y)
    raw_predictions_deepar = best_model.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
    for idx in range(val_batch_size):  # plot 10 examples
        best_model.plot_prediction(raw_predictions_deepar.x, raw_predictions_deepar.output, idx=idx, add_loss_to_title=True)
    print(prediction_deepar)

if __name__ == '__main__':
    filename1 = r'F:/集中数据1.csv'
    con_length = 8
    pre_length = 4
    mode = 'Day'
    filename2 = 'E:\\model\\'
    data = pd.read_csv(filename1, index_col=0)
    deepar(data,mode,con_length,pre_length,filename2)

