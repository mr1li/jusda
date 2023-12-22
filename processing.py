import  pandas as pd
import numpy as np
def make_traindata(df,mode,sku,pre_len,con_len):
    length=con_len+pre_len
    "length代表至少需要多少数据"
    df['eta']=pd.to_datetime(df['eta'])
    grouped = df.groupby(sku)
    def sum_by_week(group):
        group=group.resample('W-Mon', on='eta')['quantity'].sum()
        return group
    def sum_by_month(group):
        group=group.resample('M', on='eta')['quantity'].sum()
        return group
    def sum_by_day(group):
        group=group.resample('D', on='eta')['quantity'].sum()
        return group
    if mode=='Day':
        df = grouped.apply(sum_by_day).reset_index()
    if mode=='Month':
        df = grouped.apply(sum_by_month).reset_index()
    if mode=='Week':
        df = grouped.apply(sum_by_week).reset_index()
    df['time_idx']=df.groupby(sku).cumcount()
    df = df.groupby(sku).filter(lambda group: len(group) >= length)
    return  df

def make_predict_data(df,mode,sku,pre_len,con_len):
    df['eta']=pd.to_datetime(df['eta'])

    grouped = df.groupby(sku)
    if len(grouped)>1:
        def sum_by_week(group):
            group=group.resample('W-Mon', on='eta')['quantity'].sum()
            return group
        def sum_by_month(group):
            group=group.resample('M', on='eta')['quantity'].sum()
            # group['time_idx']=range(0,len(group))
            return group
        def sum_by_day(group):
            print(group)
            group=group.resample('D', on='eta')['quantity'].sum()

            print(group)
            return group
        if mode=='Day':
            df = grouped.apply(sum_by_day).reset_index()
        if mode=='Month':
            df = grouped.apply(sum_by_month).reset_index()
        if mode=='Week':
            df = grouped.apply(sum_by_week).reset_index()
        print(df.columns)
        def sum_by_month2(group):
            last=group.tail(1)
            date=pd.to_datetime(last['eta'],errors='coerce')
            last = pd.concat([last] * pre_len, ignore_index=True)
            last['quantity']=0
            for i in range(len(last)):
                t=date + pd.DateOffset(months=1)
                last.iloc[i, last.columns.get_loc('eta')] = t
                date=last['eta'].loc[i]
            group=pd.concat([group,last],ignore_index=True)
            return group
        def sum_by_day2(group):
            last=group.tail(1)
            # print(last)
            date=pd.to_datetime(last['eta'],errors='coerce')
            last = pd.concat([last] * pre_len, ignore_index=True)
            last['quantity']=0
            for i in range(len(last)):
                t=date + pd.DateOffset(days=1)
                last.iloc[i, last.columns.get_loc('eta')] = t
                date=last['eta'].loc[i]
            group=pd.concat([group,last],ignore_index=True)
            return group
        def sum_by_week2(group):

            last=group.tail(1)
            date=pd.to_datetime(last['eta'],errors='coerce')
            last = pd.concat([last] * pre_len, ignore_index=True)
            last['quantity']=0
            for i in range(len(last)):
                t=date + pd.DateOffset(weeks=1)
                last.iloc[i, last.columns.get_loc('eta')] = t
                date=last['eta'].loc[i]
            group=pd.concat([group,last],ignore_index=True)

            return group

        grouped=df.groupby(sku)
        if mode == 'Day':
            df = grouped.apply(sum_by_day2).reset_index(drop=True)
        if mode == 'Month':
            df = grouped.apply(sum_by_month2).reset_index(drop=True)
        if mode == 'Week':
            df = grouped.apply(sum_by_week2).reset_index(drop=True)
        df['time_idx'] = df.groupby(sku).cumcount()
        df = df.groupby(sku).filter(lambda group: len(group) >= con_len)
        return df
    else:
        if mode=='Week':
            result_df = df.resample('W-Mon', on='eta')['quantity'].sum().reset_index()
            last = result_df.tail(1)
            date = pd.to_datetime(last['eta'], errors='coerce')
            last = pd.concat([last] * pre_len, ignore_index=True)
            last['quantity'] = 0
            for i in range(len(last)):
                t = date + pd.DateOffset(weeks=1)
                last.iloc[i, last.columns.get_loc('eta')] = t
                date = last['eta'].loc[i]
            result_df = pd.concat([result_df, last], ignore_index=True)
            for i in sku:
                result_df[i]=df[i]
            result_df = result_df.ffill()
        if mode=='Day':
            result_df = df.resample('D', on='eta')['quantity'].sum().reset_index()
            last = result_df.tail(1)
            date = pd.to_datetime(last['eta'], errors='coerce')
            last = pd.concat([last] * pre_len, ignore_index=True)
            last['quantity'] = 0
            for i in range(len(last)):
                t = date + pd.DateOffset(days=1)
                last.iloc[i, last.columns.get_loc('eta')] = t
                date = last['eta'].loc[i]
            result_df = pd.concat([result_df, last], ignore_index=True)
            for i in sku:
                result_df[i]=df[i]
            result_df = result_df.ffill()
        if mode=='Month':
            result_df = df.resample('M', on='eta')['quantity'].sum().reset_index()
            last = result_df.tail(1)
            date = pd.to_datetime(last['eta'], errors='coerce')
            last = pd.concat([last] * pre_len, ignore_index=True)
            last['quantity'] = 0
            for i in range(len(last)):
                t = date + pd.DateOffset(months=1)
                last.iloc[i, last.columns.get_loc('eta')] = t
                date = last['eta'].loc[i]
            result_df = pd.concat([result_df, last], ignore_index=True)
            for i in sku:
                result_df[i]=df[i]
            result_df = result_df.ffill()
        result_df['time_idx'] = result_df.groupby(sku).cumcount()
        result_df = result_df.groupby(sku).filter(lambda group: len(group) >= con_len)
        return result_df
if __name__=='__main__':
    filename1 = r'F:\test_pre_data.csv'  # outbound文件
    df = pd.read_csv(filename1)
    mode='Month'#'Day' 'Week' 'Month'
    sku=['customer_name', 'customer_part_no']
    df=make_predict_data(df,mode,sku,4,2)
    df.to_csv('f:\\after.csv')