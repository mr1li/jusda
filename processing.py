import  pandas as pd
import numpy as np
def make_traindata(df,mode,sku,pre_len,con_len):
    length=pre_len
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
    print(df)
    df = df.groupby(sku).filter(lambda group: len(group) >= length)
    print(df)
    return  df
def make_yanzheng(df,mode,sku,pre_len,con_len,start,end):
    length=pre_len
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
    def full(group):
        date_range = pd.date_range(start=start, end=end, freq='D')
        new_df = pd.DataFrame({'eta': date_range})
        # 将原始DataFrame与新DataFrame按照时间列连接起来
        merged_df = pd.merge(new_df, group, left_on='eta', right_on='eta', how='left')
        # 将缺失值填充为0
        merged_df['quantity'] = merged_df['quantity'].fillna(0)
        merged_df[sku] = group[sku].iloc[0]
        result_df = merged_df[sku + ['eta', 'quantity']]
        return result_df
        # 删除多余的列
    # print(df)
    # date_range = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
    # new_df = pd.DataFrame({'时间': date_range})
    # print(new_df)
    # # 将原始DataFrame与新DataFrame按照时间列连接起来
    # merged_df = pd.merge(new_df, df, left_on='时间', right_on='eta', how='left')
    #
    # # 将缺失值填充为0
    # merged_df['quantity'] = merged_df['quantity'].fillna(0)
    # merged_df[sku]=df[sku]
    # # 删除多余的列
    # result_df = merged_df[sku+['时间', 'quantity']]
    # result_df.to_csv('f:\o0231223.csv')
    # print(result_df)


    print(df)
    grouped=df.groupby(sku)
    df = grouped.apply(full).reset_index(drop=True)



    df['time_idx']=df.groupby(sku).cumcount()
    print(df)
    df = df.groupby(sku).filter(lambda group: len(group) >= length)
    print(df)
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
    filename1 = r'E:\pre_data_20231223.csv'  # outbound文件
    df = pd.read_csv(filename1)
    mode='Day'#'Day' 'Week' 'Month'
    # sku=['customer_name', 'customer_part_no','supplier_name',	'supplier_part_no','manufacture_name',	'site_db']
    sku = ['customer_name', 'customer_part_no']
    start='2022-10-01'
    end='2022-11-07'
    df=make_yanzheng(df,mode,sku,14,14,start,end)
    df.to_csv('f:\\yuceshiyan.csv')