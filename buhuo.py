import pandas as pd
import numpy as np
import numpy as np
from datetime import datetime,timedelta
from pre_multi_dimension import predict
def buhuo(df_asn,df_inbound,df_demand,period,pre_length,inventory,fahuo,baozhuang,filename1 ,filename2,filename3,filename4):
    df_demand=predict(df_demand, filename1 ,filename2,filename3,filename4,pre_length=pre_length)
    first_date = df_demand['eta'].iloc[0]
    first_date = pd.to_datetime(first_date)
    df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime']).dt.date
    df_inbound['eta'] = pd.to_datetime(df_inbound['eta']).dt.date
    df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])
    df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime'])
    # 检查 'asn_create_datetime' 列的数据类型
    merged_df = pd.merge(df_asn, df_inbound, on='asn_no')
    merged_df['datetime_diff'] = (merged_df['eta'] - merged_df['asn_create_datetime']).dt.days
    lead_time = round(merged_df['datetime_diff'].mean())
    three_months_ago = first_date - pd.DateOffset(months=3)
    # 仅保留在最近三个月内的数据
    three_months_ago = pd.to_datetime(three_months_ago)
    df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime'])
    selected_rows = df_asn[df_asn['asn_create_datetime'].between(three_months_ago, first_date)]
    grouped = selected_rows.groupby('asn_create_datetime')
    group_count = len(grouped)
    freq = round((pd.to_datetime(first_date) - pd.to_datetime(three_months_ago)).days / (group_count + 1))



    if period=='Day':
        inbound = df_inbound
        inbound_grouped = inbound.groupby('eta')['quantity'].sum().reset_index()
        min_date = inbound_grouped['eta'].min()
        df_demand['eta'] = pd.to_datetime(df_demand['eta'])
        date2 = df_demand['eta'].max()
        date_range = pd.date_range(min_date, date2, freq='D')
        new_df = pd.DataFrame({'eta': date_range})
        new_df['eta'] = pd.to_datetime(new_df['eta'])
        result_df = pd.concat([new_df.set_index('eta'), inbound_grouped.set_index('eta')], axis=1,
                                          sort=True).fillna(0)
        inbound = result_df.reset_index()
        inbound = inbound[(inbound['eta'] >= first_date) & (inbound['eta'] <= date2)]
        inbound = np.array(inbound['quantity'])
        # 补货验证：
        # 1：首先创建好数据表格：
        demand_yuanshi = np.array((df_demand['quantity']))
        xiancun_data = np.zeros((pre_length + lead_time + freq + 1))
        buchong_data = np.zeros((pre_length + lead_time + freq))
        daohuo_data = np.zeros((pre_length + lead_time + freq))
        demand_data1 = np.zeros((pre_length))
        demand_data2 = np.zeros((pre_length + lead_time + freq))
        inbound_data = np.zeros((pre_length))


        tjyz = 1.5

        #安全水平计算
        initial = inventory
        df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])
        # 按日期排序
        df_inbound = df_inbound.sort_values(by='eta')
        # 计算最近一年的开始日期
        one_year_ago = df_inbound['eta'].max() - pd.DateOffset(years=1)
        # 选择最近一年的数据
        recent_data = df_inbound[df_inbound['eta'] >= one_year_ago]
        # 计算 'quantity' 列的 0.95 分位点
        quantile_95 = recent_data['quantity'].quantile(1)
        min_secure = quantile_95
        mean=recent_data['quantity'].mean()
        max_secure=(( lead_time+freq)*mean+min_secure)*2

        demand_data2[:len(demand_yuanshi)] = demand_yuanshi * tjyz
        avg_value = np.mean(demand_yuanshi) * tjyz
        for k in range(demand_data1.shape[0], demand_data2.shape[0]):
            demand_data2[k] = avg_value
        demand_data2 = np.sort(demand_data2)[::-1]


        df_asn = df_asn.groupby('asn_create_datetime')['quantity'].sum().reset_index()
        for index, row in df_asn.sort_values(by='asn_create_datetime', ascending=False).iterrows():
            days_diff = (first_date - row['asn_create_datetime']).days
            if days_diff <= lead_time and days_diff > 0:
                print(days_diff)
                daohuo_data[(lead_time - days_diff) ] = row['quantity']

        xiancun_data[0] = inventory
        for h in range(lead_time):
            xiancun_data[h + 1] = xiancun_data[h] + daohuo_data[h] - demand_data2[h]
        for m in range(lead_time, len(demand_data1) + lead_time):
            xiancun_data[m + 1] = xiancun_data[m] - demand_data2[m]
            if xiancun_data[m + 1] <= min_secure:
                q = 0
                chu = xiancun_data[m + 1]
                print(chu)
                for n in range(m, m + freq):
                    q = q + demand_data2[n]
                t = q - chu + min_secure

                t = max(t, fahuo) + (baozhuang - (max(t, fahuo) % baozhuang))

                buchong_data[m - lead_time] = t
                daohuo_data[m] = t
                xiancun_data[m + 1] = t + chu
        date_list = [first_date + timedelta(days=i) for i in range(len(demand_yuanshi))]
        date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
        buchong_data = buchong_data[:pre_length]
        buchong = buchong_data.reshape(demand_yuanshi.shape, order='F')
        out = {'date': date_strings, 'buhuo': buchong}
        out = pd.DataFrame(out)
        # out_name = 'buhuo_' + '.csv'
        # out.to_csv(out_name, index=False)
        return out,min_secure,max_secure

            # 动态更新：
        #     inventory = inventory + sum(daohuo_data[:pre_length]) - sum(outbound_data[:])
        #     print(inventory)
        #     true = sum(outbound_data[:])
        #     pre = sum(demand_data2[:pre_length])
        #     if true < pre:
        #         tjyz = max(1.3, tjyz - 0.1)
        #     else:
        #         tjyz = min(2, tjyz + 0.1)
        #

        # 评价体系：计算从first_date到结束的真实与算法库存量，先比需求满足，在比库存下降。
        # 首先计算两个库存量：
        # 第一个真实的：


    if period=='Week':
        inbound = df_inbound
        inbound_grouped = inbound.groupby('eta')['quantity'].sum().reset_index()
        min_date = inbound_grouped['eta'].min()
        df_demand['eta'] = pd.to_datetime(df_demand['eta'])
        date2 = df_demand['eta'].max()+ pd.DateOffset(days=7)
        date_range = pd.date_range(min_date, date2, freq='D')
        new_df = pd.DataFrame({'eta': date_range})
        new_df['eta'] = pd.to_datetime(new_df['eta'])

        result_df = pd.concat([new_df.set_index('eta'), inbound_grouped.set_index('eta')], axis=1,sort=True).fillna(0)
        inbound = result_df.reset_index()
        inbound = inbound[(inbound['eta'] >= first_date) & (inbound['eta'] <= date2)]
        inbound = np.array(inbound['quantity'])

        # 补货验证：
        # 1：首先创建好数据表格：
        pre_length=pre_length*7
        demand_yuanshi = np.array(df_demand['quantity'])
        demand_yuanshi = np.repeat(demand_yuanshi, 7)
        # 如果你需要将结果转换回 DataFrame，可以使用以下代码
        demand_yuanshi=demand_yuanshi/7
        print(demand_yuanshi)

        xiancun_data = np.zeros((pre_length + lead_time + freq + 1))
        buchong_data = np.zeros((pre_length + lead_time + freq))
        daohuo_data = np.zeros((pre_length + lead_time + freq))
        demand_data1 = np.zeros((pre_length))
        demand_data2 = np.zeros((pre_length + lead_time + freq))

        inbound_data = np.zeros((pre_length))

        tjyz = 1.5

        # 安全水平计算
        initial = inventory
        df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])
        # 按日期排序
        df_inbound = df_inbound.sort_values(by='eta')
        # 计算最近一年的开始日期
        one_year_ago = df_inbound['eta'].max() - pd.DateOffset(years=1)
        # 选择最近一年的数据
        recent_data = df_inbound[df_inbound['eta'] >= one_year_ago]
        # 计算 'quantity' 列的 0.95 分位点
        quantile_95 = recent_data['quantity'].quantile(1)
        min_secure = quantile_95
        mean = recent_data['quantity'].mean()
        max_secure = ((lead_time + freq) * mean + min_secure) * 2

        demand_data2[:len(demand_yuanshi)] = demand_yuanshi * tjyz
        avg_value = np.mean(demand_yuanshi) * tjyz
        for k in range(demand_data1.shape[0], demand_data2.shape[0]):
            demand_data2[k] = avg_value
        demand_data2 = np.sort(demand_data2)[::-1]

        df_asn = df_asn.groupby('asn_create_datetime')['quantity'].sum().reset_index()
        for index, row in df_asn.sort_values(by='asn_create_datetime', ascending=False).iterrows():
            days_diff = (first_date - row['asn_create_datetime']).days
            if days_diff <= lead_time and days_diff > 0:
                print(days_diff)
                daohuo_data[(lead_time - days_diff)] = row[
                    'quantity']

        xiancun_data[0] = inventory
        for h in range(lead_time):
            xiancun_data[h + 1] = xiancun_data[h] + daohuo_data[h] - demand_data2[h]
        for m in range(lead_time, len(demand_data1) + lead_time):
            xiancun_data[m + 1] = xiancun_data[m] - demand_data2[m]
            if xiancun_data[m + 1] <= min_secure:
                q = 0
                chu = xiancun_data[m + 1]
                print(chu)
                for n in range(m, m + freq):
                    q = q + demand_data2[n]
                t = q - chu + min_secure

                t = max(t, fahuo) + (baozhuang - (max(t, fahuo) % baozhuang))
                buchong_data[m - lead_time] = t
                daohuo_data[m] = t
                xiancun_data[m + 1] = t + chu

        date_list = [first_date + timedelta(days=i) for i in range(len(demand_yuanshi))]
        date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
        buchong_data = buchong_data[:pre_length]
        buchong = buchong_data.reshape(demand_data1.shape, order='F')
        out = {'date': date_strings, 'buhuo': buchong}
        out = pd.DataFrame(out)
        out_name = 'buhuo_' + '.csv'
        out.to_csv(out_name, index=False)
        return out,min_secure,max_secure









    if period == 'Month':
        inbound = df_inbound
        inbound_grouped = inbound.groupby('eta')['quantity'].sum().reset_index()

        min_date = inbound_grouped['eta'].min()
        max_date = inbound_grouped['eta'].max()
        df_demand['eta'] = pd.to_datetime(df_demand['eta'])
        date2 = df_demand['eta'].max() + pd.DateOffset(days=30)

        date_range = pd.date_range(min_date, date2, freq='D')
        new_df = pd.DataFrame({'eta': date_range})
        new_df['eta'] = pd.to_datetime(new_df['eta'])

        result_df = pd.concat([new_df.set_index('eta'), inbound_grouped.set_index('eta')], axis=1,
                                          sort=True).fillna(0)
        inbound = result_df.reset_index()
        inbound = inbound[(inbound['eta'] >= first_date) & (inbound['eta'] <= date2)]
        inbound = np.array(inbound['quantity'])

        # 补货验证：
        # 1：首先创建好数据表格：
        pre_length = pre_length * 30
        demand_yuanshi = np.array(df_demand['quantity'])
        print(demand_yuanshi)
        demand_yuanshi = np.repeat(demand_yuanshi, 30) / 30
        # 如果你需要将结果转换回 DataFrame，可以使用以下代码
        print(demand_yuanshi)
        inbound_yuanshi = inbound
        xiancun_data = np.zeros((pre_length + lead_time + freq + 1))
        buchong_data = np.zeros((pre_length + lead_time + freq))
        daohuo_data = np.zeros((pre_length + lead_time + freq))
        demand_data1 = np.zeros((pre_length))
        demand_data2 = np.zeros((pre_length + lead_time + freq))
        inbound_data = np.zeros((pre_length))


        tjyz = 1.5

        # 安全水平计算

        df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])
        # 按日期排序
        df_inbound = df_inbound.sort_values(by='eta')
        # 计算最近一年的开始日期
        one_year_ago = df_inbound['eta'].max() - pd.DateOffset(years=1)
        # 选择最近一年的数据
        recent_data = df_inbound[df_inbound['eta'] >= one_year_ago]
        # 计算 'quantity' 列的 0.95 分位点
        quantile_95 = recent_data['quantity'].quantile(1)
        min_secure = quantile_95
        mean = recent_data['quantity'].mean()
        max_secure = ((lead_time + freq) * mean + min_secure) * 2

        demand_data2[:len(demand_yuanshi)] = demand_yuanshi * tjyz
        avg_value = np.mean(demand_yuanshi) * tjyz
        for k in range(demand_data1.shape[0], demand_data2.shape[0]):
            demand_data2[k] = avg_value
        demand_data2 = np.sort(demand_data2)[::-1]







        df_asn = df_asn.groupby('asn_create_datetime')['quantity'].sum().reset_index()
        for index, row in df_asn.sort_values(by='asn_create_datetime', ascending=False).iterrows():
            days_diff = (first_date - row['asn_create_datetime']).days
            if days_diff <= lead_time and days_diff > 0:
                print(days_diff)
                daohuo_data[(lead_time - days_diff)] = row['quantity']

        xiancun_data[0] = inventory
        for h in range(lead_time):
            xiancun_data[h + 1] = xiancun_data[h] + daohuo_data[h] - demand_data2[h]
        for m in range(lead_time, len(demand_data1) + lead_time):
            xiancun_data[m + 1] = xiancun_data[m] - demand_data2[m]
            if xiancun_data[m + 1] <= min_secure:
                q = 0
                chu = xiancun_data[m + 1]
                print(chu)
                for n in range(m, m + freq):
                    q = q + demand_data2[n]
                t = q - chu + min_secure

                t = max(t, fahuo) + (baozhuang - (max(t, fahuo) % baozhuang))
                buchong_data[m - lead_time] = t
                daohuo_data[m] = t
                xiancun_data[m + 1] = t + chu
        date_list = [first_date + timedelta(days=i) for i in range(len(demand_yuanshi))]
        date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
        buchong_data = buchong_data[:pre_length]
        buchong = buchong_data.reshape(demand_data1.shape, order='F')
        out = {'date': date_strings, 'buhuo': buchong}
        out = pd.DataFrame(out)
        out_name = 'buhuo_' + '.csv'
        out.to_csv(out_name, index=False)
        return out,min_secure,max_secure




df_asn=pd.read_csv(r'E:\补货实验\a_asn.csv')
df_inbound=pd.read_csv(r'E:\补货实验\a_inbound.csv')
df_demand=pd.read_csv(r'E:\补货实验\a_outbound.csv')
period='Month'
pre_length=1
inventory=4000
fahuo=20
baozhuang=20
buhuo(df_asn,df_inbound,df_demand,period,pre_length,inventory,fahuo,baozhuang)