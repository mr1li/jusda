import pandas as pd
import pandas as pd
import os
import numpy as np
from datetime import datetime,timedelta
def buhuo(file1,file2,file3,date,initial_inventory,a1,a2,a3,customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db):
    df1 = pd.read_csv(file1, encoding='latin1')
    # 读取第二个CSV文件
    df2 = pd.read_csv(file2, encoding='latin1')
    # 合并两个DataFrame，根据指定的列进行匹配
    merged_df = pd.merge(df1, df2, on=["supplier_name", "supplier_part_no", "customer_name", "customer_part_no", "manufacture_name", "site", "site_db", "asn_no"])
    # 选择需要的列
    result_df = merged_df[["supplier_name", "supplier_part_no", "customer_name", "customer_part_no", "manufacture_name", "site", "site_db", "asn_no", "quantity1", "asn_create_datetime", "quantity2", "eta"]]
    sorted_df = result_df.sort_values(by=["supplier_name", "supplier_part_no", "customer_name", "customer_part_no", "manufacture_name", "site", "site_db", "asn_no"])
    # 保存到新的CSV文件
    # sorted_df.to_csv('sorted_file.csv', index=False)
    df= sorted_df[(result_df['customer_name'] == customer_name) & (result_df['customer_part_no'] ==customer_part_no)]
    #求交付期
    df['asn_create_datetime'] = pd.to_datetime(df['asn_create_datetime']).dt.date
    df['eta'] = pd.to_datetime(df['eta']).dt.date
    df['time_difference'] = (df['eta'] - df['asn_create_datetime']).dt.days
    average_time_difference = df['time_difference'].mean()
    lead_time=round(int(average_time_difference))
    print(lead_time)
    #求补货频率
    max_date = df['asn_create_datetime'].max()
    three_months_ago = max_date - pd.DateOffset(months=3)
    # 仅保留在最近三个月内的数据
    df_recent = df[df['asn_create_datetime'] >= three_months_ago]
    # 计算频率
    freq = (pd.to_datetime(max_date )-  pd.to_datetime(three_months_ago)).days/len(df_recent)
    # 计算每一行的'eta'列与'asn_create_datetime'之差的平均值
    freq=round(freq)

    name=customer_name + '_' + customer_part_no+'_' +supplier_name + '_'+supplier_part_no + '_'+manufacture_name + '_'+site_db
    base_name=file3+'//'+'model_'+name
    forecast_name=base_name+'//'+'forecast_'+name+'.csv'
    error_name=base_name+'//'+'error_'+name+'.npy'
    demand_data=pd.read_csv(forecast_name)
    demand_data=np.array((demand_data['mean']))

    error=np.load(error_name)
    demand_data2=error[int(a3*len(error))]*demand_data
    #先算出交付期内的数据
    daohuo=np.zeros(len(demand_data))
    xiancun=np.zeros(len(demand_data))
    buchong=np.zeros(len(demand_data))
    df['asn_create_datetime'] = pd.to_datetime(df['asn_create_datetime'])

    today = pd.to_datetime(date)
    for index, row in df.sort_values(by='asn_create_datetime', ascending=False).iterrows():
        days_diff = (today - row['asn_create_datetime']).days
        if days_diff <= lead_time:
            daohuo[days_diff] = row['quantity1']

    xiancun[0]=initial_inventory
    for i in range(lead_time):
        if i==0:
            xiancun[i]=xiancun[i]+daohuo[i]-demand_data2[i]
        else:
            xiancun[i]=xiancun[i-1]+daohuo[i]-demand_data2[i]

    #开始补货
    for i in range(lead_time,len(demand_data)-freq-3):
        xiancun[i]=xiancun[i-1]-demand_data2[i]
        if xiancun[i]<=demand_data2[i+1]+demand_data2[i+2]+demand_data2[i+3]:
            q=0
            chu=xiancun[i]
            for i in range(freq+3):
                q=q+demand_data2[i]
            xiancun[i]=q
            buchong[i-lead_time]=q-chu





    #保存文件
    date_list = [today + timedelta(days=i) for i in range(len(demand_data))]
    date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
    out= {'date':date_strings,'buhuo': buchong}
    print(out)
    out=pd.DataFrame(out)
    out_name=base_name+'buhuo_'+name+'.csv'
    print(out_name)
    out.to_csv(out_name,index=False,line_terminator='\n')


    # filtered_data.to_csv('a.csv')
    # print(filtered_data)
if __name__ == "__main__":
    file1=r'E:\JusLink＆浙大数据20230523\v_islm_asn_1.csv'
    file2=r'E:\JusLink＆浙大数据20230523\v_islm_inbound.csv'
    file3='E://mxnet//tmp'
    date='2022-08-06'
    initial_inventory=10000
    zuixiaobaozhuang=100
    zuixiaofahuo=100
    xuqiumanzu=0.8
    customer_name='7e0d847c78df7e95d8e8779df7557fbc'
    customer_part_no='2fa8c3e5b021c333cff52f4280171a7b'
    supplier_name='a'
    supplier_part_no='a'
    manufacture_name='a'
    site_db='a'
    buhuo(file1,file2,file3,date,initial_inventory,zuixiaobaozhuang,zuixiaofahuo,xuqiumanzu,customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db)
