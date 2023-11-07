import  pandas as pd

#按天
df=pd.read_csv(r'E:/qqq.csv')
grouped=df.groupby(['customer_name', 'customer_part_no'])
def operation(group):
    date_range = pd.date_range(group['eta'].min(), group['eta'].max(), freq='D')
    # # # # 使用 merge 将原始数据与完整的日期范围合并，并使用 fillna 填充缺失的值
    fill_values = {'quantity': 0,'customer_name':group['customer_name'].iloc[0], 'customer_part_no':group['customer_part_no'].iloc[0]}
    group = date_range.to_frame(index=False, name='eta').merge(group, on='eta', how='left').fillna(fill_values)
    group['time_idx'] = group['eta'].apply(lambda x: int(x.toordinal()))
    group["time_idx"] -= group["time_idx"].min()
    group = group.sort_values(by="time_idx")
    group['2_mean'] = group['quantity'].rolling(window=2, min_periods=1).mean()
    group['3_mean'] = group['quantity'].rolling(window=3, min_periods=1).mean()
    group['4_mean'] = group['quantity'].rolling(window=4, min_periods=1).mean()
    group['5_mean'] = group['quantity'].rolling(window=5, min_periods=1).mean()
    group['7_mean'] = group['quantity'].rolling(window=7, min_periods=1).mean()
    group['14_mean'] = group['quantity'].rolling(window=14, min_periods=1).mean()
    group['28_mean'] = group['quantity'].rolling(window=28, min_periods=1).mean()
    group['2_max'] = group['quantity'].rolling(window=2, min_periods=1).max()
    group['2_min'] = group['quantity'].rolling(window=2, min_periods=1).min()
    group['4_max'] = group['quantity'].rolling(window=4, min_periods=1).max()
    group['4_min'] = group['quantity'].rolling(window=4, min_periods=1).min()
    group['7_max'] = group['quantity'].rolling(window=7, min_periods=1).max()
    group['7_min'] = group['quantity'].rolling(window=7, min_periods=1).min()
    group['lag'] = group['quantity'].shift().fillna(0)
    return group
# 对每个分组应用操作，并使用 concat 将结果拼接成一个新的DataFrame
new_df = pd.concat([operation(group) for name, group in grouped  if len(operation(group)) > 30],ignore_index=True)
new_df.to_csv('qqq.csv')



#########按周或者按，两周，三周，月
df['eta'] = pd.to_datetime(df['eta'])
grouped = df.groupby(['customer_name', 'customer_part_no', 'site_db'])
def sum_by_week(group):
    return group.resample('W-Mon', on='eta')['quantity'].sum()##这里可以把W-Mon改成2W-Mon，3W-Mon，4W-Mon，代表2周，三周，四周。
# 对每组应用上述函数，得到结果
result_df = grouped.apply(sum_by_week).reset_index()
result_df.to_csv('week_outbound.csv')