import  pandas as pd
def make_traindata(df,mode,sku):
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
    return  df
if __name__=='__main__':
    filename1 = 'F:\\all.csv'  # outbound文件
    df = pd.read_csv(filename1)
    mode='week'#'Day' 'Week' 'Month'
    sku=['customer_name', 'customer_part_no', 'site_db']