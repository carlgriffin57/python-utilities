import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_index(df, date_col):
    '''
    Converts column to datetime and sets as the index
    '''
    df[date_col] = pd.to_datetime(df[date_col])
    
    df = df.set_index(date_col).sort_index()
    
    return df

def visualize(df, x, y, title):
    '''
    plots a scatter plot of x vs y, and then a pairplot of the complete df
    '''

    plt.scatter(x=x, y=y)
    plt.title('title')
    plt.show()
    
    sns.pairplot(df)
    
def create_date_columns(df, date_types, date_col):
    '''
    'year','month','day','hour','week','weekday','weekday_name','quarter'
    create columns of these date types using date index or column
    date_col must be set to a pandas datetime
    '''
    
    # if date columns has already been set to index
    if date_col == 'index':
        for x in date_types:
            
            # will add the date column for every date type in the list
            if x == 'year':
                df['year'] = df.index.year
                
            if x == 'month':
                df['month'] = df.index.month
                
            if x == 'day':
                df['day'] = df.index.day
                
            if x == 'hour':
                df['hour'] = df.index.hour
                
            if x == 'week':
                df['week'] = df.index.week
                
            if x == 'weekday':
                df['weekday'] = df.index.weekday
                
            if x == 'weekday_name':
                df['weekday_name'] = df.index.day_name()
                
            if x == 'quarter':
                df['quarter'] = df.index.quarter
                
    # if date column has not yet been set to index
    else:
        for x in date_types:
            
            # will add the date column for every date type in the list
            if x == 'year':
                df['year'] = df[date_col].dt.year
                
            if x == 'month':
                df['month'] = df[date_col].dt.month
                
            if x == 'day':
                df['day'] = df[date_col].dt.day
                
            if x == 'hour':
                df['hour'] = df[date_col].dt.hour
                
            if x == 'week':
                df['week'] = df[date_col].dt.week
                
            if x == 'weekday':
                df['weekday'] = df[date_col].dt.weekday
                
            if x == 'weekday_name':
                df['weekday_name'] = df[date_col].dt.day_name()
                
            if x == 'quarter':
                df['quarter'] = df[date_col].dt.quarter
    
    return df
    
def sales_total():
    '''
    creates a new column for sales total
    '''
    df['sales_total'] = df.sale_amount * df.item_price
    
    return df