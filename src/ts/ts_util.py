import pandas as pd


def create_date_features(source_df, target_df, feature_name):
    '''
    Create new features related to dates
    
    source_df : DataFrame consisting of the timestamp related feature
    target_df : DataFrame where new features will be added
    feature_name : Name of the feature of date type which needs to be decomposed.
    '''
    target_df.loc[:, 'year'] = source_df.loc[:, feature_name].dt.year.astype('uint16')
    target_df.loc[:, 'month'] = source_df.loc[:, feature_name].dt.month.astype('uint8')
    target_df.loc[:, 'quarter'] = source_df.loc[:, feature_name].dt.quarter.astype('uint8')
    target_df.loc[:, 'weekofyear'] = source_df.loc[:, feature_name].dt.weekofyear.astype('uint8')
    
    target_df.loc[:, 'hour'] = source_df.loc[:, feature_name].dt.hour.astype('uint8')
    #target_df.loc[:, 'minute'] = source_df.loc[:, feature_name].dt.minute.astype('uint32')
    #target_df.loc[:, 'second'] = source_df.loc[:, feature_name].dt.second.astype('uint32')
    
    target_df.loc[:, 'day'] = source_df.loc[:, feature_name].dt.day.astype('uint8')
    target_df.loc[:, 'dayofweek'] = source_df.loc[:, feature_name].dt.dayofweek.astype('uint8')
    target_df.loc[:, 'dayofyear'] = source_df.loc[:, feature_name].dt.dayofyear.astype('uint8')
    target_df.loc[:, 'is_month_start'] = source_df.loc[:, feature_name].dt.is_month_start
    target_df.loc[:, 'is_month_end'] = source_df.loc[:, feature_name].dt.is_month_end
    target_df.loc[:, 'is_quarter_start']= source_df.loc[:, feature_name].dt.is_quarter_start
    target_df.loc[:, 'is_quarter_end'] = source_df.loc[:, feature_name].dt.is_quarter_end
    target_df.loc[:, 'is_year_start'] = source_df.loc[:, feature_name].dt.is_year_start
    target_df.loc[:, 'is_year_end'] = source_df.loc[:, feature_name].dt.is_year_end
    
    # This is of type object
    #target_df.loc[:, 'month_year'] = source_df.loc[:, feature_name].dt.to_period('M')
    
    return target_df


def fill_with_gauss(df, w=12):
    """
    Fill missing values in a time series data using gaussian
    """
    return df.fillna(df.rolling(window=w, win_type='gaussian', center=True, min_periods=1).mean(std=2))


def fill_with_po3(df):
    """
    Fill missing values in a time series data using interpolation (polynomial, order 3)
    """
    df = df.fillna(df.interpolate(method='polynomial', order=3))
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])         


def fill_with_lin(df):
    """
    Fill missing values in a time series data using interpolation (linear)
    """
    df =  df.fillna(df.interpolate(method='linear'))
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])         


def fill_with_mix(df):
    """
    Fill missing values in a time series data using interpolation (linear + polynomial)
    """
    df = (df.fillna(df.interpolate(method='linear', limit_direction='both')) +
               df.fillna(df.interpolate(method='polynomial', order=3, limit_direction='both'))
              ) * 0.5
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])

def find_missing_dates(date_sr, start_date, end_date):
    """
    Returns the dates which are missing in the series
    date_sr between the start_date and end_date
    
    date_sr: Series consisting of date
    start_date: Start date in String format
    end_date: End date in String format
    """
    return pd.date_range(
        start=start_date, end=end_date).difference(date_sr)


def get_first_date_string(date_sr, date_format='%Y-%m-%d'):
    """
    Returns the first date of the series date_sr
    
    date_sr: Series consisting of date
    date_format: Format to be used for converting date into String
    """
    return _get_boundary_date_string(date_sr, boundary='first', date_format='%Y-%m-%d')


def get_last_date_string(date_sr, date_format='%Y-%m-%d'):
    """
    Returns the last date of the series date_sr

    date_sr: Series consisting of date
    date_format: Format to be used for converting date into String
    """
    return _get_boundary_date_string(date_sr, boundary='last', date_format='%Y-%m-%d')


def _get_boundary_date_string(date_sr, boundary, date_format='%Y-%m-%d'):
    """
    Returns the first or last date of the series date_sr based on the 
    value passed in boundary. 
    
    date_sr: Series consisting of date
    boundary: Allowed values are 'first' or 'last'
    date_format: Format to be used for converting date into String
    """
    return date_sr.describe().loc[boundary].strftime(format='%Y-%m-%d')

