import pandas as pd

def display_head(df):
    display(df.head(2))

    
def check_null(df):
    print('Checking Null Percentage..')
    return df.isna().sum() * 100/len(df)


def check_duplicate(df, subset):
    print(f'Number of duplicate rows considering {len(subset)} features..')
    if subset is not None: 
        return df.duplicated(subset=subset, keep=False).sum()
    else:
        return df.duplicated(keep=False).sum()

    
def count_unique_values(df, feature_name):
    return df[feature_name].nunique()


def do_value_counts(df, feature_name):
    return df[feature_name].value_counts(normalize=True, dropna=False).sort_values(ascending=False) * 100


def check_id(df, column_name, data_set_name):
    '''
    Check if the identifier column is continous and monotonically increasing
    '''
    print(f'Is the {column_name} monotonic : {df[column_name].is_monotonic}')
    # Plot the column
    ax = df[column_name].plot(title=data_set_name)
    plt.show()
    
    
def get_fetaure_names(df, feature_name_substring) :
    """
    Returns the list of features with name matching 'feature_name_substring'
    """
    return [col_name for col_name in df.columns if col_name.find(feature_name_substring) != -1]


def check_value_counts_across_train_test(train_df, test_df, feature_name, normalize=True):
    """
    Create a DF consisting of value_counts of a particular feature for 
    train and test
    """
    train_counts = train_df[feature_name].sort_index().value_counts(normalize=normalize, dropna=True) * 100
    test_counts = test_df[feature_name].sort_index().value_counts(normalize=normalize, dropna=True) * 100
    count_df = pd.concat([train_counts, test_counts], axis=1).reset_index(drop=True)
    count_df.columns = [feature_name, 'train', 'test']
    return count_df


def get_non_zero_meter_reading_timestamp(df, building_id, start_time, stop_time, meter=0):
    """
    For ASHRAE
    
    For a particular building, when was the first non-zero meter reading appeared.
    given the start and stop time and the type of the meter
    """
    return df[(df.building_id == building_id) 
                & (df.timestamp >= np.datetime64(start_time)) 
                & (df.timestamp < np.datetime64(stop_time)) 
                & (df.meter_reading > 0) & (df.meter == meter)]['timestamp'].iloc[0]