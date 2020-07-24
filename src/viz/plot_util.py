import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_meter_reading_for_site(df, site_id, meter_name):
    """
    Plot meter_reading for an entire site for all buildings
    """
    df = df.set_index('timestamp')
    building_id_list = df.columns
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df.loc[:, building_id]
        fig.add_trace(go.Scatter(
             x=df_subset.index,
             y=df_subset.values,
             name=f"{meter_name}",
             hoverinfo=f'x+y+name',
             opacity=0.7))

        fig.update_layout(width=1000,
                        height=500,
                        title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                        xaxis_title="timestamp",
                        yaxis_title="meter_reading",)
        fig.show()
        

def plot_meter_reading_for_building(df, site_id, building_id, meter_name):
    """
    Plot meter_reading for an entire site for all buildings
    """
    df = df.set_index('timestamp')
    building_id_list = df.columns
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df.loc[:, building_id]
        fig.add_trace(go.Scatter(
             x=df_subset.index,
             y=df_subset.values,
             name=f"{meter_name}",
             hoverinfo=f'x+y+name',
             opacity=0.7))

        fig.update_layout(width=1000,
                        height=500,
                        title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                        xaxis_title="timestamp",
                        yaxis_title="meter_reading",)
        fig.show()
        

def display_all_site_meter_reading(df, site_id=0, meter=0):
    """
    Plot meter reading for the entire site for a particular type of meter 
    """
    df_meter_subset = df[(df.site_id == site_id) & (df.meter == meter)]
    df_meter_subset = df_meter_subset.pivot(index='timestamp', columns='building_id', values='meter_reading')

    column_names = df_meter_subset.reset_index().columns.values
    df_meter_subset.reset_index(inplace=True)
    df_meter_subset.columns = column_names
    
    print(f'Missing Values for {site_id}')
    display(df_meter_subset.isna().sum())
    
    plot_meter_reading_for_site(df_meter_subset, site_id, meter_dict[meter])


def plot_hist_train_test_overlapping(df_train, df_test, feature_name, kind='hist'):
    """
    Plot histogram for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    df_train[feature_name].plot(kind=kind, figsize=(15, 5), label='train', 
                         bins=50, alpha=0.4, 
                         title=f'Train vs Test {feature_name} distribution')
    df_test[feature_name].plot(kind='hist',label='test', bins=50, alpha=0.4)
    plt.legend()
    plt.show()
    

def plot_barh_train_test_side_by_side(df_train, df_test, feature_name, normalize=True, sort_index=False):
    """
    Plot histogram for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    print(f'Number of unique values in train : {count_unique_values(df_train, feature_name)}')
    print(f'Number of unique values in test : {count_unique_values(df_test, feature_name)}')
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 8))
    
    if sort_index == True:
            df_train[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_index().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax1,
                grid=True,
                title=f'Bar plot for {feature_name} for train')
    
            df_test[feature_name].value_counts(
                    normalize=normalize, dropna=False).sort_index().plot(
                    kind='barh', figsize=(15, 5), 
                    ax=ax2,
                    grid=True,
                    title=f'Bar plot for {feature_name} for test')
    else:
        df_train[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax1,
                grid=True,
                title=f'Bar plot for {feature_name} for train')

        df_test[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax2,
                grid=True,
                title=f'Bar plot for {feature_name} for test')

    
    plt.legend()
    plt.show()
    
    
def plot_line_train_test_overlapping(df_train, df_test, feature_name):
    """
    Plot line for a particular feature both for train and test
    """
    df_train[feature_name].plot(kind='line', figsize=(10, 5), label='train', 
                          alpha=0.4, 
                         title=f'Train vs Test {feature_name} distribution')
    df_test[feature_name].plot(kind='line',label='test', alpha=0.4)
    plt.ylabel(f'Value of {feature_name}')
    plt.legend()
    plt.show()
    
    
def plot_hist(df, feature_name, kind='hist', bins=100, log=True):
    """
    Plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(kind='hist', 
                                              bins=bins, 
                                              figsize=(15, 5), 
                                              title=f'Distribution of log1p[{feature_name}]')
    else:
        df[feature_name].plot(kind='hist', 
                              bins=bins, 
                              figsize=(15, 5), 
                              title=f'Distribution of {feature_name}')
    plt.show()


def plot_barh(df, feature_name, normalize=True, kind='barh', figsize=(15,5), sort_index=False):
    """
    Plot barh for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    if sort_index==True:
        df[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_index().plot(
                kind=kind, figsize=figsize, grid=True,
                title=f'Bar plot for {feature_name}')
    else:   
        df[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind=kind, figsize=figsize, grid=True,
                title=f'Bar plot for {feature_name}')
    
    plt.legend()
    plt.show()
    

def plot_boxh(df, feature_name, kind='box', log=True):
    """
    Box plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(kind='box', vert=False, 
                                                  figsize=(10, 6), 
                                                  title=f'Distribution of log1p[{feature_name}]')
    else:
        df[feature_name].plot(kind='box', vert=False, 
                              figsize=(10, 6), 
                              title=f'Distribution of {feature_name}')
    plt.show()
    
    
def plot_boxh_groupby(df, feature_name, by):
    """
    Box plot with groupby feature
    """
    df.boxplot(column=feature_name, by=by, vert=False, 
                              figsize=(10, 6))
    plt.title(f'Distribution of {feature_name} by {by}')
    plt.show()


def save_feature_importance_as_fig(best_features_df, dir_name, file_name):
    plt.figure(figsize=(16, 12))
    sns.barplot(
        x="importance", y="feature", data=best_features_df.sort_values(
            by="importance", ascending=False))
    plt.title('Features (avg over folds)')
    plt.savefig(f'{dir_name}/{file_name}')