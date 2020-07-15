import pandas as pd
import numpy as np
from itertools import product


def read_data(data_dir, train=True,
              test=True, sample_submission=True,
              items=True, item_categories=True,
              shops=True):
    """Read all the different data files
    """
    print(f'Reading Data from {data_dir}...')

    train_df = None
    test_df = None
    sample_submission_df = None
    items_df = None
    item_categories_df = None
    shops_df = None

    if train:
        train_df = pd.read_csv(f'{data_dir}/sales_train.csv')
        print(f'Shape of train_df : {train_df.shape}')
    if test:
        test_df = pd.read_csv(f'{data_dir}/test.csv')
        print(f'Shape of test_df : {test_df.shape}')
    if sample_submission:
        sample_submission_df = pd.read_csv(f'{data_dir}/sample_submission.csv')
        print(f'Shape of sample_submission_df : {sample_submission_df.shape}')
    if items:
        items_df = pd.read_csv(f'{data_dir}/items.csv')
        print(f'Shape of items_df : {items_df.shape}')
    if item_categories:
        item_categories_df = pd.read_csv(f'{data_dir}/item_categories.csv')
        print(f'Shape of item_categories_df : {item_categories_df.shape}')
    if shops:
        shops_df = pd.read_csv(f'{data_dir}/shops.csv')
        print(f'Shape of shops_df : {shops_df.shape}')

    return train_df, test_df, sample_submission_df, items_df, item_categories_df, shops_df


def change_dtype(df, source_dtype, target_dtype):
    for col in df.select_dtypes([source_dtype]).columns:
        print(f'Changing dtype of [{col}] from [{source_dtype}] to [{target_dtype}]')
        df[col] = df[col].astype(target_dtype)
    return df


def merge_df(left_df, right_df, how, on):
    """
    Wrapper on top of Pandas merge. Prints the shape & missing 
    values before and after merge
    """
    print(f"Before merge missing values on left_df")
    print(left_df.isna().sum())
    print(f"Before merge missing values on right_df")
    print(right_df.isna().sum())
    print(f"Before merge shape of left_df: {left_df.shape}")
    print(f"Before merge shape of right_df: {right_df.shape}")
    merged_df = pd.merge(left_df, right_df, how=how, on=on)
    print(f"After merge missing values in merged_df")
    print(merged_df.isna().sum())
    print(f"After merge shape of merged_df {merged_df.shape}")
    return merged_df


def create_train_data(train_df):
    """Creates training data for predict future sales
    """
    index_cols = ['shop_id', 'item_id', 'date_block_num']

    # For every month we create a grid from all shops/items
    # combinations from that month
    grid = []
    for block_num in train_df['date_block_num'].unique():
        cur_shops = train_df[
            train_df['date_block_num'] == block_num]['shop_id'].unique()
        cur_items = train_df[
            train_df['date_block_num'] == block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[
            cur_shops, cur_items, [block_num]])), dtype='int32'))

    # Turn the grid into pandas dataframe
    grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

    # Get aggregated values for (shop_id, item_id, month)
    gb = train_df.groupby(
        index_cols, as_index=False).agg({'item_cnt_day': {'target': 'sum'}})

    # Fix column names
    gb.columns = [
        col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    # Join aggregated data to the grid
    all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)
    # Sort the data
    all_data.sort_values([
        'date_block_num', 'shop_id', 'item_id'], inplace=True)
    return all_data
