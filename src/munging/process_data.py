import pandas as pd
import numpy as np
import itertools as itertools


def read_raw_data(logger, data_dir, train=True,
                  test=True, sample_submission=True,
                  items=True, item_categories=True,
                  shops=True):
    """Read all the different data files
    """
    logger.info(f'Reading Data from {data_dir}...')

    train_df = None
    test_df = None
    sample_submission_df = None
    items_df = None
    item_categories_df = None
    shops_df = None

    if train:
        train_df = pd.read_csv(f'{data_dir}/sales_train.csv')
        logger.info(f'Shape of train_df : {train_df.shape}')
    if test:
        test_df = pd.read_csv(f'{data_dir}/test.csv')
        logger.info(f'Shape of test_df : {test_df.shape}')
    if sample_submission:
        sample_submission_df = pd.read_csv(f'{data_dir}/sample_submission.csv')
        logger.info(f'Shape of sample_submission_df : {sample_submission_df.shape}')
    if items:
        items_df = pd.read_csv(f'{data_dir}/items.csv')
        logger.info(f'Shape of items_df : {items_df.shape}')
    if item_categories:
        item_categories_df = pd.read_csv(f'{data_dir}/item_categories.csv')
        logger.info(f'Shape of item_categories_df : {item_categories_df.shape}')
    if shops:
        shops_df = pd.read_csv(f'{data_dir}/shops.csv')
        logger.info(f'Shape of shops_df : {shops_df.shape}')

    return train_df, test_df, sample_submission_df, items_df, item_categories_df, shops_df


def read_processed_data(logger, data_dir, train=True, test=True,
                        sample_submission=True, items=True,
                        item_categories=True, shops=True):
    """Read all the processed data files
    """
    logger.info(f'Reading Data from {data_dir}...')

    train_df = None
    test_df = None
    sample_submission_df = None
    items_df = None
    item_categories_df = None
    shops_df = None

    if train:
        train_df = pd.read_feather(f'{data_dir}/train_processed.feather')
        logger.info(f'Shape of train_df : {train_df.shape}')
    if test:
        test_df = pd.read_feather(f'{data_dir}/test_processed.feather')
        logger.info(f'Shape of test_df : {test_df.shape}')
    if sample_submission:
        sample_submission_df = pd.read_feather(f'{data_dir}/submission_processed.feather')
        logger.info(f'Shape of sample_submission_df : {sample_submission_df.shape}')
    if items:
        items_df = pd.read_feather(f'{data_dir}/items_processed.feather')
        logger.info(f'Shape of items_df : {items_df.shape}')
    if item_categories:
        item_categories_df = pd.read_feather(f'{data_dir}/item_categories_processed.feather')
        logger.info(f'Shape of item_categories_df : {item_categories_df.shape}')
    if shops:
        shops_df = pd.read_feather(f'{data_dir}/shops_processed.feather')
        logger.info(f'Shape of shops_df : {shops_df.shape}')

    return train_df, test_df, sample_submission_df, items_df, item_categories_df, shops_df


def change_dtype(logger, df, source_dtype, target_dtype):
    for col in df.select_dtypes([source_dtype]).columns:
        logger.info(f'Changing dtype of [{col}] from [{source_dtype}] to [{target_dtype}]')
        df[col] = df[col].astype(target_dtype)
    return df


def merge_df(logger, left_df, right_df, how, on):
    """
    Wrapper on top of Pandas merge. Prints the shape & missing
    values before and after merge
    """
    logger.info(f"Before merge missing values on left_df")
    logger.info(left_df.isna().sum())
    logger.info(f"Before merge missing values on right_df")
    logger.info(right_df.isna().sum())
    logger.info(f"Before merge shape of left_df: {left_df.shape}")
    logger.info(f"Before merge shape of right_df: {right_df.shape}")
    merged_df = pd.merge(left_df, right_df, how=how, on=on)
    logger.info(f"After merge missing values in merged_df")
    logger.info(merged_df.isna().sum())
    logger.info(f"After merge shape of merged_df {merged_df.shape}")
    return merged_df


def create_train_data(train_df, items_df, item_categories_df, shops_df):
    """Creates training data
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
        grid.append(np.array(list(itertools.product(*[
            cur_shops, cur_items, [block_num]])), dtype='int32'))

    # Turn the grid into pandas dataframe
    grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

    # Add item category ids to DF
    grid = pd.merge(grid, items_df, how='left', on=['item_id'])
    # Add item category names to DF
    grid = pd.merge(grid, item_categories_df, how='left', on='item_category_id')
    # Add shop names to DF
    grid = pd.merge(grid, shops_df, how='left', on='shop_id')

    # For existing training data generated features
    gb = train_df.groupby(index_cols).agg(
        item_cnt_month=('item_cnt_day', sum), item_price_mean=('item_price', np.mean), item_price_variance=('item_price', np.var)).reset_index()

    # Join aggregated data to the grid
    all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)
    # Sort the data
    all_data.sort_values([
        'date_block_num', 'shop_id', 'item_id'], inplace=True)
    all_data.reset_index(drop=True, inplace=True)
    return all_data


def prepare_test_data(logger, test_df, items_df, item_categories_df, shops_df):
    """
    Merge test data with shops, items and item categories
    """
    logger.info("Merging test_df with items ....")
    test_df = pd.merge(test_df, items_df, how='left', on='item_id')
    logger.info("Merging with item_categories ....")
    test_df = pd.merge(test_df, item_categories_df, how='left', on='item_category_id')
    logger.info("Merging with shops ....")
    test_df = pd.merge(test_df, shops_df, how='left', on='shop_id')

    test_df['date_block_num'] = 34
    test_df['date_block_num'] = test_df['date_block_num'].astype('int32')
    test_df = test_df.reset_index(drop=True)
    logger.info(f'Shape of the final DF {test_df.shape}')
    return test_df
