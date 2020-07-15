"""This module merges the training data with items, item_categories
and shops data. Final output is written to the specified directory

Sample Usage:
    ~/kaggle/predict_future_sales/src$ python -m merge_data
"""

import pandas as pd

import munging.process_data as process_data


def main():
    INPUT_DATA_DIR = '/home/jupyter/kaggle/predict_future_sales/data/read_only'
    OUTPUT_DATA_DIR = '/home/jupyter/kaggle/predict_future_sales/data/processed'

    print(f'Reading data from directory [{INPUT_DATA_DIR}]...')
    train_df, test_df, sample_submission_df, items_df, \
        item_categories_df, shops_df = process_data.read_data(INPUT_DATA_DIR)

    print('Changing data type of train data ..')
    train_df.date = pd.to_datetime(train_df.date, format='%d.%m.%Y')
    train_df.sort_values(['date'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    train_df = process_data.change_dtype(train_df, 'int64', 'int32')
    train_df = process_data.change_dtype(train_df, 'float64', 'float32')

    print('Changing data type of test  data ..')
    test_df = process_data.change_dtype(test_df, 'int64', 'int32')

    print('Changing data type of items  data ..')
    items_df = process_data.change_dtype(items_df, 'int64', 'int32')

    print('Changing data type of item categories  data ..')
    item_categories_df = process_data.change_dtype(
        item_categories_df, 'int64', 'int32')

    print('Changing data type of shops  data ..')
    shops_df = process_data.change_dtype(shops_df, 'int64', 'int32')

    print('Changing data type of submission  data ..')
    sample_submission_df = process_data.change_dtype(
        sample_submission_df, 'int64', 'int32')
    sample_submission_df = process_data.change_dtype(
        sample_submission_df, 'float64', 'float32')

    train_df.to_feather(f'{OUTPUT_DATA_DIR}/train_processed.feather')
    test_df.to_feather(
        f'{OUTPUT_DATA_DIR}/test_processed.feather')
    items_df.to_feather(
        f'{OUTPUT_DATA_DIR}/items_processed.feather')
    item_categories_df.to_feather(
        f'{OUTPUT_DATA_DIR}/item_categories_processed.feather')
    shops_df.to_feather(
        f'{OUTPUT_DATA_DIR}/shops_processed.feather')
    sample_submission_df.to_feather(
        f'{OUTPUT_DATA_DIR}/submission_processed.feather')


if __name__ == "__main__":
    main()
