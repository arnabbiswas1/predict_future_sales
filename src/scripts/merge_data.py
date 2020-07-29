"""This module merges the training data with items, item_categories
and shops data. Final output is written to the specified directory

Sample Usage:
    ~/kaggle/predict_future_sales/src$ python -m merge_data
"""

import pandas as pd

from config import constants
from common import com_util as util
import munging.process_data as process_data


def main():
    # Create a Stream only logger
    logger = util.get_logger('merge_data')

    logger.info(f'Reading data from directory [{constants.INPUT_DATA_DIR}]...')
    train_df, test_df, sample_submission_df, items_df, \
        item_categories_df, shops_df = process_data.read_raw_data(logger, data_dir=constants.INPUT_DATA_DIR)

    logger.info('Changing data type of train data ..')
    train_df.date = pd.to_datetime(train_df.date, format='%d.%m.%Y')
    train_df.sort_values(['date'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    train_df = process_data.change_dtype(logger, train_df, 'int64', 'int32')
    train_df = process_data.change_dtype(logger, train_df, 'float64', 'float32')

    logger.info('Changing data type of test  data ..')
    test_df = process_data.change_dtype(logger, test_df, 'int64', 'int32')

    logger.info('Changing data type of items  data ..')
    items_df = process_data.change_dtype(logger, items_df, 'int64', 'int32')

    logger.info('Changing data type of item categories  data ..')
    item_categories_df = process_data.change_dtype(logger, item_categories_df, 'int64', 'int32')

    logger.info('Changing data type of shops  data ..')
    shops_df = process_data.change_dtype(logger, shops_df, 'int64', 'int32')

    logger.info('Changing data type of submission  data ..')
    sample_submission_df = process_data.change_dtype(logger, sample_submission_df, 'int64', 'int32')
    sample_submission_df = process_data.change_dtype(logger, sample_submission_df, 'float64', 'float32')

    logger.info(f'Writing processed feather files to {constants.PROCESSED_DATA_DIR}')
    train_df.to_feather(f'{constants.PROCESSED_DATA_DIR}/train_processed.feather')
    test_df.to_feather(f'{constants.PROCESSED_DATA_DIR}/test_processed.feather')
    items_df.to_feather(f'{constants.PROCESSED_DATA_DIR}/items_processed.feather')
    item_categories_df.to_feather(f'{constants.PROCESSED_DATA_DIR}/item_categories_processed.feather')
    shops_df.to_feather(f'{constants.PROCESSED_DATA_DIR}/shops_processed.feather')
    sample_submission_df.to_feather(f'{constants.PROCESSED_DATA_DIR}/submission_processed.feather')

    logger.info('Creating merged test data..')
    test_merged = process_data.prepare_test_data(logger, test_df, items_df, item_categories_df, shops_df)
    test_merged.to_feather(f'{constants.PROCESSED_DATA_DIR}/test_all_merged.feather')

    del test_merged, test_df, sample_submission_df
    util.trigger_gc(logger)

    logger.info('Creating merged training data..')
    train_merged_df = process_data.create_train_data(
        train_df=train_df, items_df=items_df, item_categories_df=item_categories_df, shops_df=shops_df)

    train_merged_df.to_feather(f'{constants.PROCESSED_DATA_DIR}/train_all_merged.feather')
    logger.info('Complete')


if __name__ == "__main__":
    main()
