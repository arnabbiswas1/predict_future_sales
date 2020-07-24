import os
from timeit import default_timer as timer
from datetime import datetime

import common.com_util as common
import config.constants as constants
from munging import process_data as process_data

common.set_timezone()
start = timer()

# Create run_id
RUN_ID = datetime.now().strftime("%m%d_%H%M")
MODEL_NAME = os.path.basename(__file__).split('.')[0]

SEED = 42
EXP_DETAILS = "Predict based on sale of items from 2005 October"
IS_TEST = False
PLOT_FEATURE_IMPORTANCE = False

# General configuration stuff
LOGGER_NAME = 'modeling'

logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
common.set_seed(SEED)
logger.info(f'Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]')

common.update_tracking(
    RUN_ID, "model_number", MODEL_NAME, drop_incomplete_rows=True)
common.update_tracking(RUN_ID, "model_type", MODEL_NAME)
common.update_tracking(RUN_ID, "is_test", IS_TEST)

train_df, test_df, submission_df, _, _, _ = process_data.read_processed_data(
    logger,
    constants.PROCESSED_DATA_DIR, train=True,
    test=True, sample_submission=True,
    items=False, item_categories=False,
    shops=False)

train_df_sel_month = train_df[
    (train_df.date.dt.year == 2015) & (train_df.date.dt.month == 10)]
train_sel_month_summarized = train_df_sel_month.groupby(
    ['shop_id', 'item_id'])['item_cnt_day'].sum()
train_sel_month_summarized = train_sel_month_summarized.reset_index()
train_sel_month_summarized.rename(
    columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)

test_df_filled_oct_2015 = process_data.merge_df(logger, test_df, train_sel_month_summarized, how='left', on=['shop_id', 'item_id'])

percent_of_null = test_df_filled_oct_2015.item_cnt_month.isna().sum()*100/len(test_df_filled_oct_2015)
logger.info(f'Percent of missing values for item_cnt_month in test data: {percent_of_null}')

test_df_filled_oct_2015.item_cnt_month.fillna(value=0, inplace=True)
test_df_filled_oct_2015.item_cnt_month.clip(lower=0, upper=20, inplace=True)
submission = test_df_filled_oct_2015[['ID', 'item_cnt_month']]

common.save_file(logger, submission, constants.SUBMISSION_DIR, f"sub_{MODEL_NAME}_{RUN_ID}_baseline.csv")

end = timer()
common.update_tracking(RUN_ID, "training_time", end - start, is_integer=True)
common.update_tracking(RUN_ID, "comments", EXP_DETAILS)
logger.info('Execution Complete')
