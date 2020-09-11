"""
Benchamrk with CatBoost. item_cnt_month clipped between o to 20, validation on last month of training data (33).
Final model is built using month upto 32 and final prediction is done on test data using that model.
"""

import os
from timeit import default_timer as timer
from datetime import datetime

import pandas as pd

import common.com_util as common
import config.constants as constants
import cv.cv_util as cv
import modeling.train_util as train

from catboost import Pool


common.set_timezone()
start = timer()

# Create RUN_ID
RUN_ID = datetime.now().strftime("%m%d_%H%M")
MODEL_NAME = os.path.basename(__file__).split('.')[0]

SEED = 42
EXP_DETAILS = "Benchamrk with CatBoost. item_cnt_month clipped between o to 20, validation on last month of training data (33). Final model is built using month upto 32 and final prediction is done on test data using that model."
IS_TEST = False
PLOT_FEATURE_IMPORTANCE = False

TARGET = 'item_cnt_month'
ID = 'ID'

MODEL_TYPE = "cat"
OBJECTIVE = "RMSE"
CUSTOM_METRIC = "RMSE"
EVAL_METRIC = "RMSE"
N_ESTIMATORS = 10000
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 100
USE_BEST_MODEL = True
MAX_DEPTH = 16
MAX_BIN = 254
N_THREADS = -1
BOOSTING_TYPE = "Plain"

cat_params = {
                "objective": OBJECTIVE,  # Alias loss_function
                "custom_metric": CUSTOM_METRIC,
                "eval_metric": CUSTOM_METRIC,
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LEARNING_RATE,
                "random_seed": SEED,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "use_best_model": USE_BEST_MODEL,
                "max_depth": MAX_DEPTH,
                "max_bin": MAX_BIN,
                "thread_count": N_THREADS,
                'boosting_type': BOOSTING_TYPE,
                'verbose_eval': 100,
                }

LOGGER_NAME = 'modeling'
logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
common.set_seed(SEED)
logger.info(f'Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]')

common.update_tracking(
    RUN_ID, "model_number", MODEL_NAME, drop_incomplete_rows=True)
common.update_tracking(RUN_ID, "model_type", MODEL_TYPE)
common.update_tracking(RUN_ID, "is_test", IS_TEST)
common.update_tracking(RUN_ID, "n_estimators", N_ESTIMATORS)
common.update_tracking(RUN_ID, "learning_rate", LEARNING_RATE)
common.update_tracking(RUN_ID, "early_stopping_rounds", EARLY_STOPPING_ROUNDS)

logger.info("Reading data..")
train_df = pd.read_feather('/home/jupyter/kaggle/predict_future_sales/data/processed/train_all_merged.feather')
test_df = pd.read_feather('/home/jupyter/kaggle/predict_future_sales/data/processed/test_all_merged.feather')

train_features = ['shop_id', 'item_id', 'date_block_num', 'item_category_id', 'item_cnt_month']
train_df = train_df[train_features]

test_features = ['ID', 'shop_id', 'item_id', 'date_block_num', 'item_category_id']
test_df = test_df[test_features]
test = test_df.drop(['ID'], axis='columns')

sample_submission = pd.read_feather('/home/jupyter/kaggle/predict_future_sales/data/processed/submission_processed.feather')

# Clip item_cnt_month between 0, 20
train_df.item_cnt_month = train_df.item_cnt_month.clip(lower=0, upper=20)

training_months = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
holdout_months = [33]

logger.info(f"Training months {training_months}")
logger.info(f"Holdout months {holdout_months}")

training, validation = cv.get_data_splits_by_date_block(logger, train_df,
                                                        train_months=training_months,
                                                        validation_months=holdout_months)

predictors = ['shop_id', 'item_id', 'date_block_num', 'item_category_id']

common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)

model, validation_score = train.cat_train_validate_on_holdout(
        logger=logger, training=training, validation=validation,
        predictors=predictors, target=TARGET, cat_features=predictors,
        params=cat_params, test_X=None)

best_iteration = model.get_best_iteration()

logger.info(f"Best iteration {best_iteration}, best validation score {validation_score}")
common.update_tracking(RUN_ID, "validation_type", "holdout")
common.update_tracking(RUN_ID, "best_iteration", best_iteration, is_integer=True)
common.update_tracking(RUN_ID, "best_validation_score", validation_score)

logger.info("Predicting with the model used for holdout validation...")
test_pool = Pool(data=test, feature_names=predictors, cat_features=predictors)
prediction = model.predict(test_pool, ntree_end=best_iteration)

submission = pd.DataFrame({'ID': test_df.ID, 'item_cnt_month': prediction})

logger.info("Saving submission file...")
common.save_file(logger, submission, constants.SUBMISSION_DIR, f"sub_{MODEL_NAME}_{RUN_ID}_cat_baseline.csv")

end = timer()
common.update_tracking(RUN_ID, "training_time", end - start, is_integer=True)
common.update_tracking(RUN_ID, "comments", EXP_DETAILS)
logger.info('Execution Complete')
