import gc
import random
import logging
import time
import sys
import os

import pandas as pd
import numpy as np

import config.constants as constants
import viz.plot_util as plot_util


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def trigger_gc():
    """
    Trigger GC
    """
    print(gc.collect())


def set_timezone():
    """
    Sets the time zone to Kolkata.
    """
    os.environ["TZ"] = "Asia/Calcutta"
    time.tzset()


def get_logger(logger_name, model_number, run_id, path):
    """
    https://realpython.com/python-logging/
    https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    """
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    s_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(f'{path}/{model_number}_{run_id}.log')
    formatter = logging.Formatter(FORMAT)
    s_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    logger.addHandler(s_handler)
    return logger


def update_tracking(run_id,
                    field,
                    value,
                    csv_file=constants.TRACKING_FILE,
                    is_integer=False,
                    no_of_digits=None,
                    drop_incomplete_rows=False):
    """
    Function to update the tracking CSV with information about the model

    https://github.com/RobMulla/kaggle-ieee-fraud-detection/blob/master/scripts/M001.py#L98

    """
    try:
        df = pd.read_csv(csv_file, index_col=[0])
        df['lb_score'] = 0

        # If the file exists, drop rows (without final results)
        # for previous runs which has been stopped inbetween.
        if (drop_incomplete_rows & ('oof_score' in df.columns)):
            df = df.loc[~df['oof_score'].isna()]

    except FileNotFoundError:
        df = pd.DataFrame()

    if is_integer:
        value = round(value)
    elif no_of_digits is not None:
        value = round(value, no_of_digits)

    # Model number is index
    df.loc[run_id, field] = value
    df.to_csv(csv_file)


def save_file(logger, df, dir_name, file_name):
    """
    Utility method to save submission, off files etc.
    """
    logger.info(f'Saving {dir_name}/{file_name}')
    df.to_csv(f'{dir_name}/{file_name}', index=False)


def save_artifacts(logger, is_test, is_plot_fi,
                   result_dict,
                   submission_df,
                   model_number,
                   run_id, sub_dir, oof_dir, fi_dir, fi_fig_dir):
    """
    Save the submission, OOF predictions, feature importance values
    and plos to different directories.
    """
    score = result_dict['avg_cv_scores']

    if is_test is False:
        # Save submission file
        submission_df.target = result_dict['prediction']
        save_file(logger,
                  submission_df,
                  sub_dir,
                  f'sub_{model_number}_{run_id}_{score:.4f}.csv')

        # Save OOF
        oof_df = pd.DataFrame(result_dict['yoof'])
        save_file(logger,
                  oof_df,
                  oof_dir,
                  f'oof_{model_number}_{run_id}_{score:.4f}.csv')

    if is_plot_fi is True:
        # Feature Importance
        feature_importance_df = result_dict['feature_importance']
        save_file(logger,
                  feature_importance_df,
                  fi_dir,
                  f'fi_{model_number}_{run_id}_{score:.4f}.csv')

        # Save the plot
        best_features = result_dict['best_features']
        plot_util.save_feature_importance_as_fig(
            best_features, fi_fig_dir,
            f'fi_{model_number}_{run_id}_{score:.4f}.png')
