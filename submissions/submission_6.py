"""Submission generated by cat_benchmark_w_entire_data.py on 11th September, 2020
"""

import os
import pandas as pd

COMPETITION_NAME = 'competitive-data-science-predict-future-sales'

SUBMISSION_DIR = '.'
SUBMISSION_FILE = 'sub_cat_benchmark_w_entire_data_0911_1318_cat_baseline_full_training_data.csv'
SUBMISSION_MESSAGE = "\"Benchamrk with CatBoost. item_cnt_month clipped between o to 20, validation on last month of training data (33). Final model is built using all data (including month 33) and final prediction is done on test data using that model.\""

df = pd.read_csv(f'{SUBMISSION_DIR}/{SUBMISSION_FILE}')
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)