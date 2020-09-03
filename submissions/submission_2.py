import os
import pandas as pd

COMPETITION_NAME = 'competitive-data-science-predict-future-sales'

SUBMISSION_DIR = '.'
SUBMISSION_FILE = 'sub_lgb_benchmark_0903_1149_lgb_baseline.csv'
SUBMISSION_MESSAGE = "\"Benchamrk with LGB item_cnt_month clipped between o to 20, validation on month before prediction (33)\""

df = pd.read_csv(f'{SUBMISSION_DIR}/{SUBMISSION_FILE}')
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
