import os
import pandas as pd

COMPETITION_NAME = 'competitive-data-science-predict-future-sales'

SUBMISSION_DIR = '.'
SUBMISSION_FILE = 'sub_previous_value_benchmark_0724_1810_baseline.csv'
SUBMISSION_MESSAGE = "\"This is the case suggested in Coursera. Monthly sales for Oct 2015 has been used to predict the montly sales of the next month. The monthly sales value is clipped between 0 to 20\""

df = pd.read_csv(f'{SUBMISSION_DIR}/{SUBMISSION_FILE}')
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
