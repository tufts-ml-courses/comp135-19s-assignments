import numpy as np
import pandas as pd

x_train_df = pd.read_csv('data_reviews/x_train.csv')
y_train_df = pd.read_csv('data_reviews/y_train.csv')

tr_text_list = x_train_df['text'].values.tolist()
for text in tr_text_list:
    print(text)

