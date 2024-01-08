#Step 1 : Reading Dataset

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#Reading train file

train_df = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/train.csv')

print(train_df.head())
print(train_df.columns)

#Reading article info file

article_info_df = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/article_info.csv')

print(article_info_df.head())

print(article_info_df.columns)


#Reading test file

test_df = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/test.csv')

print(test_df.head())

print(test_df.columns)


