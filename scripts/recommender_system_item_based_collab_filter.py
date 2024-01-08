import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Reading train file

train_ratings = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/train.csv')

print(train_ratings.head())
print(train_ratings.columns)

