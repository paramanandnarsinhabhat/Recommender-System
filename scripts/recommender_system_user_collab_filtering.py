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


#Step 1: Data Preparation
'''We'll start by merging the article_info dataset with the
 train and test datasets. This is to ensure that the article details 
 are incorporated into the model training and evaluation process.'''


# Merging article information with the training and test datasets

train_df = train_df.merge(article_info_df, on='article_id', how='left')

print(train_df.columns)

test_df = test_df.merge(article_info_df, on='article_id', how='left')

print(test_df.columns)

# Displaying the first few rows of the merged training and test datasets
merged_data_samples = {
    "Merged Training Data": train_df.head(),
    "Merged Test Data": test_df.head()
}

print(merged_data_samples)
