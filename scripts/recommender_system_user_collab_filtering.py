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

'''Merged Training Data: Contains user_id, article_id, rating, website, title, and content.
Merged Test Data: Contains user_id, article_id, website, title, and content, but lacks the rating '''

#Step 2: Model Adaptation
from sklearn.model_selection import train_test_split

# Splitting the training data into training and validation sets
train_data, validation_data = train_test_split(train_df, test_size=0.25, random_state=42)

# Function to compute the RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Creating the ratings matrix for the training data
r_matrix = train_data.pivot_table(values='rating', index='user_id', columns='article_id')

# User Based Collaborative Filter using Mean Ratings
def cf_user_mean(user_id, article_id):
    if article_id in r_matrix:
        mean_rating = r_matrix[article_id].mean()
    else:
        mean_rating = train_data['rating'].mean()
    return mean_rating

# RMSE Score Function
def rmse_score(model, data):
    id_pairs = zip(data['user_id'], data['article_id'])
    y_pred = np.array([model(user, article) for (user, article) in id_pairs])
    y_true = np.array(data['rating'])
    return rmse(y_true, y_pred)

# Compute RMSE for the Mean Model on Validation Data
rmse_mean_model = rmse_score(cf_user_mean, validation_data)

print(rmse_mean_model)

# Compute the Pearson Correlation using the ratings matrix
pearson_corr = r_matrix.T.corr(method='pearson')

# User Based Collaborative Filter using Weighted Mean Ratings
def cf_user_wmean(user_id, article_id):
    if article_id in r_matrix:
        # Mean rating for active user
        ra = r_matrix.loc[user_id].mean()
        sim_scores = pearson_corr[user_id]
        # Keep only positive correlations
        sim_scores_pos = sim_scores[sim_scores > 0]
        m_ratings = r_matrix[article_id][sim_scores_pos.index]
        idx = m_ratings[m_ratings.isnull()].index
        m_ratings = m_ratings.dropna()
        if len(m_ratings) == 0:
            wmean_rating = r_matrix[article_id].mean()
        else:   
            sim_scores_pos = sim_scores_pos.drop(idx)
            m_ratings = m_ratings - r_matrix.loc[m_ratings.index].mean(axis=1)
            wmean_rating = ra + (np.dot(sim_scores_pos, m_ratings) / sim_scores_pos.sum())
    else:
        wmean_rating = train_data['rating'].mean()
    
    return wmean_rating

# Compute RMSE for the Weighted Mean Model on Validation Data
rmse_wmean_model = rmse_score(cf_user_wmean, validation_data)
print(rmse_wmean_model)

# Predicting ratings for the test dataset using the User-Based Collaborative Filtering with Weighted Mean Ratings model
test_predictions_wmean = test_df.copy()
test_predictions_wmean['predicted_rating'] = test_predictions_wmean.apply(
    lambda x: cf_user_wmean(x['user_id'], x['article_id']), axis=1)


# Renaming the 'predicted_rating' column to 'rating'
test_predictions_wmean_renamed = test_predictions_wmean.rename(columns={'predicted_rating': 'rating'})

# Saving the test dataset with predicted ratings to a CSV file
output_file_path_wmean = '/Users/paramanandbhat/Downloads/ImplementationforItemBasedCollaborativeFiltering-201024-234420 (1)/test_with_predicted_ratings_wmean.csv'
test_predictions_wmean_renamed.to_csv(output_file_path_wmean, index=False)

print(output_file_path_wmean)





