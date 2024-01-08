
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

#Step 1: Merge Article Information
# Merging article information with the training and test datasets
train_df = train_df.merge(article_info_df, on='article_id', how='left')
test_df = test_df.merge(article_info_df, on='article_id', how='left')


# Splitting the training data into training and validation sets
train_data, validation_data = train_test_split(train_df, test_size=0.25, random_state=42)

# Creating the ratings matrix for the training data
r_matrix = train_data.pivot_table(values='rating', index='user_id', columns='article_id')

# Item-Based Collaborative Filter using Mean Ratings
def cf_item_mean(user_id, article_id):
    # Compute the mean of all the ratings given to the article
    mean_rating = r_matrix[article_id].mean() if article_id in r_matrix else train_data['rating'].mean()
    return mean_rating

# RMSE Score Function
def rmse_score(model, data):
    id_pairs = zip(data['user_id'], data['article_id'])
    y_pred = np.array([model(user, article) for (user, article) in id_pairs])
    y_true = np.array(data['rating'])
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Compute RMSE for the Mean Model on Validation Data
rmse_mean_model = rmse_score(cf_item_mean, validation_data)
rmse_mean_model
print(rmse_mean_model)

from sklearn.metrics.pairwise import cosine_similarity

# Create a dummy ratings matrix with all null values imputed to 0
r_matrix_dummy = r_matrix.copy().fillna(0)

# Compute the cosine similarity matrix using the dummy ratings matrix
cosine_sim = cosine_similarity(r_matrix_dummy.T, r_matrix_dummy.T)

# Convert into pandas dataframe 
cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.columns, columns=r_matrix.columns)

# Item-Based Collaborative Filter using Weighted Mean Ratings
def cf_item_wmean(user_id, article_id):
    if article_id in r_matrix:
        # Get the similarity scores for the item in question with every other item
        sim_scores = cosine_sim[article_id]
        
        # Get the user ratings for the item in question
        m_ratings = r_matrix.loc[user_id]
        
        # Extract the indices containing NaN in the m_ratings series
        idx = m_ratings[m_ratings.isnull()].index
        
        # Drop the NaN values from the m_ratings Series (removing unrated items)
        m_ratings = m_ratings.dropna()
        
        # Drop the corresponding cosine scores from the sim_scores series
        sim_scores = sim_scores.drop(idx)
        
        # Compute the final weighted mean
        wmean_rating = np.dot(sim_scores, m_ratings) / sim_scores.sum() if sim_scores.sum() > 0 else m_ratings.mean()
    else:
        # Default to average rating in the absence of any information on the article in train set
        wmean_rating = train_data['rating'].mean()
    
    return wmean_rating

# Compute RMSE for the Weighted Mean Model on Validation Data
rmse_wmean_model = rmse_score(cf_item_wmean, validation_data)
rmse_wmean_model

print(rmse_wmean_model)


# Saving the test dataset with predicted ratings to a CSV file

test_predictions_wmean = test_df.copy()
output_file_path_wmean = '/Users/paramanandbhat/Downloads/ImplementationforItemBasedCollaborativeFiltering-201024-234420 (1)/itembasedcollab.csv'
test_predictions_wmean.to_csv(output_file_path_wmean, index=False)

# Renaming the 'predicted_rating' column to 'rating'
test_predictions_renamed = test_predictions_wmean.rename(columns={'predicted_rating': 'rating'})

# Saving the modified dataset to a CSV file
output_file_path_wmean_renamed = '/Users/paramanandbhat/Downloads/ImplementationforItemBasedCollaborativeFiltering-201024-234420 (1)/itembasedcollab.csv'
test_predictions_renamed.to_csv(output_file_path_wmean_renamed, index=False)

print(output_file_path_wmean_renamed)


