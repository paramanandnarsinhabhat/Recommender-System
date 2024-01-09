import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Reading train file

train_ratings = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/train.csv')

print(train_ratings.head())
print(train_ratings.columns)


#Reading article info file

article_info = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/article_info.csv')

print(article_info.head())

print(article_info.columns)


#Reading test file

test_ratings = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/test.csv')

print(test_ratings.head())

print(test_ratings.columns)

# Displaying the first few rows of each dataset to understand their structure
train_ratings.head(), test_ratings.head(), article_info.head()  

print(train_ratings.head(), test_ratings.head(), article_info.head())

# STEP 3: Merge movie (article) information to ratings dataframe
train_ratings = train_ratings.merge(article_info[['article_id', 'title']], how='left', on='article_id')

# STEP 4: Combine article ID and title separated by ': ' and store it in a new column named 'article'
train_ratings['article'] = train_ratings['article_id'].map(str) + ': ' + train_ratings['title'].map(str)

# STEP 5: Keeping only the columns 'article', 'user_id', and 'rating' in the ratings dataframe
train_ratings = train_ratings.drop(['article_id', 'title'], axis=1)

# Displaying the updated train_ratings dataframe to verify the changes
train_ratings.head()

print(train_ratings.head())

# STEP 6: Creating train & test data & setting evaluation metric

# Splitting the data into training and test datasets
X_train, X_test_split = train_test_split(train_ratings, test_size=0.25, random_state=42)

# Function to compute the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Displaying the first few rows of the train and test splits to verify
X_train.head(), X_test_split.head()
print(X_train.head(), X_test_split.head())

# STEP 7: Implementing the simple baseline using the average of all ratings

# Define the baseline model to always return the average of all available ratings in the train set
def baseline(user_id, movie):
    return X_train['rating'].mean()

# Function to compute the RMSE score obtained on the test set by a model
def rmse_score(model, X_test):
    # Construct a list of user-article tuples from the test dataset
    id_pairs = zip(X_test['user_id'], X_test['article'])
    
    # Predict the rating for every user-article tuple
    y_pred = np.array([model(user, article) for (user, article) in id_pairs])
    
    # Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])
    
    # Return the final RMSE score
    return rmse(y_true, y_pred)

# Calculate and print the RMSE for the baseline model
baseline_rmse = rmse_score(baseline, X_test_split)
print("Baseline rmse is" ,baseline_rmse)

# STEP 8: Item-Based Collaborative Filtering with Simple Item Mean

# STEP 8.1: Build the ratings matrix using pivot_table function
r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='article')

# STEP 8.2: Item-Based Collaborative Filter using Mean Ratings
def cf_item_mean(user_id, movie):
    # Compute the mean of all the ratings given by the user
    mean_rating = r_matrix.loc[user_id].mean()

    return mean_rating

# Compute RMSE for the Mean model
rmse_item_mean = rmse_score(cf_item_mean, X_test_split)
rmse_item_mean
print('RMSE for the Mean model' ,rmse_item_mean)

from sklearn.metrics.pairwise import cosine_similarity


# STEP 9: Item-Based Collaborative Filtering with Similarity Weighted Mean

# Creating a dummy ratings matrix with all null values imputed to 0
r_matrix_dummy = r_matrix.copy().fillna(0)

# Compute the cosine similarity matrix using the dummy ratings matrix
cosine_sim = cosine_similarity(r_matrix_dummy.T, r_matrix_dummy.T)

# Convert into pandas dataframe 
cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.columns, columns=r_matrix.columns)

# Item-Based Collaborative Filter using Weighted Mean Ratings
def cf_item_wmean(user_id, movie_id):
    if movie_id in r_matrix:
        # Get the similarity scores for the item in question with every other item
        sim_scores = cosine_sim[movie_id]

        # Get the movie ratings for the user in question
        m_ratings = r_matrix.loc[user_id]

        # Extract the indices containing NaN in the m_ratings series
        idx = m_ratings[m_ratings.isnull()].index

        # Drop the NaN values from the m_ratings Series and corresponding scores
        m_ratings = m_ratings.dropna()
        sim_scores = sim_scores.drop(idx)

        # Compute the final weighted mean, if denominator is not zero
        if sim_scores.sum() != 0:
            wmean_rating = np.dot(sim_scores, m_ratings) / sim_scores.sum()
        else:
            # Default to user's mean if there are no similar items
            wmean_rating = r_matrix.loc[user_id].mean()
    else:
        # Default to train set mean rating in the absence of any information
        wmean_rating = X_train['rating'].mean()

    return wmean_rating

# Compute RMSE for the Weighted Mean model
rmse_item_wmean = rmse_score(cf_item_wmean, X_test_split)
rmse_item_wmean

print("RMSE for weighted mean model",rmse_item_wmean)

from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms import KNNWithMeans

# STEP 10: Grid Search for Neighbourhood Size and Similarity Measure

# Reader object to import ratings from X_train
reader = Reader(rating_scale=(1, 5))

# Storing Data in surprise format from X_train
data = Dataset.load_from_df(X_train[['user_id', 'article', 'rating']], reader)

# Defining the parameter grid
param_grid = {
    'k': [5, 10, 20],  # Number of nearest neighbors to consider
    'sim_options': {
        'name': ['msd', 'cosine', 'pearson'],  # Different similarity metrics
        'user_based': [False]  # Item-based CF
    }
}

# GridSearchCV for parameter tuning
gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], cv=5, n_jobs=-1)

# Fitting the grid search to the data
gs.fit(data)

# Best RMSE score and the corresponding parameters
best_rmse = gs.best_score['rmse']
best_params = gs.best_params['rmse']

best_rmse, best_params

print("RMSe using Grid search", best_rmse)

# Prepare the Test Dataset by merging with the article info
test_ratings_merged = test_ratings.merge(article_info[['article_id', 'title']], how='left', on='article_id')
test_ratings_merged['article'] = test_ratings_merged['article_id'].map(str) + ': ' + test_ratings_merged['title'].map(str)

# Function to predict ratings using the Weighted Mean model
def predict_ratings(test_set, rating_model):
    predictions = []
    for index, row in test_set.iterrows():
        user_id = row['user_id']
        article_id = row['article']
        predicted_rating = rating_model(user_id, article_id)
        predictions.append(predicted_rating)
    return predictions

# Predicting the ratings for the test set
test_ratings_merged['rating'] = predict_ratings(test_ratings_merged, cf_item_wmean)

# Preparing the final output with required columns
final_output = test_ratings_merged[['user_id', 'article_id', 'rating']]

# Displaying the first few rows of the final output
final_output.head()

# Saving the final output to a CSV file
output_file_path = '/Users/paramanandbhat/Downloads/item based/item_based_predicted_ratings.csv'
final_output.to_csv(output_file_path, index=False)

output_file_path

