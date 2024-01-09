import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

## 1. Reading Dataset 
#Reading ratings file:
ratings = pd.read_csv('/Users/paramanandbhat/Downloads/MatrixfactorizationBasedCollaborativeFilteringusingSurprise-201024-234535 (1)/ratings.csv')

#Reading Movie Info File
movie_info = pd.read_csv('/Users/paramanandbhat/Downloads/MatrixfactorizationBasedCollaborativeFilteringusingSurprise-201024-234535 (1)/movie_info.csv')

## 2.  Merging Movie information to ratings dataframe   

ratings = ratings.merge(movie_info[['movie id','movie title']], how='left', left_on = 'movie_id', right_on = 'movie id')

print(ratings.head())

ratings['movie'] = ratings['movie_id'].map(str) + str(': ') + ratings['movie title'].map(str)

print(ratings.columns)
ratings = ratings.drop(['movie id', 'movie title', 'movie_id','unix_timestamp'], axis = 1)

ratings = ratings[['user_id','movie','rating']]

## 3. Creating Train & Test Data & Setting Evaluation Metric
#Assign X as the original ratings dataframe
X = ratings.copy()
#Split into training and test datasets
X_train, X_test = train_test_split(X, test_size = 0.25, random_state=42)

#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

## 4. Importing Surprise & Loading Dataset
#Importing functions to be used in this notebook from Surprise Package
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

#Reader object to import ratings from X_train
reader = Reader(rating_scale=(1, 5))

#Storing Data in surprise format from X_train
data = Dataset.load_from_df(X_train[['user_id','movie','rating']], reader)                          

 ## 5. Fitting SVD Model with 100 latent factors on train set and checking performance on test set 
# Train a new SVD with 100 latent features (number was chosen arbitrarily)
model = SVD(n_factors=100)

#Build full trainset will essentially fits the knnwithmeans on the complete train set instead of a part of it
#like we do in cross validation
model.fit(data.build_full_trainset())

#id pairs for test set
id_pairs = zip(X_test['user_id'], X_test['movie'])

#Making predictions for test set using predict method from Surprise
y_pred = [model.predict(uid = user, iid = movie)[3] for (user, movie) in id_pairs]

#Actual rating values for test set
y_true = X_test['rating']

# Checking performance on test set
rmse(y_true, y_pred)

print(rmse(y_true, y_pred))

## 6. Examining the user and item matrices
#Number of movies & users in train data
X_train.movie.nunique(), X_train.user_id.nunique()


# 1642*100 (movie matrix)  943*100 (user matrix) # 1642*943 (user movie matrix)
model.qi.shape, model.pu.shape,X_train.movie.nunique(), X_train.user_id.nunique() 

#Percentage reduction in size wrt user item matrix
(1642*943 - 943*100 - 1642*100)/(1642*943)*100

#Extracting id for Toy story within qi matrix
movie_row_idx = model.trainset._raw2inner_id_items['1: Toy Story (1995)']
np.array(model.qi[movie_row_idx])

#Latent factors learnt from Funk SVD
ts_vector = np.array(model.qi[movie_row_idx])

#Extracting id for Wizard of Oz within qi matrix
movie_row_idx = model.trainset._raw2inner_id_items['132: Wizard of Oz, The (1939)']
woz_vector = np.array(model.qi[movie_row_idx])

#Checking the similarity in latent factors for wizard of oz & Toy Story
from scipy import spatial
1 - spatial.distance.cosine(ts_vector,woz_vector)

## 7. Grid Search for better performance with SVD
#Defining the parameter grid for SVD and fixing the random state
param_grid = {'n_factors':list(range(1,50,5)), 'n_epochs': [5, 10, 20], 'random_state': [42]}

#Defining the grid search with the parameter grid and SVD algorithm optimizing for RMSE
gs = GridSearchCV(SVD, 
                  param_grid, 
                  measures=['rmse'], 
                  cv=5, 
                  n_jobs = -1)

#Fitting the mo
gs.fit(data)
 
#Printing the best score
print(gs.best_score['rmse'])

#Printing the best set of parameters
print(gs.best_params['rmse'])

#Fitting the model on train data with the best parameters
model = SVD(n_factors = 11, n_epochs = 20, random_state = 42)

#Build full trainset will essentially fits the SVD on the complete train set instead of a part of it
#like we do in cross validation for grid search
model.fit(data.build_full_trainset())

# aid pairs for test set
id_pairs = zip(X_test['user_id'], X_test['movie'])

#Making predictions for test set using predict method from Surprise
y_pred = [model.predict(uid = user, iid = movie)[3] for (user, movie) in id_pairs]

#Actual rating values for test set
y_true = X_test['rating']

# Checking performance on test set
rmse(y_true, y_pred)

print(rmse(y_true, y_pred))


