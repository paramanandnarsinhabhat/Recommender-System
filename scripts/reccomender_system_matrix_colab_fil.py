import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split  # Corrected import
from sklearn.metrics import mean_squared_error

# Custom RMSE Function
def custom_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load 
train_df = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/train.csv')
article_info_df = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/article_info.csv')


print(train_df.columns)
print(train_df.head())

 #Merge Data
train_df_merged = train_df.merge(article_info_df, on='article_id', how='left')

# Split Train Data into Train and Validation Sets
X_train, X_valid = train_test_split(train_df_merged, test_size=0.25, random_state=42)

# Prepare Data for Surprise Library
reader = Reader(rating_scale=(1, 5))
data_train = Dataset.load_from_df(X_train[['user_id', 'article_id', 'rating']], reader)

# Fitting SVD Model with 100 Latent Factors on Train Set
trainset = data_train.build_full_trainset()
model = SVD(n_factors=100)
model.fit(trainset)

# Validate Model
validset = X_valid[['user_id', 'article_id', 'rating']].values.tolist()
predictions_valid = model.test(validset)
y_true_valid = [pred.r_ui for pred in predictions_valid]
y_pred_valid = [pred.est for pred in predictions_valid]
accuracy_valid = custom_rmse(y_true_valid, y_pred_valid)
print(f"Validation RMSE: {accuracy_valid}")

# Grid Search for SVD
param_grid = {'n_factors': list(range(1, 50, 5)), 'n_epochs': [5, 10, 20], 'random_state': [42]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=-1)
gs.fit(data_train)

# Best SVD Model
model_best = SVD(n_factors=gs.best_params['rmse']['n_factors'], n_epochs=gs.best_params['rmse']['n_epochs'], random_state=42)
model_best.fit(trainset)

# Test Set Predictions
test_df = pd.read_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/test.csv')
testset = list(zip(test_df['user_id'], test_df['article_id'], [None]*len(test_df)))
predictions_test = model_best.test(testset)
# Add Predictions to Test DataFrame
test_df['rating'] = [pred.est for pred in predictions_test]

# Rename 'predicted_rating' column to 'rating'
# test_df.rename(columns={'predicted_rating': 'rating'}, inplace=True)

# Save the DataFrame with the new 'rating' column
test_df.to_csv('/Users/paramanandbhat/Downloads/Article_Recommendation 2/matrix_test_with_predictions.csv', index=False)

# Print First Few Rows of Test DataFrame with Predictions
print(test_df.head())
