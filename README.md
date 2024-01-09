# Recommender System Analysis

## Overview

This repository contains the implementation and analysis of various collaborative filtering methods used to build an efficient recommender system. We have explored several models to determine the most accurate in predicting user preferences.

## Models Evaluated

- **Matrix-Based Collaborative Filtering**: Utilizes singular value decomposition to predict user preferences with a high degree of accuracy.
- **Item-Based Collaborative Filtering (Weighted Mean Model)**: Predicts user preferences based on item similarities and weighted means.
- **User-Based Collaborative Filtering**: Makes predictions based on the preferences of similar users.
- **Mean Model**: A baseline model that predicts average ratings.

## Performance

The RMSE (Root Mean Square Error) was used as the benchmark for evaluation. The results are as follows:

- **Matrix-Based Collaborative Filtering**: RMSE = 0.9135
- **Item-Based Collaborative Filtering (Weighted Mean Model)**: RMSE = 1.0316
- **Mean Model**: RMSE = 1.0420
- **User-Based Collaborative Filtering**: RMSE = 1.0950

The Matrix-Based model outperformed all others, making it the preferred choice for implementation in our system.

## Repository Structure

RECOMMENDER-SYSTEM
│
├── data
│   ├── test
│   │   └── test.csv
│   ├── train
│   │   └── train.csv
│   └── article_info.csv
│
├── myenv
│   ├── bin
│   ├── include
│   └── lib
│
├── notebook
│   ├── recommender_system_matrix_colab_filter.ipynb
│   ├── recommender_system_item_based_collab_filter.ipynb
│   └── recommender_system_user_collab_filtering.ipynb
│
├── scripts
│   ├── recommender_system_matrix_colab_fil.py
│   ├── recommender_system_item_based_collab_filter.py
│   └── recommender_system_user_collab_filtering.py
│
├── .gitignore
├── analysis.txt
├── LICENSE
└── README.md


## Dependencies
pandas
numpy
scikit-surprise
scikit-learn

## Getting Started

To run the scripts in your local environment:

1. Clone the repository.
2. Ensure Python 3.8+ is installed.
3. Install dependencies: `pip install -r requirements.txt`
4. Execute the desired script: `python scripts/<script_name>.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.


