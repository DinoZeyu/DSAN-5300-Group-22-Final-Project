# DSAN-5300-Group-22-Final-Project
This is the Data Science and Analytics 5300 Statistical Learning Final Project

## Introduction 

Airbnb has transformed travel by offering unique accommodations that provide a local experience, different from traditional hotels. In Thailand, with its rich culture and picturesque landscapes, Airbnb has seen a surge in popularity. This project focuses on examining Airbnb in Thailand, particularly its popularity and distribution.

We will analyze the location, pricing, ratings, and availability of Airbnb properties across Thailand to determine their level of popularity. By examining these factors, we aim to identify trends and patterns that explain why certain Airbnb listings are more favored by travelers. This analysis will help us understand the dynamics of Airbnb's market in Thailand and provide insights into the preferences of guests.



## Methodology

Logistic regression, Support Vector Machine (SVM),RandomForest Classifier , Decision Tree Classifier,Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA) and Neural Networks (ANN) to predict the popularity of Airbnb listings in Thailand based on the selected features. The models will be evaluated based on their accuracy, precision, recall, and F1 score. The best model will be selected based on these metrics.

## Usage 

* data_clean.py: This script cleans the raw data and prepares it for analysis. It removes missing values, duplicates, and irrelevant columns and assign the target variable based on the popularity of the Airbnb listings.It also encodes categorical variables and scales numerical variables.

* feature_selection.py: This script performs exploratory data analysis on the cleaned data. It generates visualizations to show the distribution of the features and their relationship with the target variable.

* ml_model.py: This script fits the models derived from methodlogy to the data and evaluates their performance using cross-validation. It also selects the best model based on the evaluation metrics.

* dl_model.py: This script fits an ann neural network model to the data and evaluates its performance using cross-validation. It also tunes the hyperparameters of the model to improve its performance.
