# mobile_prices

## Overview
This repository contains code to solve Kaggle's [Mobile Price Classification Problem](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification). 

## Structure
1. src
  -  `main.py`: This script runs the final hyperparameter tuned model on the full training set, generates predictions for the test set, and saves
              the test set with predictions in the output folder. 

2. data
  - input: This folder contains the training (`'train.csv'`) and test (`'test.csv'`) sets provided by Kaggle.
  - output: This folder contains the final predictions generated for the test set (`'preditions.csv'`).

3. playground
  - `explore_data.ipynb`: Jupyter Notebook for EDA, feature engineering, model-fitting, cross-validation and hyperparameter tuning on the 
                        training data. 
