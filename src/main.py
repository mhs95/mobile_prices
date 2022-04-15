import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

def create_predictions(training_data, test_data):


    X_training = training_data.drop('price_range',axis=1)
    y_training = training_data['price_range']

    X_test = test_data.drop('id', axis=1)

    # Create objects for categorical features and numerical features
    numerical_cols = [col for col in X_training.columns if X_training[col].dtype in ['int64', 'float64']]
    log_cols = ['sc_w','clock_speed','fc']

    # Make pipeline for numerical variable transformations
    log_pipe = make_pipeline(PowerTransformer())
    standard_pipe = make_pipeline(StandardScaler())

    # Make full processor
    full = ColumnTransformer(
        transformers=[
            ("log", log_pipe, log_cols),
            ("standardize", standard_pipe, numerical_cols),
        ]
    )

    # Final pipeline
    final_pipeline = Pipeline(
        steps=[
            ("preprocess", full),
            ("fit", LogisticRegression(penalty = 'l2', C = 1000.0, max_iter=1000)),
        ]
    )
    
    print('Training model...')
    final_pipeline.fit(X_training, y_training)
    print('Making predictions for test set...')
    preds = final_pipeline.predict(X_test)

    test_data['predicted_price_category'] = preds
    test_data.to_csv('../data/output/predictions.csv', index=False)

    return print('Prediction file saved in output folder')

if __name__ == '__main__':

    training = pd.read_csv('../data/input/train.csv')
    test = pd.read_csv('../data/input/test.csv')

    create_predictions(training, test)