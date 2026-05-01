#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    ## Loading Dataset
    data = pd.read_csv('customer.csv')

    ## Data Exploration
    print("Data head:")
    print(data.head())
    print("Data info:")
    print(data.info())
    print("Data describe:")
    print(data.describe())
    print("Null sums:")
    print(data.isnull().sum())

    #Data Preprocessing
    #Handling missing values
    data.ffill(inplace=True)

    ## drop irrelevant columns
    data.drop(['index'], axis=1, inplace=True)

    ## Split the data into features and target variable
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    ## Data preprocessing
    # Identify categorical and numerical columns
    categorical_cols = ['Age Group', 'Tariff Plan', 'Status']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    print("Categorical cols:", categorical_cols)
    print("Numerical cols:", numerical_cols)

    ## Encoding categorical variables and scaling numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    ## Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## Model Building using Logistic Regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    ## Model Training
    model.fit(X_train, y_train)

    ## Model Evaluation
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    #Hyperparameter Tuning using GridSearchCV
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'lbfgs']
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    print("Classification Report for Best Model:\n", classification_report(y_test, y_pred_best))
    print("Confusion Matrix for Best Model:\n", confusion_matrix(y_test, y_pred_best))
    print("Accuracy Score for Best Model:", accuracy_score(y_test, y_pred_best))

    # Save the best model
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print("Model saved as model.pkl")

except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()