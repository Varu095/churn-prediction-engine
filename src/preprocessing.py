import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['AvgSpendPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

    target_col = 'Churn' if 'Churn' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true', 'y'] else 0)
    
    return X, y

def get_preprocessing_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor