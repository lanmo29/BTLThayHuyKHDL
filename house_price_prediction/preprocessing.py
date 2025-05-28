import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Preprocess data for model training"""
    # Xử lý missing values
    df = df.drop(['Id'], axis=1, errors='ignore')

    # Tách biến mục tiêu
    if 'SalePrice' in df.columns:
        y = df['SalePrice']
        X = df.drop(['SalePrice'], axis=1)
    else:
        y = None
        X = df.copy()

    # Phân loại các cột
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Loại bỏ các cột có quá nhiều giá trị missing
    cols_to_drop = []
    for col in X.columns:
        if X[col].isna().mean() > 0.5:
            cols_to_drop.append(col)

    numerical_cols = [col for col in numerical_cols if col not in cols_to_drop]
    categorical_cols = [col for col in categorical_cols if col not in cols_to_drop]

    # Tạo pipeline cho xử lý dữ liệu
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X, y, preprocessor, numerical_cols, categorical_cols


def get_important_features():
    """Return list of important features for the model"""
    return [
        'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
        'FullBath', 'YearBuilt', 'YearRemodAdd', '1stFlrSF',
        'GarageArea', 'TotRmsAbvGrd', 'Fireplaces', 'BsmtFinSF1',
        'LotArea', 'MasVnrArea', 'BsmtFullBath', 'WoodDeckSF',
        'OpenPorchSF', 'Neighborhood', 'MSZoning', 'Exterior1st',
        'Exterior2nd', 'KitchenQual', 'GarageType', 'SaleType'
    ]