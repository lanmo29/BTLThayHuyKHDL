import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


def train_model(X, y, preprocessor, model_type='random_forest'):
    """Train model with the given preprocessor and data"""
    # Tiền xử lý dữ liệu
    X_processed = preprocessor.fit_transform(X)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Chọn và huấn luyện mô hình
    if model_type == 'linear_regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
    else:  # random_forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)

        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_

    # Đánh giá mô hình
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)

    results = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'y_test': y_test,
        'test_predictions': test_predictions
    }

    return results


def save_model(model, preprocessor, file_path):
    """Save model and preprocessor to file"""
    joblib.dump({'model': model, 'preprocessor': preprocessor}, file_path)


def load_model(file_path):
    """Load model and preprocessor from file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tệp mô hình không tồn tại tại: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Tệp mô hình tại {file_path} rỗng")
    saved_data = joblib.load(file_path)
    return saved_data['model'], saved_data['preprocessor']


def predict_price(model, preprocessor, features, numerical_cols, categorical_cols):
    """Predict house price given features"""
    # Chuyển dictionary thành DataFrame
    features_df = pd.DataFrame([features])

    # Lấy danh sách các cột gốc mà preprocessor mong đợi
    expected_cols = numerical_cols + categorical_cols

    # Điền các cột thiếu bằng giá trị mặc định
    for col in expected_cols:
        if col not in features_df.columns:
            # Giả định giá trị mặc định (có thể điều chỉnh dựa trên dữ liệu huấn luyện)
            if col in numerical_cols:
                features_df[col] = 0  # Giá trị mặc định cho numerical (có thể dùng mean/median)
            elif col in categorical_cols:
                features_df[col] = 'None'  # Giá trị mặc định cho categorical

    # Sắp xếp lại cột theo thứ tự mong đợi
    features_df = features_df[expected_cols]

    # Thực hiện transform
    features_processed = preprocessor.transform(features_df)
    prediction = model.predict(features_processed)[0]
    return prediction


def plot_feature_importance(model, feature_names):
    """Plot feature importance for Random Forest model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        return plt
    return None