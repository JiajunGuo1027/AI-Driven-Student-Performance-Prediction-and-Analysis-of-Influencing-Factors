import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
math_df = pd.read_csv(r'C:\Users\小皇帝陛下\Desktop\AI in education\Datasets\Math_Numeric.csv')
por_df = pd.read_csv(r'C:\Users\小皇帝陛下\Desktop\AI in education\Datasets\Por_Numeric.csv')

# Features to consider
features = ['G1', 'G2', 'failures', 'Medu', 'higher']
target = 'G3'

def prepare_data(df, features, target):
    X = df[features]
    y = df[target]
    return X, y

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

def cross_validate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(-mse_scores)
    mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
    return mse_scores, rmse_scores, mae_scores, r2_scores

def plot_predictions(y_test, y_pred, dataset_name):
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Random Forest Actual vs Predicted {dataset_name} Grades')

# Prepare data
X_math, y_math = prepare_data(math_df, features, target)
X_por, y_por = prepare_data(por_df, features, target)

# Split data
X_math_train, X_math_test, y_math_train, y_math_test = train_test_split(X_math, y_math, test_size=0.2, random_state=42)
X_por_train, X_por_test, y_por_train, y_por_test = train_test_split(X_por, y_por, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler().fit(pd.concat([X_math_train, X_por_train]))
X_math_train = scaler.transform(X_math_train)
X_math_test = scaler.transform(X_math_test)
X_por_train = scaler.transform(X_por_train)
X_por_test = scaler.transform(X_por_test)

# Initialize lists to store results for plotting
feature_importances_math = None
feature_importances_por = None

# Train and evaluate models
for dataset_name, X_train, X_test, y_train, y_test in [
    ('Math', X_math_train, X_math_test, y_math_train, y_math_test),
    ('Por', X_por_train, X_por_test, y_por_train, y_por_test)
]:
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse, rmse, mae, r2 = evaluate_model(y_test, y_pred)
    print(f"{dataset_name} Dataset - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}")
    
    # Cross-validation
    mse_cv, rmse_cv, mae_cv, r2_cv = cross_validate_model(model, X_train, y_train)
    print(f"{dataset_name} Dataset - Cross-Validation MSE: {mse_cv.mean()}, RMSE: {rmse_cv.mean()}, MAE: {mae_cv.mean()}, R²: {r2_cv.mean()}")

    # Store feature importances for plotting
    if dataset_name == 'Math':
        feature_importances_math = model.feature_importances_
    else:
        feature_importances_por = model.feature_importances_
    
    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 5))
    plot_predictions(y_test, y_pred, dataset_name)
    plt.show()

# Plot combined feature importances
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(features))

plt.bar(index, feature_importances_math, bar_width, label='Math')
plt.bar(index + bar_width, feature_importances_por, bar_width, label='Por')

plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importances for Math and Por Datasets')
plt.xticks(index + bar_width / 2, features)
plt.legend()
plt.tight_layout()
plt.show()