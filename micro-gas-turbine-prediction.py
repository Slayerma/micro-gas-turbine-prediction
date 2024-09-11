import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_and_explore_data(filepath):
    dataset = pd.read_csv(filepath)
    print(dataset.head())
    print(dataset.describe())
    print(dataset.isnull().sum())
    return dataset

def visualize_data(dataset):
    plt.figure(figsize=(10, 6))
    plt.hist(dataset['el_power'], bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Electrical Power')
    plt.xlabel('Electrical Power')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='time', y='el_power', hue='input_voltage', data=dataset)
    plt.title('Time vs Electrical Power')
    plt.xlabel('Time')
    plt.ylabel('Electrical Power')
    plt.show()

    corr_matrix = dataset.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def detect_outliers(dataset):
    Q1 = dataset['el_power'].quantile(0.25)
    Q3 = dataset['el_power'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = dataset[(dataset['el_power'] < (Q1 - 1.5 * IQR)) | (dataset['el_power'] > (Q3 + 1.5 * IQR))]
    print(outliers)
    return outliers

def preprocess_data(dataset):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset[['time', 'input_voltage', 'el_power']])
    scaled_dataset = pd.DataFrame(scaled_data, columns=['time', 'input_voltage', 'el_power'])
    return scaled_dataset

def split_data(scaled_dataset):
    X = scaled_dataset[['time', 'input_voltage']]
    y = scaled_dataset['el_power']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Actual vs Predicted Electrical Power')
    plt.xlabel('Actual Electrical Power')
    plt.ylabel('Predicted Electrical Power')
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

def cross_validation(X, y, model):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'Cross-Validation R-squared Scores: {cv_scores}')
    print(f'Average Cross-Validation R-squared: {cv_scores.mean()}')

def check_multicollinearity(X_train):
    vif = pd.DataFrame()
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['features'] = X_train.columns
    print(vif)
    return vif

def polynomial_regression(X_train, X_test, y_train):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y_train)
    y_pred_poly = model_poly.predict(poly.fit_transform(X_test))
    return model_poly, y_pred_poly

def decision_tree_regression(X_train, X_test, y_train):
    model_dt = DecisionTreeRegressor()
    model_dt.fit(X_train, y_train)
    y_pred_dt = model_dt.predict(X_test)
    return model_dt, y_pred_dt

def grid_search(X_train, y_train):
    rf_model = RandomForestRegressor()
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
    rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2')
    rf_grid.fit(X_train, y_train)
    print(f'Best parameters for Random Forests: {rf_grid.best_params_}')
    print(f'Average Cross-Validation R-squared for Random Forests: {rf_grid.best_score_}')

    gb_model = GradientBoostingRegressor()
    gb_params = {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [50, 100, 200]}
    gb_grid = GridSearchCV(gb_model, gb_params, cv=5, scoring='r2')
    gb_grid.fit(X_train, y_train)
    print(f'Best parameters for Gradient Boosting: {gb_grid.best_params_}')
    print(f'Average Cross-Validation R-squared for Gradient Boosting: {gb_grid.best_score_}')

    return rf_grid, gb_grid

def feature_importance_analysis(final_model, X):
    if isinstance(final_model, RandomForestRegressor):
        feature_importances = final_model.feature_importances_
    else:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)
        feature_importances = np.mean(np.abs(shap_values), axis=0)

    feature_importances_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
    feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def main():
    filepath = 'micro+gas+turbine+electrical+energy+prediction/train/ex_1.csv'
    dataset = load_and_explore_data(filepath)
    visualize_data(dataset)
    detect_outliers(dataset)
    scaled_dataset = preprocess_data(dataset)
    X_train, X_test, y_train, y_test = split_data(scaled_dataset)
    train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    model = LinearRegression()
    cross_validation(X_train, y_train, model)
    check_multicollinearity(X_train)
    
    model_poly, y_pred_poly = polynomial_regression(X_train, X_test, y_train)
    model_dt, y_pred_dt = decision_tree_regression(X_train, X_test, y_train)
    
    rf_grid, gb_grid = grid_search(X_train, y_train)
    
    final_model = rf_grid.best_estimator_ if rf_grid.best_score_ > gb_grid.best_score_ else gb_grid.best_estimator_
    feature_importance_analysis(final_model, X_test)
    
    y_pred_rf = rf_grid.best_estimator_.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print(f"Mean Absolute Error (Random Forest): {mae_rf}")
    print(f"Mean Squared Error (Random Forest): {mse_rf}")
    print(f"R-squared (Random Forest): {r2_rf}")

    explainer = shap.TreeExplainer(rf_grid.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

if __name__ == "__main__":
    main()
