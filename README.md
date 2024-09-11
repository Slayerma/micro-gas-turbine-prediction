# micro-gas-turbine-prediction

## Project Overview

This project focuses on building a machine learning pipeline to predict electrical power consumption based on various input features. The pipeline includes data loading, exploration, visualization, preprocessing, and model training. Several regression techniques such as **Linear Regression**, **Polynomial Regression**, **Decision Tree**, **Random Forest**, and **Gradient Boosting** are used, and models are fine-tuned using **GridSearchCV**. The project also includes **feature importance analysis** using **SHAP** values.

## Prerequisites

Before running the project, ensure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap statsmodels
```

## Files

- **train/ex_1.csv**: The dataset used for training the machine learning models.
  
## How to Run

1. Clone the repository and navigate to the directory.
2. Install the required dependencies as specified above.
3. Run the `main()` function to execute the entire pipeline:

```bash
python <micro-gas-turbine-prediction>.py
```

## Key Modules and Functions

### 1. **Data Loading and Exploration**

```python
def load_and_explore_data(filepath)
```

- Loads the dataset from a CSV file and prints the first few rows, statistical summary, and checks for missing values.

### 2. **Data Visualization**

```python
def visualize_data(dataset)
```

- Visualizes the distribution of electrical power, scatterplots, and a correlation matrix.

### 3. **Outlier Detection**

```python
def detect_outliers(dataset)
```

- Detects outliers in the `el_power` feature using the **IQR (Interquartile Range)** method.

### 4. **Data Preprocessing**

```python
def preprocess_data(dataset)
```

- Standardizes the features using **StandardScaler**.

### 5. **Data Splitting**

```python
def split_data(scaled_dataset)
```

- Splits the dataset into training and testing sets.

### 6. **Model Training and Evaluation**

```python
def train_and_evaluate_model(X_train, X_test, y_train, y_test)
```

- Trains a **Linear Regression** model and evaluates its performance using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared** score. 
- Also includes residual and actual vs. predicted plots.

### 7. **Cross-Validation**

```python
def cross_validation(X, y, model)
```

- Performs **5-fold cross-validation** and reports the average R-squared score.

### 8. **Multicollinearity Check**

```python
def check_multicollinearity(X_train)
```

- Computes the **Variance Inflation Factor (VIF)** to check for multicollinearity among the features.

### 9. **Polynomial Regression**

```python
def polynomial_regression(X_train, X_test, y_train)
```

- Trains a **Polynomial Regression** model with a degree of 2.

### 10. **Decision Tree Regression**

```python
def decision_tree_regression(X_train, X_test, y_train)
```

- Trains a **Decision Tree Regressor**.

### 11. **Model Fine-Tuning with Grid Search**

```python
def grid_search(X_train, y_train)
```

- Performs **GridSearchCV** on **Random Forest** and **Gradient Boosting** models to find the best hyperparameters.

### 12. **Feature Importance Analysis**

```python
def feature_importance_analysis(final_model, X)
```

- Uses **RandomForestRegressor** feature importance or **SHAP values** to analyze the importance of features and visualizes the results with a bar plot.

## Example Workflow

The project follows these steps:

1. Load the dataset and perform exploratory data analysis.
2. Visualize key relationships and the correlation matrix.
3. Detect outliers and preprocess the data.
4. Split the data into training and testing sets.
5. Train the initial model using **Linear Regression** and evaluate its performance.
6. Perform cross-validation and check for multicollinearity.
7. Implement **Polynomial Regression** and **Decision Tree Regression** models.
8. Use **GridSearchCV** to optimize **Random Forest** and **Gradient Boosting** models.
9. Perform feature importance analysis using **SHAP** values.

## Output

The script outputs several key metrics and visualizations:

- Model evaluation metrics such as **MAE**, **MSE**, and **R-squared**.
- Plots including **Actual vs. Predicted** values and **Residuals vs. Predicted** values.
- Feature importance plots.

## SHAP Analysis

The final model uses **SHAP** values to interpret the impact of each feature on the model's predictions. The SHAP summary plot provides a bar chart displaying feature importance.

## Conclusion

This project demonstrates a complete pipeline from data loading and preprocessing to model training, evaluation, and feature importance analysis for electrical power consumption prediction.


