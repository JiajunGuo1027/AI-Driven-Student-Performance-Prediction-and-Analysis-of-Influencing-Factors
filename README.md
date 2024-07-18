# AI-Driven-Student-Performance-Prediction-and-Analysis-of-Influencing-Factors
This study aims to use machine learning models to predict students' academic performance and identify the key factors that influence their performance. The study utilizes the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance), focusing on two subjects: Mathematics (Math) and Portuguese (Por).

## Overview
This project aims to predict students' final grades (G3) based on various features such as their first period grade (G1), second period grade (G2), number of past class failures, mother's education level (Medu), and higher education aspirations (higher). The models developed include:
- Linear Regression
- Random Forest
- Ensemble Model (combining Linear Regression, Random Forest, and Gradient Boosting)

## Dataset
The dataset used in this study is the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance), which includes information on students' performance in Mathematics and Portuguese. The dataset files are located in the `Datasets` directory:
- `Math_Numeric.csv`
- `Por_Numeric.csv`

## Preprocessing
The preprocessing steps include:
1. Data Loading: Loading the datasets into DataFrames.
2. Feature Selection: Extracting relevant features and the target variable.
3. Data Splitting: Splitting the data into training (80%) and testing (20%) sets.
4. Standardization: Standardizing the features to have a mean of zero and a standard deviation of one.

## Models
The models developed and evaluated are:
- **Linear Regression**: Assumes a linear relationship between input features and the target variable.
- **Random Forest**: An ensemble learning method that constructs multiple decision trees during training and outputs the mean prediction of the individual trees.
- **Ensemble Model**: Combines predictions from Linear Regression, Random Forest, and Gradient Boosting Regressor to improve prediction accuracy.

The code for each model is located in the `code` directory:
- `LinearRegression.py`
- `RandomForest.py`
- `EnsembleModel.py`

## Evaluation
The models were evaluated using the following metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² (Coefficient of Determination)**

Additionally, K-Fold cross-validation (k=5) was performed to ensure the robustness and reliability of the models.

## Results
The results of the models, including prediction plots and feature importance analysis, are saved in the `images` directory. The images include:
- Actual vs Predicted Grades for each model and subject.
- Feature Importance for each model and subject.
