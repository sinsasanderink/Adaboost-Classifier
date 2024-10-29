# Employee Attrition Prediction Model

This project analyzes employee attrition in an HR dataset using machine learning. It follows a pipeline from data preparation to model fitting, aiming to predict whether an employee will leave ("Attrition: Yes") or stay ("Attrition: No") using AdaBoost Classifier.

### Table of Contents
1. **Data Loading and Exploration**
2. **Data Preprocessing and Feature Engineering**
3. **Modeling with AdaBoost**
4. **Evaluation**

---

### 1. Data Loading and Exploration
- Loads dataset using Pandas.
- Explores target variable `Attrition` to understand class distribution.
- Checks for missing values.

### 2. Data Preprocessing and Feature Engineering
- **Feature Engineering**: Separates numeric and categorical fields.
- **Encoding**: Uses `pd.get_dummies()` to encode categorical fields.
- **Scaling**: Scales numeric fields with `StandardScaler`.
- **Feature Merging**: Combines processed numeric and categorical fields.
- Maps `Attrition` target values to binary (Yes: 1, No: 0).

### 3. Train-Test Split
- Splits data into training and test sets (80-20 ratio) using `train_test_split` with stratification.

### 4. Model Fitting with AdaBoost Classifier
- Trains an AdaBoost model with 200 weak learners (using `DecisionTreeClassifier` as base).
- Key Parameters:
  - `n_estimators`: Number of weak learners.
  - `learning_rate`: Controls the contribution of each model iteration.

### 5. Evaluation
- **Accuracy**: Prints accuracy score.
- **Confusion Matrix**: Displays confusion matrix for performance insights.
