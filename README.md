# Salary Prediction Model

A machine learning project that predicts salaries based on education level, industry, years of experience, and age using linear regression.

## Overview

This project implements a salary prediction model using scikit-learn's Linear Regression algorithm. The model processes both categorical and numerical features to predict salary in USD based on employee characteristics.

## Dataset

The model expects a CSV file named `pred.csv` with the following columns:
- `age`: Employee age (filtered to 18-70 years)
- `years_experience`: Years of work experience (filtered to 0-40 years)
- `education_level`: Education level (High School, BSc, MSc, PhD)
- `industry`: Industry sector (categorical)
- `salary_usd`: Target variable - salary in USD (filtered to $1,000-$300,000)

## Features

### Data Preprocessing
- **Missing Value Handling**: Uses mode imputation for categorical variables
- **Outlier Removal**: Filters unrealistic values for age, experience, and salary
- **Data Validation**: Removes inconsistent records where experience exceeds age
- **Duplicate Removal**: Eliminates duplicate rows

### Feature Engineering
- **Ordinal Encoding**: Applied to education levels (High School < BSc < MSc < PhD)
- **One-Hot Encoding**: Applied to industry categories
- **Numerical Features**: Age and years of experience passed through unchanged

### Model Pipeline
- Uses scikit-learn Pipeline for streamlined preprocessing and modeling
- Combines ColumnTransformer for feature preprocessing with LinearRegression

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

1. Clone or download the project files
2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Ensure your dataset file `pred.csv` is in the same directory as the script

## Usage

```python
python salary.py
```

The script will:
1. Load and preprocess the dataset
2. Split data into training (80%) and testing (20%) sets
3. Train a linear regression model
4. Evaluate model performance
5. Display results and visualization

## Model Performance

The model outputs:
- **R² Score**: Measures how well the model explains variance in salary
- **RMSE**: Root Mean Squared Error in salary prediction
- **Scatter Plot**: Visual comparison of actual vs predicted salaries

## Data Quality Controls

- Age range: 18-70 years
- Experience range: 0-40 years
- Salary range: $1,000-$300,000
- Logical consistency: Experience must be less than age
- Missing value imputation using most frequent values

## File Structure

```

├── salary.py          # Main script
├── pred.csv        # Dataset (required)
└── README.md       # This file
```

## Model Architecture

1. **Data Loading**: Reads CSV file using pandas
2. **Data Cleaning**: Removes nulls, duplicates, and outliers
3. **Feature Preprocessing**: 
   - Ordinal encoding for education levels
   - One-hot encoding for industries
   - Passthrough for numerical features
4. **Model Training**: Linear regression with 80/20 train-test split
5. **Evaluation**: R² score, RMSE, and visualization

## Visualization

The script generates a scatter plot showing:
- X-axis: Actual salary values
- Y-axis: Predicted salary values
- Red dashed line: Perfect prediction line (y = x)
- Data points: Individual predictions

## Customization

To modify the model:
- Change education level ordering in `education_order`
- Adjust outlier thresholds for age, experience, or salary
- Modify train-test split ratio in `train_test_split`
- Replace LinearRegression with other algorithms

## Performance Interpretation

- **R² Score**: Higher values (closer to 1.0) indicate better model fit
- **RMSE**: Lower values indicate more accurate predictions
- **Scatter Plot**: Points closer to the red line indicate better predictions

## Notes

- The model assumes a linear relationship between features and salary
- Categorical variables are properly encoded to work with linear regression
- The pipeline ensures consistent preprocessing for both training and prediction
- Random state is set for reproducible results

## Future Improvements

- Feature scaling for numerical variables
- Cross-validation for more robust evaluation
- Feature importance analysis
- Advanced algorithms (Random Forest, XGBoost)
- Hyperparameter tuning
