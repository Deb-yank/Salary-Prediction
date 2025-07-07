

import pandas as pd
# Import the pandas library for data manipulation and analysis

import numpy as np
# Import NumPy for numerical operations, especially arrays and mathematical functions

import matplotlib.pyplot as plt
# Import Matplotlib's pyplot for plotting graphs and charts

import seaborn as sns
# Import seaborn for advanced data visualisation (built on top of matplotlib)

from sklearn.model_selection import train_test_split
# Import train_test_split to split the dataset into training and testing sets

from sklearn.linear_model import LinearRegression
# Import LinearRegression model for performing linear regression

from sklearn.metrics import mean_squared_error, r2_score
# Import metrics to evaluate the model: mean squared error and R² score

from sklearn.impute import SimpleImputer
# Import SimpleImputer to fill in missing values (e.g. using mean, median, or most frequent value)

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# Import encoders: OneHotEncoder for nominal categorical data, OrdinalEncoder for ordered categorical data



from sklearn.compose import ColumnTransformer

# Import ColumnTransformer to apply different preprocessing steps to specific columns

#load Dataset
df= pd.read_csv('pred.csv')

df.head()# reading the first five rows

df.isnull().sum()
# Check for missing (null) values in each column of the DataFrame

df.dropna(how='all',inplace=True)
# Drop rows where all the values are missing (completely empty rows)

df.duplicated().sum()
# Count the number of duplicate rows in the dataset

df.drop_duplicates(inplace=True)
# dropping duplicated rows

df.duplicated().sum()
# Count the number of duplicate rows in the dataset

df =df[(df['age']>=18)&(df['age']<=70)]
# removing outliers in ages



# remove outliers in year_expereince
df=df[(df['years_experience']>=0 )&(df['years_experience']<=40)]



# removing outliers in salary colums
df = df[(df['salary_usd'] >= 1000) & (df['salary_usd'] <= 300000)]

mode_imputer= SimpleImputer(strategy='most_frequent')
# Create an imputer that fills missing values with the most frequent value (mode) in each column

df['education_level'] = mode_imputer.fit_transform(df[['education_level']]).ravel()

# Fill missing values in the 'education_level' column using the most frequent value (mode)
# .fit_transform() learns the mode and replaces missing values
# .ravel() is used to flatten the 2D result into a 1D array to fit the DataFrame column

df.isnull().sum()
# Count the number of duplicate rows in the dataset



before = df.shape[0]
# Store the number of rows in the DataFrame before applying a filter (used for comparison)
before
 # Displays the number of rows before filtering

df=df[df['years_experience'] < df['age']]
# Remove rows where years of experience is not less than age (data inconsistency)

after = df.shape[0]
# Store the number of rows after filtering into the variable 'after'
after
# Display the number of rows after removing inconsistent entries (where experience >= age)



# Define which columns are ordinal and which are nominal
categorical_ordinal = ['education_level']
categorical_nominal = ['industry']
numeric_features = ['years_experience', 'age']

# Define the order for education levels
education_order = [['High School', 'BSc', 'MSc', 'PhD']]

# Create the ColumnTransformer for encoding
preprocessor = ColumnTransformer(transformers=[
    # OrdinalEncoder for education_level (ordered)
    ('edu', OrdinalEncoder(categories=education_order), categorical_ordinal),

    # OneHotEncoder for industry (unordered)
    ('ind', OneHotEncoder(drop='first'), categorical_nominal),

    # Pass through numeric features (no transformation needed)
    ('num', 'passthrough', numeric_features)
])



X = df[['education_level', 'industry', 'years_experience', 'age']]
# Select the input features (independent variables) for the model

y = df['salary_usd']

# Select the target variable (dependent variable) the model will predict

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Split the dataset into training and testing sets
# 80% of the data will be used for training, and 20% for testing
# random_state ensures the split is reproducible

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Create a pipeline that first preprocesses the data, then applies linear regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),     # Apply transformations like encoding and scaling
    ('regressor', LinearRegression())   # Fit a linear regression model
])

model.fit(X_train, y_train)

# Fit (train) the pipeline model on the training data
# This will first apply the preprocessing steps defined in the pipeline
# and then train the Linear Regression model on the transformed data







from sklearn.metrics import r2_score, mean_squared_error

y_pred = model.predict(X_test)
# Use the trained model to make predictions on the test set

# Calculate and print the R² Score (how well the model explains the variance in the target variable)
print("R² Score:", r2_score(y_test, y_pred))


# Calculate Mean Squared Error (average of squared differences between actual and predicted values)
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)
# Calculate Root Mean Squared Error (square root of MSE, gives error in the same unit as target variable)

print("RMSE:", rmse)

plt.scatter(y_test, y_pred)
# Create a scatter plot to compare actual vs predicted salary

plt.xlabel("Actual Salary")
# Label the x-axis as the actual salary values from the test set

plt.ylabel("Predicted Salary")
# Label the y-axis as the predicted salary values from the model

plt.title("Actual vs Predicted Salary")
# Add a title to the plot


plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# Plot a red dashed reference line (ideal predictions line where predicted = actual)

plt.show()
# display the plot

