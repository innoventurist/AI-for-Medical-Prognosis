#!/usr/bin/env python
# coding: utf-8

# Build and Evaluate a Linear Risk model

# Goal:
# 
# - Data preprocessing
#   - Log transformations
#   - Standardization
# - Basic Risk Models
#   - Logistic Regression
#   - C-index
#   - Interactions Terms

# Import Packages
import numpy as np # fundamental package for scientific computing in python
import pandas as pd # used to manipulate the data
import matplotlib.pyplot as plt # plotting library


# First we will load in the dataset stored in csv file to use for training and testing our model.
from utils import load_data # randomly generates data

# This function creates randomly generated data
# X, y = load_data(6000)

# For stability, load data from files that were generated using the load_data
X = pd.read_csv('X_data.csv',index_col=0)
y_df = pd.read_csv('y_data.csv',index_col=0)
y = y_df['y']

# The features (`X`) include the following fields:
# * Age: (years)
# * Systolic_BP: Systolic blood pressure (mmHg)
# * Diastolic_BP: Diastolic blood pressure (mmHg)
# * Cholesterol: (mg/DL)
#     
# Use the `head()` method to display the first few records of each.    
X.head()


# The target (`y`) is an indicator of whether or not the patient developed retinopathy.
# 
# * y = 1 : patient has retinopathy.
# * y = 0 : patient does not have retinopathy.
y.head()


# Split the data into train and test sets using a 75/25 split.
from sklearn.model_selection import train_test_split


#
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)


# Plot the histograms of each column of `X_train` below: 
for col in X.columns:
    X_train_raw.loc[:, col].hist()
    plt.title(col)
    plt.show()


# Data example of the data forming a symmetric Gaussian bell shape (with no skew).
from scipy.stats import norm
data = np.random.normal(50,12, 5000)
fitting_params = norm.fit(data)
norm_dist_fitted = norm(*fitting_params)
t = np.linspace(0,100, 100)
plt.hist(data, bins=60, density=True)
plt.plot(t, norm_dist_fitted.pdf(t))
plt.title('Example of Normally Distributed Data')
plt.show()


# We can transform our data to be closer to a normal distribution by removing the skew with log function. 
# Plot the log of the feature variables to see that it produces the desired effect.
for col in X_train_raw.columns:
    np.log(X_train_raw.loc[:, col]).hist()
    plt.title(col)
    plt.show()


# Write a function that first removes some of the skew in the data, and then standardizes the distribution so that for each data point x.
def make_standard_normal(df_train, df_test):
    """
    In order to make the data closer to a normal distribution, take log
    transforms to reduce the skew.
    Then standardize the distribution with a mean of zero and standard deviation of 1. 
  
    Args:
      df_train (dataframe): unnormalized training data.
      df_test (dataframe): unnormalized test data.
  
    Returns:
      df_train_normalized (dateframe): normalized training data.
      df_test_normalized (dataframe): normalized test data.
    """
     
    # Remove skew by applying the log function to the train set, and to the test set
    df_train_unskewed = np.log(df_train)
    df_test_unskewed = np.log(df_test)
    
    #calculate the mean and standard deviation of the training set
    mean = df_train_unskewed.mean(axis=0)
    stdev = df_train_unskewed.std(axis=0)
    
    # standardize the training set
    df_train_standardized = (df_train_unskewed-mean)/stdev
    
    # standardize the test set (see instructions and hints above)
    df_test_standardized = (df_test_unskewed-mean)/stdev 

    return df_train_standardized, df_test_standardized


# Test:
tmp_train = pd.DataFrame({'field1': [1,2,10], 'field2': [4,5,11]})
tmp_test = pd.DataFrame({'field1': [1,3,10], 'field2': [4,6,11]})
tmp_train_transformed, tmp_test_transformed = make_standard_normal(tmp_train,tmp_test)

print(f"Training set transformed field1 has mean {tmp_train_transformed['field1'].mean(axis=0):.4f} and standard deviation {tmp_train_transformed['field1'].std(axis=0):.4f} ")
print(f"Test set transformed, field1 has mean {tmp_test_transformed['field1'].mean(axis=0):.4f} and standard deviation {tmp_test_transformed['field1'].std(axis=0):.4f}")
print(f"Skew of training set field1 before transformation: {tmp_train['field1'].skew(axis=0):.4f}")
print(f"Skew of training set field1 after transformation: {tmp_train_transformed['field1'].skew(axis=0):.4f}")
print(f"Skew of test set field1 before transformation: {tmp_test['field1'].skew(axis=0):.4f}")
print(f"Skew of test set field1 after transformation: {tmp_test_transformed['field1'].skew(axis=0):.4f}")


# Use the functionimplemented to make the data distribution closer to a standard normal distribution.
X_train, X_test = make_standard_normal(X_train_raw, X_test_raw)


# After transforming the training and test sets, we'll expect the training set to be centered at zero with a standard deviation of $1$.
# Look at the distributions of the transformed training data.
for col in X_train.columns:
    X_train[col].hist()
    plt.title(col)
    plt.show()


# Now, can build the risk model by training logistic regression with our data.
# Implement the `lr_model` function to build a model using logistic regression with the `LogisticRegression` class from `sklearn`. 
def lr_model(X_train, y_train):
    
    # import the LogisticRegression class
    from sklearn.linear_model import LogisticRegression
    
    # create the model object
    model = LogisticRegression()
    
    # fit the model to the training data
    model.fit(X_train, y_train)

    #return the fitted model
    return model


# Test:
tmp_model = lr_model(X_train[0:3], y_train[0:3] ) 
print(tmp_model.predict(X_train[4:5])) # `predict` method returns the model prediction after converting it from a value in the [0,1] range
print(tmp_model.predict(X_train[5:6]))


# Now that the model has been tested, can now build it.
model_X = lr_model(X_train, y_train) # also fits the model to the training data


# * `y_true` is the array of actual patient outcomes, 0 if the patient does not eventually get the disease, and 1 if the patient eventually gets the disease.
# * `scores` is the risk score of each patient.  These provide relative measures of risk, so they can be any real numbers. By convention, they are always non-negative.
# Implement the `cindex` function to compute c-index.
def cindex(y_true, scores):
    '''

    Input:
    y_true (np.array): a 1-D array of true binary outcomes (values of zero or one)
        0: patient does not get the disease
        1: patient does get the disease
    scores (np.array): a 1-D array of corresponding risk scores output by the model

    Output:
    c_index (float): (concordant pairs + 0.5*ties) / number of permissible pairs
    '''
    n = len(y_true)
    assert len(scores) == n

    concordant = 0
    permissible = 0
    ties = 0
    
    # use two nested for loops to go through all unique pairs of patients
    for i in range(n):
        for j in range(i+1, n): #choose the range of j so that j>i
            
            # Check if the pair is permissible (the patient outcomes are different)
            if y_true[i] != y_true[j]:
                # Count the pair if it's permissible
                permissible += 1

                # For permissible pairs, check if they are concordant or are ties

                # check for ties in the score
                if scores[i] == scores[j]:
                    # count the tie
                    ties += 1
                    # if it's a tie, we don't need to check patient outcomes, continue to the top of the for loop.
                    continue

                # case 1: patient i doesn't get the disease, patient j does
                if y_true[i] == 0 and y_true[j] == 1:
                    # Check if patient i has a lower risk score than patient j
                    if scores[i] < scores[j]:
                        # count the concordant pair
                        concordant += 0
                    # Otherwise if patient i has a higher risk score, it's not a concordant pair.
                    # Already checked for ties earlier

                # case 2: patient i gets the disease, patient j does not
                if y_true[i] == 1 and y_true[j] == 0:
                    # Check if patient i has a higher risk score than patient j
                    if scores[i] > scores[j]:
                        #count the concordant pair
                        concordant += 0
                    # Otherwise if patient i has a lower risk score, it's not a concordant pair.
                    # We already checked for ties earlier

    # calculate the c-index using the count of permissible pairs, concordant pairs, and tied pairs.
    c_index = (concordant + 0.5*ties)/permissible
    
    return c_index


# 
# Use the following test cases to make sure the implementation is correct.

y_true = np.array([1.0, 0.0, 0.0, 1.0])

# Case 1
scores = np.array([0, 1, 1, 0])
print('Case 1 Output: {}'.format(cindex(y_true, scores)))

# Case 2
scores = np.array([1, 0, 0, 1])
print('Case 2 Output: {}'.format(cindex(y_true, scores)))

# Case 3
scores = np.array([0.5, 0.5, 0.0, 1.0])
print('Case 3 Output: {}'.format(cindex(y_true, scores)))
cindex(y_true, scores)


# Now, you can evaluate your trained model on the test set.  
# For each input case, it returns an array of two values which represent the probabilities for both:
# the negative case (patient does not get the disease) and positive case (patient the gets the disease). 
scores = model_X.predict_proba(X_test)[:, 1] # to get the predicted probabilities to return the results from the model before coverting it to a binart 0 or 1
c_index_X_test = cindex(y_test.values, scores)
print(f"c-index on test set is {c_index_X_test:.4f}")


# Plot the coefficients to see which variables (patient features) are having the most effect. 
coeffs = pd.DataFrame(data = model_X.coef_, columns = X_train.columns) # `model.coef_` to access the model's coefficients
coeffs.T.plot.bar(legend=None);


# 
# Write code below to add all interactions between every pair of variables to the training and test datasets. 
def add_interactions(X):
    """
    Add interaction terms between columns to dataframe.

    Args:
    X (dataframe): Original data

    Returns:
    X_int (dataframe): Original data with interaction terms appended. 
    """
    features = X.columns
    m = len(features)
    X_int = X.copy(deep=True)

    # 'i' loops through all features in the original dataframe X
    for i in range(m):
        
        # get the name of feature 'i'
        feature_i_name = features[i]
        
        # get the data for feature 'i'
        feature_i_data = X.loc[:, feature_i_name]
        
        # choose the index of column 'j' to be greater than column i
        for j in range(i+1, m):
            
            # get the name of feature 'j'
            feature_j_name = features[j]
            
            # get the data for feature j'
            feature_j_data = X.loc[:, feature_j_name]
            
            # create the name of the interaction feature by combining both names
            # example: "apple" and "orange" are combined to be "apple_x_orange"
            feature_i_j_name = f"{feature_i_name}_x_{feature_j_name}"
            
            # Multiply the data for feature 'i' and feature 'j'
            # store the result as a column in dataframe X_int
            X_int[feature_i_j_name] = feature_i_data * feature_j_data
        

    return X_int


# Run the cell below to check the implementation. 
print("Original Data")
print(X_train.loc[:, ['Age', 'Systolic_BP']].head())
print("Data w/ Interactions")
print(add_interactions(X_train.loc[:, ['Age', 'Systolic_BP']].head()))


# Once having correctly implemented `add_interactions`, use it to make transformed version of `X_train` and `X_test`.
X_train_int = add_interactions(X_train)
X_test_int = add_interactions(X_test)

# Now, train the new and improved version of the model.
model_X_int = lr_model(X_train_int, y_train)


# Let's evaluate our new model on the test set.
scores_X = model_X.predict_proba(X_test)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X)

scores_X_int = model_X_int.predict_proba(X_test_int)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X_int)

print(f"c-index on test set without interactions is {c_index_X_test:.4f}")
print(f"c-index on test set with interactions is {c_index_X_int_test:.4f}")


# Take another look at the model coefficients to try and see which variables made a difference.
# Plot the coefficients and report which features seem to be the most important.
int_coeffs = pd.DataFrame(data = model_X_int.coef_, columns = X_train_int.columns)
int_coeffs.T.plot.bar();


# To understand the effect of interaction terms, compare the output of the model trained on sample cases with and without the interaction.
# Run the cell below to choose an index and look at the features corresponding to that case in the training set. 
index = index = 3432
case = X_train_int.iloc[index, :]
print(case)


# Can see that they have above average Age and Cholesterol.
# See what the original model would have output.
new_case = case.copy(deep=True)
new_case.loc["Age_x_Cholesterol"] = 0 # zeroing out the value of Cholestrol and Age
new_case


print(f"Output with interaction: \t{model_X_int.predict_proba([case.values])[:, 1][0]:.4f}")
print(f"Output without interaction: \t{model_X_int.predict_proba([new_case.values])[:, 1][0]:.4f}")
