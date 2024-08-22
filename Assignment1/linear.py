#write a basic linear regression and ridge regression model to predict the 
import numpy as np
import pandas as pd
import os
import sys

# Load data for Linear regression
def load_LR_data(train_file, weights_file):
    # Load training data
    train_data = pd.read_csv(train_file)
    X = train_data.iloc[:, :-1].values  # All columns except the last, this is the input matrix
    y = train_data.iloc[:, -1].values  # Last column, this is the output vector

    # Load weights
    weights = np.loadtxt(weights_file)
    return X, y, weights

def load_RR_data(train_file):
    train_data = pd.read_csv(train_file)
    X = train_data.iloc[:, :-1].values  # All columns except the last, this is the input matrix
    y = train_data.iloc[:, -1].values  # Last column, this is the output vector

    return X, y

# Preprocess data
def preprocess_data(X):
    # Adding a row of 1's for intercept term
    return np.c_[np.ones((X.shape[0], 1)), X]

# Implement weighted linear regression
def weighted_linear_regression(X, y, weights):
    # Compute (X^T * U * X)^-1 * X^T * U * y
    # where U is diagonal matrix with weights
    XTU = X.T * weights
    XTUX = XTU.dot(X)
    XTUy = XTU.dot(y)
    
    return np.linalg.solve(XTUX, XTUy)

# Implementation of ridge regression
def ridge_regression(X, y, lambda_val):
    # Compute (X^T * X + lambda * I)^-1 * X^T * y
    XT = X.T
    XTX = XT.dot(X) + lambda_val * np.identity(X.shape[1])
    XTy = XT.dot(y)
    
    return np.linalg.solve(XTX, XTy)

# Make predictions
def predict(X, w):
    return X.dot(w)

# Cross validation
def cross_validate(X, y, lambda_val):
    #10-fold cross validation
    n = X.shape[0]      
    fold_size = n//10       #size of each fold
    errors = np.zeros(10)   
    for i in range(10):
        X_train = np.concatenate((X[:i*fold_size],X[(i+1)*fold_size:]))         #training data
        y_train = np.concatenate((y[:i*fold_size],y[(i+1)*fold_size:]))        #training labels
        X_test = X[i*fold_size:(i+1)*fold_size]     #testing data
        y_test = y[i*fold_size:(i+1)*fold_size]     #testing labels
        w = ridge_regression(X_train, y_train, lambda_val)
        y_pred = predict(X_test, w)
        errors[i] = np.mean((y_pred - y_test)**2)    #mean square error
    return np.sum(errors)      #returning the sum of the errors

# Main function
def main():
    # File paths
    part = sys.argv[1]

    #Linear Regression
    if(part == 'a'):
        #System arguments for test files
        train_file = sys.argv[2]
        weights_file = sys.argv[4]
        test_file = sys.argv[3]         
        pred_file = sys.argv[5]
        wfile=sys.argv[6]

        # Load and preprocess data
        X_train, y_train, weights = load_LR_data(train_file, weights_file)
        X_train = preprocess_data(X_train)

        # Train model
        w = weighted_linear_regression(X_train, y_train, weights)

        # Load and preprocess test data
        test_data = pd.read_csv(test_file)
        X_test = preprocess_data(test_data.values)

        # Make predictions
        y_pred = predict(X_test, w)
        np.savetxt(pred_file, y_pred)
        np.savetxt(wfile, w)

    #Ridge Regression
    else:
        #Taking the system arguments
        train_file = sys.argv[2]
        lambda_file = sys.argv[4]
        test_file = sys.argv[3]       
        pred_file = sys.argv[5]
        wfile=sys.argv[6]
        best_lambda = sys.argv[7]

        #Loading and preprocessing the data
        lambda_val = np.loadtxt(lambda_file)    
        min_cross_val = np.inf
        lambda_fit = np.zeros(1)
        cross_validation = np.zeros(lambda_val.shape[0])
        X_train,y_train = load_RR_data(train_file)
        X_train = preprocess_data(X_train)       

        #cross validation of data for each lambda value
        for i in range(lambda_val.shape[0]):        
            cross_validation[i]=cross_validate(X_train, y_train, lambda_val[i]) 
            if(cross_validation[i]<min_cross_val):
                min_cross_val = cross_validation[i]     #finding the minimum cross validation error
                lambda_fit[0] = lambda_val[i]          #fitting the best lambda value

        #Training the model with the best lambda value
        w = ridge_regression(X_train, y_train, lambda_fit)
        X_test= pd.read_csv(test_file)
        X_test = preprocess_data(X_test.values)     #Testing Data preprocessing
        y_pred = predict(X_test, w)
        
        #saving the predictions, weights and best lambda
        np.savetxt(pred_file, y_pred)
        np.savetxt(wfile, w)
        np.savetxt(best_lambda, lambda_fit)

if __name__ == "__main__":
    main()
