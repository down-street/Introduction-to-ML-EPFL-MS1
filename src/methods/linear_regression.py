import numpy as np
from ..utils import mse_fn
import matplotlib.pyplot as plt
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda =0):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.weight = None
        self.task_kind = 'ridge regression' if lmda != 0 else 'linear regression'


    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        self.training_data = training_data
        self.training_labels = training_labels
        N , D = training_data.shape
        I = np.eye(D)

        self.weight = np.linalg.pinv(training_data.T @ training_data + self.lmda * I) @ (training_data.T @ training_labels)

        pred_regression_targets = training_data @ self.weight
        return pred_regression_targets


    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        pred_regression_targets = test_data @ self.weight
        return pred_regression_targets

    def get_val_loss(self, xval, cval):
        pred_cval = self.predict(xval)
        return mse_fn(pred_cval, cval)
    
def KFold_cross_validation_linear_regression(X, Y, K, lmda):
    N = X.shape[0]
    
    losses = []  # list of accuracies
    for fold_ind in range(K):
        #Split the data into training and validation folds:
        #all the indices of the training dataset
        all_ind = np.arange(N)
        split_size = N // K
        
        val_ind = all_ind[fold_ind * split_size : (fold_ind + 1) * split_size]
        train_ind = np.setdiff1d(all_ind, val_ind, assume_unique=True)
        # find the set different with arr1 and arr2
        X_train_fold = X[train_ind, :]
        Y_train_fold = Y[train_ind]
        X_val_fold = X[val_ind, :]
        Y_val_fold = Y[val_ind]

        # YOUR CODE HERE
        model = LinearRegression(lmda=lmda)
        model.fit(X_train_fold, Y_train_fold)
        losses.append(model.get_val_loss(X_val_fold, Y_val_fold))
    #Find the average validation loss over K:
    ave_loss = np.mean(losses)
    return ave_loss

def run_search_for_hyperparam_linear(xtrain, ytrain, lmdas = [0, 0.1, 1, 5, 10, 15, 25]):
    print("Start Five Fold Cross-Validation for Linear Regression")
    results = []
    for lmda in lmdas:
        ave_loss = KFold_cross_validation_linear_regression(xtrain, ytrain, 5, lmda)
        results.append((lmda, ave_loss))
            
    results = np.array(results)
    
    x = results[:, 0]
    y = results[:, 1]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('Validation Result')
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.show()
    
    return results