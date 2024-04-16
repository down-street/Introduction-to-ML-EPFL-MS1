import numpy as np
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

        pred_regression_targets = training_data.T @ self.weight
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

