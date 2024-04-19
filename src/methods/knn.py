import numpy as np
import matplotlib.pyplot as plt
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import time

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "regression"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        self.training_data = training_data
        self.training_labels = training_labels
        return training_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data using distance-weighted k-NN.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        test_labels = []
        for test_point in test_data:
            # Compute distances between test_point and all training points
            distances = [np.linalg.norm(test_point - point) for point in self.training_data]
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Inverse distances for weighting
            
            # Calculate weights, handling zero distances
            weights = np.array(distances)[k_indices]
            # Set weights: if distance is 0, use a large number; otherwise, take the reciprocal
            weights = np.where(weights == 0, 1e9, 1 / weights)
            #if weights.sum() == 0:
            #    weights = np.ones_like(weights)  # Handle the rare case where all distances are zero

            if self.task_kind == "breed_identifying":
                # For classification, weigh the vote by the inverse of the distance
                label_count = {}
                for idx, weight in zip(k_indices, weights):
                    label = self.training_labels[idx]
                    label_count[label] = label_count.get(label, 0) + weight
                most_frequent_label = max(label_count, key=label_count.get)
                test_labels.append(most_frequent_label)
            else:
                # For regression, calculate the weighted average of the neighbors' values
                neighbor_values = [self.training_labels[idx] for idx in k_indices]
                neighbor_values_array = np.array(neighbor_values)
            
                # Ensure the weights array is correctly shaped to match neighbor_values_array
                weights = weights.reshape(-1)  # This line makes sure weights are a 1-D array matching the number of neighbors
            
                # Calculate the weighted mean
                if neighbor_values_array.ndim > 1:
                    # If neighbor_values_array is multi-dimensional, specify axis along which to average
                    average_value = np.average(neighbor_values_array, weights=weights, axis=0)
                else:
                    # For a 1-D array, no need to specify the axis
                    average_value = np.average(neighbor_values_array, weights=weights)   
                test_labels.append(average_value)
        return np.array(test_labels)
    
def KFold_cross_validation_knn(X, Y, K, k, task_kind):
    N = X.shape[0]
    losses = []  # list of accuracies
    acc = []
    F1 = []
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
        model = KNN(k,task_kind=task_kind)
        model.fit(X_train_fold, Y_train_fold)
        preds = model.predict(X_val_fold)
        if(task_kind=='center_locating'):
            losses.append(mse_fn(preds, Y_val_fold))
        if(task_kind=='breed_identifying'):
            acc.append(accuracy_fn(preds, Y_val_fold))
            F1.append(macrof1_fn(preds, Y_val_fold))
    #Find the average validation loss over K:
    if(task_kind=='center_locating'):
        ave_loss = np.mean(losses)
        return ave_loss
    if(task_kind=='breed_identifying'):
        ave_acc = np.mean(acc)
        ave_f1=np.mean(F1)
        return ave_acc,ave_f1
def run_search_for_hyperparam_knn(xtrain, ytrain,ctrain,K = [x for x in range(1,20)]):
    print("Start Five Fold Cross-Validation for knn")
    results_classification = []
    results_regression = []
    results_classification_f1 = []
    print('starting regression')
    for k in K:
        print('startingk=',k)
        ave_loss = KFold_cross_validation_knn(xtrain, ctrain, 5, k,'center_locating')
        results_regression.append((k, ave_loss))
    
    print('starting calssification')
    for k in K:
        print('startingk=',k)
        ave_acc,ave_f1 = KFold_cross_validation_knn(xtrain, ytrain, 5, k,'breed_identifying')
        results_classification.append((k, ave_acc))        
        results_classification_f1.append((k, ave_f1))   
    results_classification = np.array(results_classification)
    results_regression = np.array(results_regression)
    results_classification_f1 = np.array(results_classification_f1)
    '''
    x = results_classification[:, 0]
    y = results_classification[:, 1]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('KNN Validation Result-breed_identifying')
    plt.xlabel('K')
    plt.ylabel('Acc')
    plt.show()
    '''
    x = results_classification[:, 0]  # K values
    y_acc = results_classification[:, 1]  # Loss values
    y_f1 = results_classification_f1[:, 1]  # F1 scores

    # Create a plot with two y-axes
    fig, ax1 = plt.subplots()

    # Plotting the loss values
    color = 'tab:red'
    ax1.set_xlabel('K')
    ax1.set_ylabel('Acc', color=color)
    ax1.plot(x, y_acc, marker='o', linestyle='-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the F1 scores
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('F1 Score', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y_f1, marker='s', linestyle='-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and show the plot
    plt.title('KNN Validation Result-Breed_Identifying')
    plt.show()  
    x = results_regression[:, 0]
    y = results_regression[:, 1]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('KNN Validation Result-Center_Locating')
    plt.xlabel('K')
    plt.ylabel('Loss')
    plt.show()
    
    return [results_classification,results_regression]