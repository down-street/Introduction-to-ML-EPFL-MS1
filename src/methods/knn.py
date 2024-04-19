import numpy as np
import matplotlib.pyplot as plt
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "center_locating"):
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
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        test_labels = []
        for test_point in test_data:
            # Compute distances between test_point and all training points
            distances = [np.linalg.norm(test_point - point) for point in self.training_data]
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            if self.task_kind == "breed_identifying":
                # For classification, vote the most frequent label
                label_count = {}
                for idx in k_indices:
                    label = self.training_labels[idx]
                    if isinstance(label, np.ndarray) and label.size == 1:
                        label = label[0]
                    elif isinstance(label, np.ndarray):
                        raise ValueError("Labels must be scalar or 1D array with one element for classification")
                    label_count[label] = label_count.get(label, 0) + 1
                most_frequent_label = max(label_count, key=label_count.get)
                test_labels.append(most_frequent_label)
            else:
                # For regression, calculate the mean of the neighbors' values
                neighbor_values = [self.training_labels[idx] for idx in k_indices]
                neighbor_values_array = np.array(neighbor_values)
                # Calculate the mean across the rows (axis=0), not across the columns
                average_value = np.mean(neighbor_values_array, axis=0)
                test_labels.append(average_value)

        return np.array(test_labels)
    
def KFold_cross_validation_knn(X, Y, K, k, task_kind):
    N = X.shape[0]
    losses = []  # list of accuracies
    acc = []
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
    #Find the average validation loss over K:
    if(task_kind=='center_locating'):
        ave_loss = np.mean(losses)
        return ave_loss
    if(task_kind=='breed_identifying'):
        ave_acc = np.mean(acc)
        return ave_acc
def run_search_for_hyperparam_knn(xtrain, ytrain,ctrain,K = [x for x in range(1,30)]):
    print("Start Five Fold Cross-Validation for knn")
    results_classification = []
    results_regression = []
    print('starting regression')
    for k in K:
        print('startingk=',k)
        ave_loss = KFold_cross_validation_knn(xtrain, ctrain, 5, k,'center_locating')
        results_regression.append((k, ave_loss))
    print('starting calssification')
    for k in K:
        print('startingk=',k)
        ave_acc = KFold_cross_validation_knn(xtrain, ytrain, 5, k,'breed_identifying')
        results_classification.append((k, ave_acc))        
    results_classification = np.array(results_classification)
    results_regression = np.array(results_regression)
    x = results_classification[:, 0]
    y = results_classification[:, 1]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('KNN Validation Result-breed_identifying')
    plt.xlabel('K')
    plt.ylabel('Acc')
    plt.show()
    x = results_regression[:, 0]
    y = results_regression[:, 1]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('KNN Validation Result-center_locating')
    plt.xlabel('K')
    plt.ylabel('Loss')
    plt.show()
    
    return [results_classification,results_regression]