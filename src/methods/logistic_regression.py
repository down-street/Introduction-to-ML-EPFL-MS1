import numpy as np
import matplotlib.pyplot as plt

from ..utils import label_to_onehot, accuracy_fn, macrof1_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr=0.1, max_iters=10000, 
                 print_period=2000, record_period=1000,task_kind ="classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.print_period = print_period
        self.record_period = record_period
        self.task_kind = task_kind
    
        
        self.weights = None
        self.bias = None
        self.loss_rec = []

    def initialize_parameters(self, D, C):
        """
        Initialize the weights and bias

        Args:
            D (int): dimension of features
            C (int): number of classes

        Returns:
            W (array): (D, C)
            b (array): (1, C)
        """
        W = np.random.normal(0, 0.01, (D, C))
        b = np.zeros((1, C), dtype=np.float32)
        return W, b

    def f_softmax(self, a):
        """
        Softmax Function

        Args:
            a (array): (1, C)

        Returns:
            (array): (1, C)
        """
        exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
        return exp_a / np.sum(exp_a, axis=1, keepdims=True)
    
    def compute_loss(self, Y_gt, Y_hat):
        """
        Compute the loss

        Args:
            Y_gt (array): the true labels       (N, C)
            Y_hat (array): the predicted labels (N, C)

        Returns:
            loss (float): loss value
        """
        m = Y_gt.shape[0]
        loss = -np.sum(Y_gt * np.log(Y_hat)) / m
        return loss
    
    def compute_grad(self, X, Y, Y_hat):
        """
        Compute the gradient of loss function

        Args:
            X (array): (N, D)
            Y (array): (N, C)
            Y_hat (array): (N, C)

        Returns:
            grad_W (array): (D, C)
            grad_b (array): (1, C)
        """
        m = X.shape[0]
        grad_W = X.T @ (Y_hat - Y) 
        grad_b = np.sum(Y_hat - Y, axis=0, keepdims=True) 
        return grad_W, grad_b
    
    def get_predict_labels(self, X, W, b):
        """
        Get the predicted labels(one-hot code form)

        Args:
            X (array): (N, D)
            W (array): (D, C)
            b (array): (1, C)

        Returns:
            pred_labels (array): (N, C)
        """
        a = np.dot(X, W) + b
        pred_labels = self.f_softmax(a)
        return pred_labels

    def get_predict_classes(self, pred_labels):
        """
        Get the predicted classes

        Args:
            pred_labels (array): (N, C)

        Returns:
            (array) : (N, 1)
        """
        return np.argmax(pred_labels, axis = 1)


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        loss_rec = []
        loss = 0
        train_feat = training_data
        if training_labels.ndim == 1:
            train_label = label_to_onehot(training_labels)
        elif sum(training_labels) == train_feat.shape[1]:
            train_label = training_labels
        
        D = train_feat.shape[1]
        C = train_label.shape[1]
        W, b = self.initialize_parameters(D, C)

        for i in range(self.max_iters):
            
            pred_labels = self.get_predict_labels(train_feat, W, b)
            
            dW, db = self.compute_grad(train_feat, train_label, pred_labels)
            
            W = W - self.lr * dW
            b = b - self.lr * db

            if self.print_period != 0 and (i+1) % self.print_period == 0:
                loss = self.compute_loss(train_label, pred_labels)
                print(f"Iteration {i+1}, loss: {loss}")
            
            if (i+1) % self.record_period == 0:
                loss_rec.append(loss)

        self.weights = W
        self.bias = b
        self.loss_rec = loss_rec
        
        return self.predict(train_feat)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        return self.get_predict_classes(self.get_predict_labels(test_data, self.weights, self.bias))

def KFold_cross_validation_logistic_regression(X, Y, K, lr=0.001, max_iters=500):
    N = X.shape[0]
    
    accuracies = []  # list of accuracies
    f1s = []
    for fold_ind in range(K):
        #Split the data into training and validation folds:
        print(f"fold {fold_ind+1}")
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
        model = LogisticRegression(lr=lr, max_iters=max_iters, print_period=0)
        model.fit(X_train_fold, Y_train_fold)
        pred_Y_val_fold = model.predict(X_val_fold)
        acc = accuracy_fn(pred_Y_val_fold, Y_val_fold)
        f1 = macrof1_fn(pred_Y_val_fold, Y_val_fold)
        accuracies.append(acc)
        f1s.append(f1)
        
    #Find the average validation accuracy over K:
    ave_acc = np.mean(accuracies)
    f1_acc = np.mean(f1)
    return ave_acc, f1_acc

def run_grid_search_for_hyperparam_logistic(xtrain, ytrain, 
                                            learning_rates = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2], 
                                            max_iters = [10, 50, 100, 200, 500, 1000]):
    results_lr = []
    print("---Run five-fold validation for learning rate---")
    for lr in learning_rates:
        print(f"lr={lr}")
        ave_acc, f1_acc = KFold_cross_validation_logistic_regression(xtrain, ytrain, 5, lr=lr, max_iters=100)
        results_lr.append((lr, ave_acc, f1_acc))
            
    results_lr = np.array(results_lr)
    
    print("---Run five-fold validation for iteration number---")
    results_iters = []
    for iters in max_iters:
        print(f"iters={iters}")
        ave_acc, f1_acc = KFold_cross_validation_logistic_regression(xtrain, ytrain, 5, lr=0.01, max_iters=iters)
        results_iters.append((iters, ave_acc, f1_acc))
        
    results_iters = np.array(results_iters)
    
    return results_lr, results_iters

def draw_val_result(results_lr, results_iters, iters_when_val_lr=100, lr_when_val_iters=0.01):
    lr = results_lr[:, 0]
    acc = results_lr[:, 1]
    f1 = results_lr[:, 2]

    fig, (ax1_lr, ax1_iter) = plt.subplots(1, 2, figsize=(12, 4))

    ax1_lr.set_xlabel('Learning Rate')
    ax1_lr.set_ylabel('Accuracy')
    line1 = ax1_lr.plot(lr, acc, marker='o', linestyle='-', color = 'tab:orange')
    ax1_lr.tick_params(axis='y')

    ax2_lr = ax1_lr.twinx()
    ax2_lr.set_ylabel('F1 Score')
    line2 = ax2_lr.plot(lr, f1, marker='o', linestyle='-')
    ax2_lr.tick_params(axis='y')

    lines = line1 + line2
    labels = ['Accuracy', 'F1 Score']
    ax1_lr.legend(lines, labels)

    plt.title(f'Validation Result(Max Iters={iters_when_val_lr})')

    iters = results_iters[:, 0]
    acc = results_iters[:, 1]
    f1 = results_iters[:, 2]

    ax1_iter.set_xlabel('Max Iteration')
    ax1_iter.set_ylabel('Accuracy')
    line1 = ax1_iter.plot(iters, acc, marker='o', linestyle='-', color = 'tab:orange')
    ax1_iter.tick_params(axis='y')

    ax2_iter = ax1_iter.twinx()
    ax2_iter.set_ylabel('F1 Score')
    line2 = ax2_iter.plot(iters, f1, marker='o', linestyle='-')
    ax2_iter.tick_params(axis='y')

    lines = line1 + line2
    labels = ['Accuracy', 'F1 Score']
    ax1_iter.legend(lines, labels, loc="lower right")

    plt.title(f'Validation Result(Learning Rate={lr_when_val_iters})')
    plt.subplots_adjust(wspace=0.5)
    plt.show()
