import numpy as np
import matplotlib.pyplot as plt

from ..utils import label_to_onehot, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr=0.1, max_iters=10000, 
                 print_period=2000, record_period=1000):
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
        grad_W = X.T @ (Y_hat - Y) / m
        grad_b = np.sum(Y_hat - Y, axis=0, keepdims=True) / m
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

def KFold_cross_validation_logistic_regression(X, Y, K, lr=0.1, max_iters=10000):
    N = X.shape[0]
    
    accuracies = []  # list of accuracies
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
        model = LogisticRegression(lr=lr, print_period=5000, max_iters=max_iters)
        model.fit(X_train_fold, Y_train_fold)
        acc = accuracy_fn(model.predict(X_val_fold), Y_val_fold)
        accuracies.append(acc)
    
    #Find the average validation accuracy over K:
    ave_acc = np.mean(accuracies)
    return ave_acc

def run_search_for_hyperparam_logistic(xtrain, ytrain, learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]):
    print("Start Five Fold Cross-Validation for Logistic Regression")
    results = []
    for lr in learning_rates:
        print(f"lr={lr}")
        ave_acc = KFold_cross_validation_logistic_regression(xtrain, ytrain, 5, lr=lr)
        results.append((lr, ave_acc))
        print('\n')
            
    results = np.array(results)
    
    x = results[:, 0]
    y = results[:, 1]

    plt.plot(x, y, marker='o', linestyle='-')
    plt.title('Validation Result')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.show()
    
    return results

if __name__ == "__main__":
    feature_data = np.load('features.npz',allow_pickle=True)
    xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
            feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']
            
    model = LogisticRegression()
    print(model.fit(xtrain, ytrain))