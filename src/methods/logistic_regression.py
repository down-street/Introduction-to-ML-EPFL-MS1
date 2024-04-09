import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


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

            if (i+1) % self.print_period == 0:
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


if __name__ == "__main__":
    feature_data = np.load('features.npz',allow_pickle=True)
    xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
            feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']
            
    model = LogisticRegression()
    print(model.fit(xtrain, ytrain))