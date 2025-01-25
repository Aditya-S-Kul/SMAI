import numpy as np

class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.classes = np.unique(self.y_true)
    
    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)
    
    def precision(self):
        precision_scores = {}
        for cls in self.classes:
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
            if true_positive + false_positive > 0:
                precision_scores[cls] = true_positive / (true_positive + false_positive)
            else:
                precision_scores[cls] = 0.0
        return precision_scores
    
    def recall(self):
        recall_scores = {}
        for cls in self.classes:
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))
            if true_positive + false_negative > 0:
                recall_scores[cls] = true_positive / (true_positive + false_negative)
            else:
                recall_scores[cls] = 0.0
        return recall_scores
    
    def f1_score(self):
        precision_scores = self.precision()
        recall_scores = self.recall()
        f1_scores = {}
        for cls in self.classes:
            if precision_scores[cls] + recall_scores[cls] > 0:
                f1_scores[cls] = 2 * (precision_scores[cls] * recall_scores[cls]) / (precision_scores[cls] + recall_scores[cls])
            else:
                f1_scores[cls] = 0.0
        return f1_scores

    def macro_precision(self):
        precision_scores = self.precision()
        return np.mean(list(precision_scores.values()))
    
    def macro_recall(self):
        recall_scores = self.recall()
        return np.mean(list(recall_scores.values()))
        
    def macro_f1(self):
        f1_scores = self.f1_score()
        return np.mean(list(f1_scores.values()))
    
    def micro_precision(self):
        true_positive = np.sum(self.y_true == self.y_pred)
        false_positive = np.sum((self.y_true != self.y_pred) & (self.y_pred != self.y_true))
        if true_positive + false_positive > 0:
            return true_positive / (true_positive + false_positive)
        else:
            return 0.0
    
    def micro_recall(self):
        true_positive = np.sum(self.y_true == self.y_pred)
        false_negative = np.sum((self.y_true != self.y_pred) & (self.y_true == self.y_true))
        if true_positive + false_negative > 0:
            return true_positive / (true_positive + false_negative)
        else:
            return 0.0

    def micro_f1(self):
        true_positive = np.sum(self.y_true == self.y_pred)
        false_positive = np.sum((self.y_pred != self.y_true) & (self.y_pred == self.y_pred))
        false_negative = np.sum((self.y_pred != self.y_true) & (self.y_true == self.y_true))
        if true_positive + false_positive + false_negative > 0:
            precision_micro = true_positive / (true_positive + false_positive)
            recall_micro = true_positive / (true_positive + false_negative)
            if precision_micro + recall_micro > 0:
                return 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
            else:
                return 0.0
        else:
            return 0.0



def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def calculate_std(y_true, y_pred):
    """Calculate Standard Deviation of the residuals."""
    return np.std(y_true - y_pred)

def calculate_variance(y_true, y_pred):
    """Calculate Variance of the residuals."""
    return np.var(y_true - y_pred)


def r_squared(y_actual, y_predicted):
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    
    # Calculate the total sum of squares (TSS)
    ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)
    
    # Calculate the residual sum of squares (RSS)
    ss_residual = np.sum((y_actual - y_predicted) ** 2)
    
    # Compute R-squared
    r2 = 1 - (ss_residual / ss_total)
    
    return r2

def mean_absolute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae