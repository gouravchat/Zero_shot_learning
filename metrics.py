import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score,balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# overall F1 Score

def check_and_convert_tensor_to_numpy(vector):
    """ One_hot is batch size by number of class"""
    one_hot = None
    if tf.is_tensor(vector):
        one_hot = vector.numpy()
    else:
        one_hot = vector

    #print(one_hot)

    reverse_one_hot = np.argmax(one_hot, axis = -1) 
    #print(reverse_one_hot)

    return reverse_one_hot

def overallF1Score(y_true, y_pred):
    y_true = check_and_convert_tensor_to_numpy(y_true)
    y_pred = check_and_convert_tensor_to_numpy(y_pred)    
    return f1_score(y_true, y_pred, average='micro')

# overall MCC Score


def mccScore(y_true, y_pred):
    y_true = check_and_convert_tensor_to_numpy(y_true)
    y_pred = check_and_convert_tensor_to_numpy(y_pred)
    return matthews_corrcoef(y_true, y_pred)

# class Balanced F1-score


def classBalancedF1Score(y_true, y_pred):
    y_true = check_and_convert_tensor_to_numpy(y_true)
    y_pred = check_and_convert_tensor_to_numpy(y_pred)
    return f1_score(y_true, y_pred, average='weighted',zero_division=0)

# class Balanced Accuracy Score


def classBalancedAccuracyScore(y_true, y_pred, weights):
    y_true = check_and_convert_tensor_to_numpy(y_true)
    y_pred = check_and_convert_tensor_to_numpy(y_pred)
    return balanced_accuracy_score(y_true, y_pred)

# class Balanced F1 score

def classWiseF1Score(y_true, y_pred):
    y_true = check_and_convert_tensor_to_numpy(y_true)
    y_pred = check_and_convert_tensor_to_numpy(y_pred)
    return f1_score(y_true, y_pred, average='macro')

def FscorevsClass(y_true,y_pred):
    """ Fscore vs each class"""
    y_true = check_and_convert_tensor_to_numpy(y_true)
    y_pred = check_and_convert_tensor_to_numpy(y_pred)
    return precision_recall_fscore_support(y_true, y_pred)

def weightedPrecisonScore(y_true,y_pred):
  y_true = check_and_convert_tensor_to_numpy(y_true)
  y_pred = check_and_convert_tensor_to_numpy(y_pred)
  return precision_score(y_true, y_pred, average='weighted',zero_division=0)

def weightedRecallScore(y_true,y_pred):
  y_true = check_and_convert_tensor_to_numpy(y_true)
  y_pred = check_and_convert_tensor_to_numpy(y_pred)
  return recall_score(y_true, y_pred, average='weighted', zero_division=0)



if __name__ == "__main__":
    y_true  = tf.random.uniform(shape = [5],minval=0,maxval=5, dtype = tf.int32)
    print(y_true)
    y_true_one_hot = tf.one_hot(indices=y_true,depth = 6)
    print(y_true_one_hot)
    y_pred = tf.random.uniform(shape = (5,6),minval=0,maxval=1, dtype=tf.float32)
    print(y_pred)
    print("class balanced F1-score Metrics : ",classBalancedF1Score(y_true_one_hot,y_pred))