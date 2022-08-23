"""Various functions to compute metrics."""

def accuracy_full(y_true, y_pred):
    """Accuracy metric using tp, tn, fp, fn.

    Parameters
    ----------
    y_true : array-like of shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array-like of shape = [n_samples]
        Estimated target values.

    Returns
    -------
    accuracy : float
        Accuracy metric.
    """
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
    

def accuracy_simple(y_true, y_pred):
    """Accuracy metric using only hits and misses.

    Parameters
    ----------
    y_true : array-like of shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array-like of shape = [n_samples]
        Estimated target values.

    Returns
    -------
    accuracy : float
        Accuracy metric.
    """
   
    hits = 0
    misses = 0
    
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            hits += 1
        else:
            misses += 1
        
    return (hits/(hits + misses))*100, hits, misses
    
    # vp = 0
    # vn = 0
    # fp = 0
    # fn = 0
    
    # for i in range(len(y_true)):
    #     if y_true[i] == 1 and y_pred[i] == 1:
    #         vp += 1
    #     elif y_true[i] == 0 and y_pred[i] == 1:
    #         fp += 1
    #     elif y_true[i] == 1 and y_pred[i] == 0:
    #         fn += 1
    #     elif y_true[i] == 0 and y_pred[i] == 0:
    #         vn += 1
       
    # return (vp + vn)/(vp + vn + fp + fn)

def precision(y_true, y_pred):
    """Precision metric (Uses tp and fp).

    Parameters
    ----------
    y_true : array-like of shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array-like of shape = [n_samples]
        Estimated target values.

    Returns
    -------
    precision : float
        Precision metric.
    """
    tp = 0
    fp = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fp += 1
    return tp/(tp + fp)

def recall(y_true, y_pred):
    """Recall metric (Uses tp and fn).

    Parameters
    ----------
    y_true : array-like of shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array-like of shape = [n_samples]
        Estimated target values.

    Returns
    -------
    recall : float
        Recall metric.
    """
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fn += 1
    return tp/(tp + fn)

def f_one(y_true, y_pred):
    """F1 metric (Uses tp, tn, fp, fn).

    Parameters
    ----------
    y_true : array-like of shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array-like of shape = [n_samples]
        Estimated target values.

    Returns
    -------
    f1 : float
        F1 metric.
    """
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*pre*rec/(pre + rec)

