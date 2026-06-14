import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    N = y_pred.shape[0]
    y_pred_index = np.arange(N)
    loss = -np.sum(np.log(y_pred[y_pred_index, y_true])) / N
    return loss