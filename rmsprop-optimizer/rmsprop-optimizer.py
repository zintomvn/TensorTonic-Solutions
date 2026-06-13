import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.array(w)
    s = np.array(s)
    g = np.array(g)
    
    s = beta * s + (1 - beta) * (g**2)
    w_update = lr * g / (np.sqrt(s) + eps)
    new_w = w - w_update
    
    return new_w, s