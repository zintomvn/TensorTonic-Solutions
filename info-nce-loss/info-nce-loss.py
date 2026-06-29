import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    N = len(Z1)

    Z1 = np.array(Z1)
    Z2 = np.array(Z2)

    # Normalize
    # eps = 1e-12
    # Z1 = Z1 / (np.linalg.norm(Z1, axis=1, keepdims=True) + eps)
    # Z2 = Z2 / (np.linalg.norm(Z2, axis=1, keepdims=True) + eps)
    # Similarity matrix: Z1 -> Z2
    similarities = Z1 @ Z2.T / temperature

    # Stable log-softmax
    shifted = similarities - np.max(
        similarities,
        axis=1,
        keepdims=True
    )

    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )

    # Positive pairs nằm trên đường chéo
    labels = np.arange(len(Z1))

    loss = -log_probs[
        np.arange(len(Z1)),
        labels
    ].mean()

    return loss