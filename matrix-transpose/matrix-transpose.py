import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    # pass
    result = []
    for i in range(len(A[0])):
        sub_list = []
        for j in range(len(A)):
            sub_list.append(A[j][i])
        result.append(sub_list)
    return np.array(result)
