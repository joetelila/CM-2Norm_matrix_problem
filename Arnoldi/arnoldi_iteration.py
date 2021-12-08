import numpy as np

import numpy as np

def arnoldi_iteration(A, b, n: int):
    """Computes an orthonormal basis Q for the Krylov space {b,Ab,A²b,...},

    Arguments
      A: m × n array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1
    
    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      H: (n + 1) x n array, A on basis Q.
    """
    m = len(b)  
    H = np.zeros((n+1,n))  
    Q = np.zeros((m,n+1))  
    # Normalize the input vector and Use it as the first Krylov vector q1 = b/|b| 
    Q[:,0] =b/np.linalg.norm(b,2)   
    for i in range(1,n+1): 
        v = np.dot(A,Q[:,i-1])  # Generate a new candidate vector 
        for j in range(i): 
            H[j,i-1] = np.dot(Q[:,j].T, v) 
            v = v - H[j,i-1] * Q[:,j]
    
        H[i,i-1] = np.linalg.norm(v,2) 
        Q[:,i] = v / H[i,i-1] 

    return Q, H  





