import numpy as np

QR_ALGORITHM_TOLERANCE = 1e-8 

def arnoldi_iteration(A, b, n: int):
    """Computes an orthonormal basis Q for the Krylov space {b,Ab,A²b,...},

    Arguments
      A: m × m array
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

    return H  

def func_(A):
  """
  Given a matrix A compute product  A A^T 

    Arguments
      A: m × n array
    
    Returns
      prodcut_of_matrix: where prodcut_of_matrix = A A^T
  """
  return np.dot(A,A.T)   

def func_2(A):
    return np.dot(A.T,A)
  

def chope_lastrow(H):
  """
  Chope the last row of H: (n + 1) x n array,and return H:(n) x n a squre matrix 

    Arguments
      H: m × n array
    
    Returns
      H: (n) x n array,
  """
  return np.delete(H, (len(H)-1), axis=0)  

def QR_algorithm(H):
    H = chope_lastrow(H)
    n, m = H.shape
    if m!= n:
        raise np.linalg.LinAlgError("Array must be square.") 
        
    convergence_measure = []
    λ = np.zeros((n, ), dtype='float') 
    n -= 1
    while n > 0:
       
        Q,R = np.linalg.qr(H)
        H = np.dot(R, Q)

        convergence_measure.append(np.abs(H[n, n - 1])) 
        
        if convergence_measure[-1] < QR_ALGORITHM_TOLERANCE:

            λ[n] = H[n, n]
            H = H[:n, :n]
            n -= 1

    λ[0] = H 
    return λ 





