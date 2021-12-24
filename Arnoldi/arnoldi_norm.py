'''
  Authors: Dawit Anelay
           Marco Pitex
           Yohannis Telila
           
    Date : 24 / 11 / 2021 

  Implemetation of the Arnoldi iterations algorithm to approximating the largest eigenvalue ofA^T A(orA A^T) using the Arnoldi process.

'''

# Imports 
import numpy as np
import math 
from arnoldi_iteration import arnoldi_iteration,chope_lastrow,QR_algorithm ,func_

epsilon = 1e-8 

class arnoldi_norm:
     # Initialize the arnoldi_norm algorithm
    def __init__(self, M,b,max_iter,verboose=False):  
        '''
        Parameters
        ----------
               M : where M = A^T A  
               b : ndarray, initial guess. Can be set to be any numpy vector
               max_iter : int , maximum number of iterations.
               verbose : bool, optional, if True prints the progress of the algorithm.

        '''
        self.M = M 
        self.b = b 
        self.max_iter = max_iter
        #self.norm_matrix = None 
        self.verboose = verboose  

    
      # arnoldi_norm algorithm 
    def arnoldi_norm(self): 

          # initialize the variables.
        M   = self.M
        b   = self.b 
        #prod = func_(M) 
        max_iter = self.max_iter
        λ = [] 
        H_1 = arnoldi_iteration(M, b, 1) 
        λ_1 = max(QR_algorithm(H_1)) 
        λ.append( λ_1)  
        
        for i in range(2,max_iter):
          H_i = arnoldi_iteration(M, b, i) 
          λ_i = max(QR_algorithm(H_i)) 
          λ.append(λ_i) 
          if (abs(λ[i-1] - λ[i-2]) < epsilon ):
            matrix_norm = np.sqrt(λ[i-1])
            return matrix_norm,i,λ  

            
  
        matrix_norm = np.sqrt(λ[-1]) 
        
            
        return matrix_norm , i , λ
        
