import numpy as np
from numpy import linalg as la
from scipy import sparse


'''
Authors: Dawit Anelay
         Marco Pitex
         Yohannis Telila
'''

# generate x0_m1 matrix
n= 1000
x0_m1 = np.round(np.random.randn(n),decimals = 3)
np.savetxt("x0_m1.txt", x0_m1)
print("[success] x0_m1 generated and saved.")


# Generate M2 matrix
M2 = np.random.randn(1000, 100)
np.savetxt("M2.txt", M2)
print("[success] M2 generated and saved.")
# generate x0 for M2
x0_m2 = np.round(np.random.randn(100),decimals = 3)
np.savetxt("x0_m2.txt", x0_m2) 
print("[success] x0_m2 generated and saved.")
 

# Generate M3 matrix
M3 = np.random.randn(100, 1000) 
np.savetxt("M3.txt", M3)
print("[success] M3 generated and saved.")
# generate x0 for M3
x0_m3 = np.round(np.random.randn(1000),decimals = 3) 
np.savetxt("x0_m3.txt", x0_m3)
print("[success] x0_m3 generated and saved.") 
 

# Generate M4 matrix
M4 = np.random.randn(100,100)
np.savetxt("M4.txt", M4)
print("[success] M4 generated and saved.")
# generate x0 for M4
x0_m4 = np.round(np.random.randn(100),decimals = 3)
np.savetxt("x0_m4.txt", x0_m4)
print("[success] x0_m4 generated and saved.")


# Generate M5 matrix
M5 = sparse.random(1000, 100, density=0.3, data_rvs=np.random.randn)
M5 = np.squeeze(np.asarray(M5.todense()))
np.savetxt("M5.txt", M5)

# generate x0 for M5
x0_m5 = np.round(np.random.randn(100),decimals = 3)
np.savetxt("x0_m5.txt", x0_m5)
print("[success] x0_m5 generated and saved.") 

# Generate M6 matrix
cond_P = 1e18     # Condition number
log_cond_P = np.log(cond_P)
n= 1000
exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n )/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
s = np.exp(exp_vec)
S = np.diag(s)
U, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
V, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
P = U.dot(S).dot(V.T)
P = P.dot(P.T) / 1e7

np.savetxt("M6.txt", P)
print("[success] M6 generated and saved.")
# generate x0_m1 matrix
x0_m1 = np.round(np.random.randn(n),decimals = 3)
np.savetxt("x0_m6.txt", x0_m1)
print("[success] x0_m6 generated and saved.")