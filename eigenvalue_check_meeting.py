import numpy as np

length = 10
eta = 0.1
I = np.eye(length)
u1 =  np.random.randn(length,5)
u2 = np.random.randn(length,5)


value,vector = np.linalg.eig((u1@u1.T + eta*I)@(u2@u2.T + eta*I))
print(value,u1.T@u1,u2.T@u2)