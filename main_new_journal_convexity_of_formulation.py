import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages





np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 10
        self.m = 50
        self.r_i = 5
        self.iteration = 600
        self.sparsity_percentage = 0.2
        self.lamb = 0.31    
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.001
        self.w_noise = 30
        self.normal_distribution = True
        self.w_zero = True

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
        I,c_1,c_2,c_tilde,eta = self.extra_setup(graph,self.r_i,U_all,1,1,self.rho,1)


        big_U = np.zeros((self.m,self.N*self.m))
        for i in range(self.m):
            big_U[i][self.N*i:self.N*(i+1)] = U_all[i]
        
        I2 = np.eye(self.N)
        V = np.kron(np.eye(self.m) -c_2,I2)

        eigenv_uv,eigenve_uv = np.linalg.eig(big_U.T@big_U + V.T@V)
        eigenvalue,eigenvector = np.linalg.eig(big_U.T@big_U + V.T@V - sorted(eigenv_uv)[0]*np.eye(self.m*self.N))
        eigenv_v,eigenve_v = np.linalg.eig(V.T@V)
        eigenv_u,eigenve_u = np.linalg.eig(big_U.T@big_U)
        # eigenv_uv,eigenve_uv = np.linalg.eig(big_U.T@big_U + V.T@V)
        # print(sorted(eigenv_u)[-self.m:],sorted(eigenv_v)[self.N+1],sorted(eigenv_uv)[0])
        # eigenvalue,eigenvector = np.linalg.eig( big_U.T@big_U - 0.0000000000000001*np.eye(self.m*self.N))
        # eigenvalue,eigenvector = np.linalg.eig( V.T@V - 0.0000000000000001*np.eye(self.m*self.N))
        print(min(eigenvalue))
        plt.hist(eigenvalue)
        plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()