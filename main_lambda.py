import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression     import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 10
        self.m = 100
        self.r_i = 80
        self.iteration =5000
        self.sparsity_percentage = 0.35
        self.lamb = 4.25
        self.eta = 0.00116
        self.B = 0.001
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 10

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        lambda_list = []
        l1_error_list = []
        mc_error_list = []
        mc_error_rho = []
        small_eig, big_eig = self.U_eigenvalue(U_all)
        for i in range(20):
            lamb = (i+1)*0.5
            error_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb/self.m,0.002849,self.rho,self.iteration,graph,w_all)
            for j in range(50):
                if j == 0:
                    rho = 10**-10
                else:
                    rho = (j)*0.2
                if rho <small_eig:
                    error_mc =self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb/self.m,0.002849,rho/self.m,self.iteration,graph,w_all)
                    mc_error_rho.append(error_mc[-1])
                    if j % 100 == 0:
                        print("iteration,",i,j)
                else:
                    break
            lambda_list.append(lamb)
            l1_error_list.append(error_l1[-1])
            mc_error_list.append(min(mc_error_rho))
            mc_error_rho = []
        plt.xlabel("lambda")
        plt.ylabel("System Mismatch (dB)")
        plt.plot(lambda_list,l1_error_list,label = "PG-EXTRA with L1 penalty")
        plt.plot(lambda_list,mc_error_list,label = "PG-EXTRA with MC penalty")
        plt.legend()
        plt.grid(which = "major")
        plt.show()

    

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()

    #l1 -44 extra -39 mc -44.7弱