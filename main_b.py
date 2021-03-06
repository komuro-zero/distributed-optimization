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
        self.iteration =10000
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
        
        b_list = []
        l1_error_list = []
        mc_error_list = []
        mc_error_lamb = []
        small_eig, big_eig = self.U_eigenvalue(U_all)
        error_l1,wl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,3.83,0.002849,self.iteration)
        
        for i in range(40):
            if i == 0:
                b = 10**-10
            else:
                b = 0.05*i
            for j in range(50):
                lamb = 0.1*(j+1)
                if b <(small_eig/lamb)**0.5:
                    error_mc,wmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,lamb,0.002849,lamb*b**2,self.iteration,self.m)
                    mc_error_lamb.append(error_mc[-1])
                    if j % 100 == 0:
                        print("iteration,",i,j)
            b_list.append(b)
            mc_error_list.append(min(mc_error_lamb))
            mc_error_lamb = []

        plt.xlabel("eta")
        plt.ylabel("System Mismatch (dB)")
        plt.plot(b_list,mc_error_list,label = "PG-EXTRA with MC penalty")
        plt.hlines(error_l1[-1],min(b_list),max(b_list),label = "PG-EXTRA with L1 penalty") 
        plt.legend()
        plt.grid(which = "major")
        plt.show()


    

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()

    #l1 -44 extra -39 mc -44.7弱