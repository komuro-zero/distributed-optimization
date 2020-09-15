import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 100
        self.m = 750
        self.r_i = 80
        self.iteration = 25
        self.sparsity_percentage = 0.1
        self.lamb = 1.1
        self.eta = 0.00093
        self.B = 2.4
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        # self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        # self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.1,0.00093,3.5,self.iteration,self.m)
        error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,1.1,0.00093,3.5,self.iteration,self.m)

        plt.grid(which = "major")
        plt.xlabel("Iterations")
        plt.ylabel("System Mismatch (dB)")
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()