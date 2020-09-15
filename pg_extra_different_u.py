import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 100
        self.m = 1000
        self.r_i = 80
        self.iteration = 4000
        self.sparsity_percentage = 0.1
        self.lamb = 1
        self.eta = 7.984864571147688e-05
        self.B = 3.5
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        
        #m 1000
        error_2,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.4,7.984864571147688e-05,0.85,self.iteration,self.m)
        error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,0.4,7.984864571147688e-05,0.85,self.iteration,self.m)
        plt.hlines(error[-1],0,self.iteration,color = "red",label = "Centralized MC penalty") 

        # plt.plot(range(len(wcmc)),wcmc,label = 'Centralized MC penalty')
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.1/self.m,0.00093,3.5/self.m,self.iteration,graph,w_all)


        plt.grid(which = "major")
        plt.xlabel("Iterations")
        plt.ylabel("System Mismatch (dB)")
        plt.legend()
        plt.show()
        # x = range(len(wdgd))
        # plt.plot(x,wdgd,label = "extra")
        # plt.plot(x,wcl1,label = "L1")
        # # plt.plot(x,extra_mc,label = "mc")
        # plt.plot(x,w_star,color = "black")
        # plt.legend()
        # plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()