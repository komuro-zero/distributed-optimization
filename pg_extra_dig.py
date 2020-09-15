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
        self.iteration = 100000
        self.sparsity_percentage = 0.1
        self.lamb = 0.55
        self.eta = 0.0033
        self.B = 0.001
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        # self.params_checker(self.rho,0.67,0.0033,U_all,self.B,200,100,graph)
        # self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        
        #m 1000
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,1000,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.81,0.00096,5.75,25000,1000)
        extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,1000,self.r_i,0.29/1000,0.0052,0.02/1000,self.iteration,graph,w_all)
        extra_mc = self.pg_extra_dig_mc_soft(U_all,d_all,w_star,L2,self.N,1000,self.r_i,0.29/1000,0.0052,0.02/1000,self.iteration,graph,w_all)
        # plt.hlines(error[-1],0,self.iteration,color = "red",label = "Centralized m = 1000") 
        # plt.hlines(error[-1],0,self.iteration,color = "red",label = "Centralized m = 100") 

        # plt.plot(range(len(extra_mc)),extra_mc,label = 'm = 100')


        # #m 1000
        # w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,1000,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.1,0.00093,3.5,400,1000)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,1000,self.r_i,1.1/1000,0.00093,3.5/1000,self.iteration,graph,w_all)
        # plt.hlines(error[-1],0,self.iteration,color = "pink",label = "Centralized m = 1000") 

        # plt.plot(range(len(extra_mc)),extra_mc,label = 'm = 1000')


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