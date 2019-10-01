import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_functions_wj import functions


np.random.seed(0)
class distributed_updates(functions):


    def __init__(self):

        self.N = 30
        self.m = 200
        self.r_i = 40
        self.iteration = 100
        self.sparsity_percentage = 0.2
        #self.lamb = 10**-3
        #self.eta = 0.2
        #self.rho = s
        #self.lamb*(10**2)
        self.lamb = 0.023
        self.eta = 0.499
        self.B = 1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.1
        self.w_noise = 20

    def run(self):
        w,w_star,U_all,d_all,L2 = self.make_variables(self.N,self.m,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        c = self.undirected_graph(self.m,self.r_i)
        self.disjoint_checker(c,self.m)
        w_all = self.make_w(self.m,self.N)

        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N)
        
        #error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.001,self.iteration)
        #error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.0031,0.001,self.iteration)
        #error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,3.2,0.0021,self.lamb*((12)**2),self.iteration)
        #self.distributed_gradient_descent_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,c,w_all,wcgd)
        #self.distributed_L1_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-3,0.21,self.iteration,c,w_all,wcl1)
        #self.distributed_mc_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((12)**2),self.iteration,c,w_all,wcmc)
        #self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,c,w_all)
        #self.distributed_L1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-3,0.21,self.iteration,c,w_all)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,c,w_all)
        self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,self.eta,self.rho,self.iteration,c,w_all)
        
        plt.legend()
        plt.show()
"""
    #not completed
    def run_distributed(self):
        w,w_star,U_all,d_all,L2 = self.make_variables()  

        lamb_l1 = 9*10**-5
        lamb_mc = 10**-3
        eta_gd = 0.1
        eta_l1 = 0.09
        eta_mc = 0.01

        self.distributed(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb_l1,lamb_mc,eta_gd,eta_l1,eta_mc,self.rho,self.iteration)
"""


if __name__ == "__main__":
    simulation = distributed_updates()
    for i in range(1):
        simulation.run()