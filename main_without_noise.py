import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_functions_revised import functions


np.random.seed(0)
class distributed_updates(functions):


    def __init__(self):

        self.N = 30
        self.m = 200
        self.r_i = 40
        self.iteration = 100
        self.sparsity_percentage = 0.2
        self.lamb = 0.31
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0
        self.w_noise = 0

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_no_noise(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.0047,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.00001,0.005,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.00005,0.005,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.0001,0.005,self.iteration)
        error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,self.lamb,self.eta,self.rho,self.iteration)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.lipschitz_checker_L1(U_all,self.m,0.0044,1.9)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        #self.distributed_gradient_descent_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,graph,w_all,wcgd)
        #self.distributed_L1_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-3,0.21,self.iteration,graph,w_all,wcl1)
        #self.distributed_mc_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((12)**2),self.iteration,graph,w_all,wcmc)
        self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,graph,w_all)
        self.distributed_L1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.007/self.m,0.2,self.iteration,graph,w_all)
        self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb/self.m,0.13,2*(10**-3)*((1.5)**2),self.iteration,graph,w_all)
        self.distributed_mc_compare(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,self.lamb/self.m,0.2,2*(10**-3)*((1.5)**2),self.iteration,graph,w_all)
        self.distributed_convexity_checker(4.3,2*10**-3,U_all,self.N)
        self.centralized_convexity_checker(1.6,2.3*10**-2,U_all,self.N)
        #self.distributed_convexity_checker(self.B,self.lamb/self.m,U_all,self.N)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,graph,w_all)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,graph,w_all)
        
        plt.legend()
        plt.show()
        # x  = range(len(wcgd))
        # plt.plot(x,wcgd,label = "LMS")
        # plt.plot(x,wcl1,label = "L1")
        # plt.plot(x,wcmc,label = "mc")
        # plt.plot(x,w_star,color = "black")
        # plt.legend()
        # plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()