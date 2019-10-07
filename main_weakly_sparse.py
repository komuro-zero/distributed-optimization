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
        self.sparsity_percentage = 0.5
        #self.lamb = 10**-3
        #self.eta = 0.2
        #self.rho = s
        #self.lamb*(10**2)
<<<<<<< HEAD
<<<<<<< HEAD
        self.lamb = 4*10**-3
        self.eta = 0.11
        self.rho = self.lamb*((1)**2)
        self.rho_checker(self.rho,self.lamb,self.eta)
        self.how_weakly_sparse = 0.1
        self.w_noise = 100

    def make_variables(self):
        w = randn(self.N,1)
        w_star = self.w_star_weakly_sparse(self.N,self.sparsity_percentage,self.how_weakly_sparse)
        #w_star = self.w_star(self.N,self.sparsity_percentage)
        U_all = randn(self.m,self.N)
        w_star_noise = w_star +randn(self.N,1)*(10**-(self.w_noise/10))
        #w_star_noise = w_star_noise/np.dot(w_star_noise.T,w_star_noise)
        d_all = np.dot(U_all,w_star_noise)
        L2 = np.dot(w_star.T,w_star)

        return w,w_star,U_all,d_all,L2,w_star_noise

    def run(self):
        w,w_star,U_all,d_all,L2,w_star_noise = self.make_variables()
        c = self.undirected_graph(self.m,self.r_i)
        self.disjoint_checker(c,self.m)
        w_all = self.make_w(self.m,self.N)
        
        #error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.001,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.0031,0.001,self.iteration)
        #error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,3.2,0.0021,self.lamb*((12)**2),self.iteration)
        #self.distributed_gradient_descent_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.006,self.iteration,c,w_all,wcgd)
        self.distributed_L1_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0031/self.m,0.24,self.iteration,c,w_all,wcl1)
        #self.distributed_mc_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,3.2/self.m,self.eta,self.rho,self.iteration,c,w_all,wcmc)
        #self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,c,w_all)
        #self.distributed_L1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-3,0.21,self.iteration,c,w_all)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,c,w_all)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,3.2/self.m,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,c,w_all)
=======
        self.lamb = 0.023
        self.eta = 0.499
        self.B = 1
=======
        self.lamb = 0.19
        self.eta = 0.0045
        self.B = 3
>>>>>>> dd57e56b3a59cf04095fe10cbbfda2121fdb7c85
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.1
        self.w_noise = 5

    def run(self):
        w,w_star,U_all,d_all,L2 = self.make_variables_noise_after(self.N,self.m,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        c = self.undirected_graph(self.m,self.r_i)
        self.disjoint_checker(c,self.m)
        w_all = self.make_w(self.m,self.N)

        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N)
        
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.0045,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,1.9,0.0044,self.iteration)
        self.lipschitz_checker_L1(U_all,self.m,0.0044,1.9)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,self.lamb,self.eta,self.rho,self.iteration)
        #self.distributed_gradient_descent_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,c,w_all,wcgd)
        #self.distributed_L1_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-3,0.21,self.iteration,c,w_all,wcl1)
        #self.distributed_mc_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((12)**2),self.iteration,c,w_all,wcmc)
        #self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.255,self.iteration,c,w_all)
        #self.distributed_L1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,5*10**-4,0.25,self.iteration,c,w_all)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,c,w_all)
        #self.distributed_convexity_checker(self.B,self.lamb/self.m,U_all,self.N)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,c,w_all)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,c,w_all)
<<<<<<< HEAD
        self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,self.eta,self.rho,self.iteration,c,w_all)
>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04
=======
>>>>>>> dd57e56b3a59cf04095fe10cbbfda2121fdb7c85
        
        plt.legend()
        plt.show()
        x  = range(len(wcgd))
        plt.plot(x,wcgd,label = "LMS")
        plt.plot(x,wcl1,label = "L1")
        plt.plot(x,wcmc,label = "mc")
        plt.plot(x,w_star,color = "black")
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