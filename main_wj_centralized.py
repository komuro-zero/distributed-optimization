import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
import numpy.linalg as LA


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 100
        self.m = 500
        self.r_i = 200
        self.iteration = 1000
        self.sparsity_percentage = 0.2
        self.lamb = 0.000001
        self.eta = 0.008
        self.B = 0.01
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 0

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_no_noise(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        square = U_all.T@U_all
        eigenvalue = LA.eig(square)[0]
        cent_step = 2/max(eigenvalue)
        print(cent_step)
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00175,self.iteration)
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.0018,self.iteration)
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.0019,self.iteration)
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.01,self.iteration)
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.51,self.iteration)
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.004,self.iteration)
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.004,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.00001,0.005,self.iteration)
        # error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,self.lamb,self.eta,self.rho,self.iteration)
        # self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        # self.lipschitz_checker_L1(U_all,self.m,0.0044,1.9)
        # self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        # self.distributed_gradient_descent_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,graph,w_all,wcgd)
        # self.distributed_L1_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-3,0.21,self.iteration,graph,w_all,wcl1)
        # self.distributed_mc_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((12)**2),self.iteration,graph,w_all,wcmc)
        # self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,graph,w_all)
        # self.distributed_L1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.007/self.m,0.2,self.iteration,graph,w_all)
        # self.distributed_convexity_checker(1,self.lamb/self.m,U_all,self.N)
        # wdmc = self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb/self.m,0.01,(self.lamb/self.m)*((1)**2),self.iteration,graph,w_all)
        # self.distributed_mc_compare(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,self.lamb/self.m,0.01,(self.lamb/self.m)*((1)**2),self.iteration,graph,w_all)
        # self.distributed_mc_compare(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,self.lamb/self.m,0.001,(self.lamb/self.m)*((1)**2),self.iteration,graph,w_all)
        # wdmc1 = self.distributed_mc_compare(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,self.lamb/self.m,self.eta,(self.lamb/self.m)*((0.1)**2),self.iteration,graph,w_all)
        # self.centralized_convexity_checker(1.6,2.3*10**-2,U_all,self.N)
        # self.distributed_convexity_checker(self.B,self.lamb/self.m,U_all,self.N)
        # self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,graph,w_all)
        # self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,graph,w_all)
        # extra  = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.0084,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00000005,0.0085,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00000007,0.0085,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0000001,0.0085,self.rho,self.iteration,graph,w_all)
        # extra_mc_nonconvex = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00000007,0.0082,0.00000007*((0.01)**2),self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0000002,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_mc_L2 = self.pg_extra_mc_L2_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0000001,0.0086,self.rho,self.iteration,graph,w_all)

        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.lamb*((0.001)**2),self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.lamb*((0.0000001)**2),self.iteration,graph,w_all)
        plt.legend()
        plt.show()
        x = range(len(extra_l1))
        plt.plot(x,extra,label = "extra")
        plt.plot(x,extra_l1,label = "L1")
        # plt.plot(x,extra_mc,label = "mc")
        # plt.plot(x,wdmc1,label = "distributed mc")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()
        # print(extra_mc)

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()