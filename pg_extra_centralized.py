import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 60
        self.m = 100
        self.r_i = 80
        self.iteration = 500
        self.sparsity_percentage = 0.2
        self.lamb = 0.0088
        self.eta = 0.002849
        self.B = 3.73
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        # extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.00657,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0000001,0.00657,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.000001,0.00657,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00001,0.00657,self.rho,self.iteration,graph,w_all)
        # error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.01*self.m,0.00657,self.m*0.0001,self.iteration)
        
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00657,self.iteration)

        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00074,0.002849,self.rho,self.iteration,graph,w_all)
        
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.0000723*self.m,0.002849,self.iteration)
        error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.02,0.002849,0.065,self.iteration,self.m)
        # error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.00883,0.002849,0.005,self.iteration)




        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.07828,0.005,0.00999,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.07828,0.001,0.00999,self.iteration,self.m)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc_non = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.05,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.001,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0001,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00001,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.001,0.00657,0.00000001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0001,0.00657,0.0000000000001,self.iteration,graph,w_all)

        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,0.000657,0.1*(0.9**2),self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,0.000657,0.1*(0.8**2),self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,0.00000657,self.rho,self.iteration,graph,w_all)
        plt.legend()
        plt.show()
        x = range(len(extra_mc))
        plt.plot(x,wcmc,label = "cent")
        plt.plot(x,extra_mc,label = "pg extra")
        # plt.plot(x,extra_mc_non,label = "extra")
        # plt.plot(x,extra_l1,label = "L1")
        # plt.plot(x,extra_mc,label = "mc")
        # plt.plot(x,wdmc1,label = "distributed mc")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()
        # print(extra_mc)

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()