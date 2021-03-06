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
        self.iteration = 20
        self.sparsity_percentage = 0.1
        self.lamb = 1
        self.eta = 0.00056
        self.B = 7
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        

        #N100 m1000 ri80 noise30 
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00096,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,1.0605,0.00096,self.iteration)
        error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,1.81,0.00096,5.75,self.iteration,self.m)
        # w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)

        # w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)

        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,4.3,0.002849,3.7,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,4.2,0.002849,3.7,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,4.1,0.002849,3.7,self.iteration,self.m)

        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.00000001,0.002849,0.055,self.iteration,self.m)

        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.000000001,0.002849,0.055,self.iteration,self.m)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.00000000000001,0.002849,self.iteration)





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
        x = range(len(wcmc))
        plt.plot(x,wcmc,label = "mc")
        plt.plot(x,wcl1,label = "l1")
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