import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions



np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 30
        self.m = 100
        self.r_i = 20
        self.iteration = 600
        self.sparsity_percentage = 0.2
        self.lamb = 0.31
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.01
        self.w_noise = 20

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
           
        # self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.088,1,self.iteration,graph,w_all)
        # extra = self.ADDA(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.54,1,self.iteration,graph,w_all)
        # extra = self.dista(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.02,0.22,1,self.iteration,graph,w_all,0.5)
        # extra = self.FDDA(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.907,0.06,1,self.iteration,graph,w_all)
        # extra = self.dfbbs(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00056,0.088,1,self.iteration,graph,w_all)
        # extra = self.APM(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.000000001,0.8,1,self.iteration,graph,w_all)
        # extra = self.pad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00028,0.143,0.07,self.iteration,graph,w_all,0.08)
        # self.pg_extra_l1_projection(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.095,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1_projection(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.255,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.25,1,self.iteration,graph,w_all)
        # self.pg_extra_l1_step_size(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,1.305,1,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.14,0.088,self.iteration,graph,w_all)

        extra = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.041/self.m,1.3,0.00145,self.iteration,graph,w_all,0.009,1.1,1,1)


        # extra = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.07145/self.m,0.085,0.0009,self.iteration,graph,w_all)
        # extra = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.085/self.m,0.085,0.001,self.iteration,graph,w_all)
        # self.pxg_extra_mc_convex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2/self.m,0.088,0.0001,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.141,0.034,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.061/self.m,0.1405,0.0016,self.iteration,graph,w_all)
        # self.pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.088,0.034,self.iteration,graph,w_all)
        # self.atc_pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2005/self.m,0.24,0.034,self.iteration,graph,w_all)

        plt.legend()
        plt.show()
        x  = range(len(w_star))
        plt.plot(x,extra,label = "extra")
        # plt.plot(x,wcl1,label = "L1")
        # plt.plot(x,wcmc,label = "mc")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()