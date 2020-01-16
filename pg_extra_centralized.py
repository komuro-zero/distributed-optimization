import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 10
        self.m = 100
        self.r_i = 80
        self.iteration =50  
        self.sparsity_percentage = 0.35
        self.lamb = 1.69
        self.eta = 0.00116
        self.B = 0.001
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 10

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        # extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.002849,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0000001,0.00657,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.000001,0.00657,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00001,0.00657,self.rho,self.iteration,graph,w_all)
        # error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.01*self.m,0.00657,self.m*0.0001,self.iteration)
        

        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00074,0.002849,self.rho,self.iteration,graph,w_all)
        
        #N60 m100 ri80 noise30 1db 
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.0164,0.002849,0.055,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,0.01,0.002849,0.055,self.iteration,self.m)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.00723,0.002849,self.iteration)

        #N60 m100 ri80 noise20 1db density 30
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.0465,0.002849,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.07,0.002849,0.2,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,0.065,0.002849,0.35,self.iteration,self.m)


        #N60 m100 ri80 noise10 1db
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.95,0.002849,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.98,0.002849,2,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,0.915,0.002849,6.01,self.iteration,self.m)

        #N60 m100 ri80 noise10 1db
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.46,0.002849,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.66,0.002849,1.71,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1,0.002849,6.57,self.iteration,self.m)

        #N100 m100 ri80 noise10 1db
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.471,0.002849,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.64,0.002849,0.049,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1,0.002849,0.049,self.iteration,self.m)

        #N10 m100 ri80 noise10 1db
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.002849,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,1.35,0.002849,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,1.36,0.002849,1.25,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.365,0.002849,1,self.iteration,self.m)

        #N10 m100 ri80 noise10 1db
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.002849,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,1.35,0.002849,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,1.36,0.002849,1.25,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.365,0.002849,1,self.iteration,self.m)

        #N10 m100 ri80 noise10 1db
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.002849,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,1.35,0.002849,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,1.69,0.002849,1.885,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.69,0.002849,1.5,self.iteration,self.m)

        #N10 m100 ri80 noise10 1db
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.002849,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,1.35,0.002849,self.iteration)
        error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,1.69,0.002849,1.885,self.iteration,self.m)
        error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.69,0.002849,1.5,self.iteration,self.m)

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