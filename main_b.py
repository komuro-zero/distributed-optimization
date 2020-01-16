import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression     import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 10
        self.m = 100
        self.r_i = 80
        self.iteration =2000
        self.sparsity_percentage = 0.35
        self.lamb = 4.25
        self.eta = 0.00116
        self.B = 0.001
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 10

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.0047,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.00001,0.005,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.00005,0.005,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.0001,0.005,self.iteration)
        # error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,self.lamb,self.eta,self.rho,self.iteration)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        # self.lipschitz_checker_L1(U_all,self.m,0.0044,1.9)
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
        # extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.00657,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00000001,0.00657,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.07098/self.m,0.00657,self.rho,self.iteration,graph,w_all)

        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00009,0.002849,0.00021,self.iteration,graph,w_all)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.008,0.002849,0.0185,self.iteration,self.m)

        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0000723,0.002849,self.rho,self.iteration,graph,w_all)
        
        
        b_list = []
        l1_error_list = []
        mc_error_list = []
        mc_error_lamb = []
        small_eig, big_eig = self.U_eigenvalue(U_all)
        error_l1,wl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,3.83,0.002849,self.iteration)
        for i in range(20):
            if i == 0:
                b = 10**-10
            else:
                b = 0.1*i
            for j in range(50):
                lamb = 0.1*(j+1)
                if b <(small_eig/lamb)**0.5:
                    error_mc,wmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,lamb,0.002849,lamb*b**2,self.iteration,self.m)
                    mc_error_lamb.append(error_mc[-1])
                    if j % 100 == 0:
                        print("iteration,",i,j)
            b_list.append(b)
            mc_error_list.append(min(mc_error_lamb))
            mc_error_lamb = []

        plt.xlabel("eta")
        plt.ylabel("Mean Square Error (dB)")
        plt.plot(b_list,mc_error_list,label = "PG-EXTRA with MC penalty")
        plt.hlines(error_l1[-1],min(b_list),max(b_list),label = "PG-EXTRA with L1 penalty") 
        plt.legend()
        plt.show()


    
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00885/self.m,0.002849,0.000999/self.m,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.07828/self.m,0.002849,0.00999/self.m,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.07828/self.m,0.002849,0.00999/self.m,self.iteration,graph,w_all)

        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,0.00000001,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,0.0000001,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,0.000001,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb/self.m,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_mc_nonconvex = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00000007,0.0082,0.00000007*((0.01)**2),self.iteration,graph,w_all)
        # extra_mc_L2 = self.pg_extra_mc_L2_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0000001,0.0086,self.rho,self.iteration,graph,w_all)

        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.lamb*((0.001)**2),self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.lamb*((0.0000001)**2),self.iteration,graph,w_all)
        # plt.legend()
        # plt.xlabel("iterations")
        # plt.ylabel("Mean Square Error (dB)")
        # plt.show()
        # x = range(len(extra_l1))
        # # plt.plot(x,extra,label = "extra")
        # plt.plot(x,extra_l1,label = "L1")
        # # plt.plot(x,extra_mc,label = "mc")
        # # plt.plot(x,wdmc1,label = "distributed mc")
        # plt.plot(x,w_star,color = "black")
        # plt.legend()
        # plt.show()
        # print(extra_mc)

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()

    #l1 -44 extra -39 mc -44.7弱