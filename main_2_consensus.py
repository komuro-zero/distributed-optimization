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
        self.iteration = 1000
        self.sparsity_percentage = 0.2
        self.lamb = 0.31
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.01
        self.w_noise = 20

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        # self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.029,self.rho,self.iteration,graph,w_all)
        # self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.01,self.rho,self.iteration,graph,w_all)
        
        # self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.056,self.rho,self.iteration,graph,w_all)
        # extra = self.atc_extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.37,self.rho,self.iteration,graph,w_all)
        # self.atc_extra_tilde(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.26,self.rho,self.iteration,graph,w_all)
        # self.atc_extra_2(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.11,self.rho,self.iteration,graph,w_all)
        # self.dig(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.03,self.rho,self.iteration,graph,w_all)
        # self.atc_dig(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.37,self.rho,self.iteration,graph,w_all)
        
        # self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.088,1,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.14,1,self.iteration,graph,w_all)
        # extra = self.ADDA(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.54,1,self.iteration,graph,w_all)
        # extra = self.dista(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.02,0.22,1,self.iteration,graph,w_all,0.5)
        # extra = self.FDDA(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.907,0.06,1,self.iteration,graph,w_all)

        # extra = self.dfbbs(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00056,0.088,1,self.iteration,graph,w_all)
        # extra = self.APM(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.000000001,0.8,1,self.iteration,graph,w_all)
        # extra = self.pad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00028,0.143,0.07,self.iteration,graph,w_all,0.08)
        # self.pg_extra_l1_projection(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.095,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1_projection(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.26,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.25,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1_3(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.01,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1_3(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.03,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1_3(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.06,1,self.iteration,graph,w_all)
    
        # self.pg_extra_mc_consensus_graph(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.088,0.034,self.iteration,graph,w_all)
        # self.pg_extra_mc_consensus_violation_consensus_graph(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.0818,0.034,self.iteration,graph,w_all)

        # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.06,0.034,self.iteration,graph,w_all)
        # self.prox_dgd_consensus_graph(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.06,0.034,self.iteration,graph,w_all)
        # self.prox_dgd_consensus_violation_consensus_graph(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.06,0.034,self.iteration,graph,w_all)
        # self.pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.088,0.034,self.iteration,graph,w_all)
        self.pg_extra_mc_consensus_graph(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.07,0.034,self.iteration,graph,w_all)
        self.pg_extra_mc_consensus_violation_consensus_graph(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.07,0.034,self.iteration,graph,w_all)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.0818,0.034,self.iteration,graph,w_all)
        # extra = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.08/self.m,0.084,0.001,self.iteration,graph,w_all)
        # extra = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.08/self.m,0.085,0.001,self.iteration,graph,w_all)
        # extra = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.08/self.m,0.086,0.001,self.iteration,graph,w_all)
        # self.atc_pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2005/self.m,0.24,0.034,self.iteration,graph,w_all)

        # extra = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.11/self.m,3,0.0015,self.iteration,graph,w_all,0.1,1,1)
        

        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.0045,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.007,0.0044,self.iteration)
        # error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,self.lamb,self.eta,self.rho,self.iteration)
        # self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        # self.lipschitz_checker_L1(U_all,self.m,0.0044,1.9)
        # self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        #self.distributed_gradient_descent_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,graph,w_all,wcgd)
        #self.distributed_L1_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-3,0.21,self.iteration,graph,w_all,wcl1)
        #self.distributed_mc_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((12)**2),self.iteration,graph,w_all,wcmc)
        # self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2,self.iteration,graph,w_all)
        # self.distributed_L1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.007/self.m,0.2,self.iteration,graph,w_all)
        # self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb/self.m,0.013,2*(10**-3)*((1.5)**2),self.iteration,graph,w_all)
        # self.distributed_mc_compare(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,self.lamb/self.m,0.013,2*(10**-3)*((1.5)**2),self.iteration,graph,w_all)
        # self.distributed_convexity_checker(4.3,2*10**-3,U_all,self.N)
        # self.centralized_convexity_checker(1.6,2.3*10**-2,U_all,self.N)
        #self.distributed_convexity_checker(self.B,self.lamb/self.m,U_all,self.N)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,graph,w_all)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.3*10**-2,0.17,2.3*(10**-2)*((4.3)**2),self.iteration,graph,w_all)
        
        plt.title("Consensus Violation")
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