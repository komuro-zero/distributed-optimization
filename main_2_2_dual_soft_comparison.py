import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages



np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 32
        self.m = 64
        self.r_i = 16
        self.iteration = 300
        self.sparsity_percentage = 0.3
        self.lamb = 0.31
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0
        self.w_noise = 10
        self.normal_distribution = True

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution)
        plt.rcParams["font.family"] = "Times New Roman" 
        pdf = PdfPages('main performance comparison journal.pdf')
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 10 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        # self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.01,self.iteration)
        # error,wcmc = self.centralized_L1(U_all,d_all,w,w_star,L2,1.2,0.01,self.iteration)
        
        error,w_mc = self.centralized_mc(U_all,d_all,w,w_star,L2,2,0.01,8,1000)
        error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_mc,L2,2,0.01,8,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,2,0.01,5.05,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,2,0.01,5.05,self.iteration,self.m)
        
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.06,0.0001,0.03,self.iteration,self.m)
        
        # extra = self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,self.iteration,graph,w_all)
        # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.01,5.05/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.02,0.00007/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_firm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        
        # extra = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,3.3/self.m,0.02,1.9/self.m,self.iteration,graph,w_all)
        # extra = self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.58/self.m,0.05,2.8/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.58/self.m,0.05,2.8/self.m,self.iteration,graph,w_all)


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

        # extra = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.041/self.m,1.3,0.00145,self.iteration,graph,w_all,0.009,1.1,1,1)


        # extra = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.085/self.m,0.085,0.001,self.iteration,graph,w_all)
        # self.pxg_extra_mc_convex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2/self.m,0.088,0.0001,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.141,0.034,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.061/self.m,0.1405,0.0016,self.iteration,graph,w_all)
        # self.pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.088,0.034,self.iteration,graph,w_all)
        # self.atc_pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2005/self.m,0.24,0.034,self.iteration,graph,w_all)

        plt.xlabel("Iterations")
        plt.ylabel("System Mismatch (dB)")
        plt.grid(which = "major")
        pdf.savefig()
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