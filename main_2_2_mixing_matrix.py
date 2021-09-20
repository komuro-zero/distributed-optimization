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
        self.iteration = 3000
        self.sparsity_percentage = 0.3
        self.lamb = 0.31
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0
        self.w_noise = 10
        self.normal_distribution = True
        self.w_zero = True

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
        # plt.rcParams['text.usetex'] = True 
        # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] 
        # plt.rcParams['font.family'] = 'sans-serif'
        # plt.rcParams['font.sans-serif'] = 'Helvetica'
        # plt.rcParams["font.family"] = "Times New Roman" 
        pdf = PdfPages('main performance comparison journal.pdf')
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 10 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        # self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.01,self.iteration)
        # error,wcmc = self.centralized_L1(U_all,d_all,w,w_star,L2,1.2,0.01,self.iteration)
        
        error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,2.2,0.01,0.09*self.m,self.iteration)
        # error_centralized_mc,wcmc = self.centralized_scad(U_all,d_all,w,w_star,L2,0.04,0.01,2,self.iteration)
        # error_centralized_dual_soft,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,2.2,0.01,0.09*self.m,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,2,0.01,8,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,2,0.01,5.05,self.iteration,self.m)
        
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.06,0.0001,0.03,self.iteration,self.m)
        


        # self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.02,0.09,self.iteration,graph,w_all)
        # self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.03,0.09,self.iteration,graph,w_all,0,0.5)
        # self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.035,0.09,self.iteration,graph,w_all,0,0.1)
        # self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.035,0.09,self.iteration,graph,w_all,0,0.1)
        # self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.035,0.09,self.iteration,graph,w_all,0.1,0.1)
        self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.025,0.09,self.iteration,graph,w_all,20,0.1)
        self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.025,0.09,self.iteration,graph,w_all,0,0.1)
        # self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.029,0.09,self.iteration,graph,w_all,0,0.1)
        # self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.035,0.09,self.iteration,graph,w_all,0,0.1)
        # self.pg_extra_mixing_matrix_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.035,0.09,self.iteration,graph,w_all,0,0.1)


        # self.pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all)
        # self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,7)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,7)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,1)
        

        plt.xlabel("Iterations",fontsize=12)
        plt.ylabel("System Mismatch (dB)",fontsize=12)
        plt.grid(which = "major")
        plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # plt.axhline(y = error_centralized_dual_soft[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        pdf.savefig()
        plt.legend(fontsize=12)
        plt.savefig('main performance comparison journal_2.pdf')
        plt.show()
        # x  = range(len(w_star))
        # plt.plot(x,extra,label = "extra")
        # plt.plot(x,wcl1,label = "L1")
        # plt.plot(x,wcmc,label = "mc")
        # plt.plot(x,w_star,color = "black")
        # plt.legend()
        # plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()