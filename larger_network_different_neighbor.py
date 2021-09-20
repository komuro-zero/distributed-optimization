import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns



np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 30
        self.m = 100
        self.r_i = 2
        self.iteration = 10000
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
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_3(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
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
        # eigval,eigvec = np.linalg.eig(U_all.T@U_all)
        # print(min(eigval))
        # self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.007,self.iteration)
        error_l1_list,wcmc = self.centralized_L1(U_all,d_all,w,w_star,L2,1.4,0.007,self.iteration)
        l1_error = error_l1_list[-1]

        
        # l1 search
        # optimal = 1
        # optimal_val = 1
        # for i in range(100):
        #     error,wcmc = self.centralized_L1(U_all,d_all,w,w_star,L2,0.1*i,0.007,self.iteration)
        #     if error[-1] < optimal:
        #         optimal = error[-1]
        #         optimal_val = 0.1*i
        # print(optimal_val)

        # mc search
        # error = 1
        # lamb_search = 1
        # rho_search = 1
        # grid = np.zeros((99,299))
        # for i in range(1,100):
        #     for j in range(1,300):
        #         error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,i*0.1,0.007,j*0.1,self.iteration)
        #         if not error_centralized_mc:
        #             grid[i-1][j-1] = None
        #         else:
        #             grid[i-1][j-1] = error_centralized_mc[-1]
        #             if error_centralized_mc[-1] < error:
        #                 error = error_centralized_mc[-1]
        #                 lamb_search = i*0.1
        #                 rho_search = j*0.1
        # plt.figure()
        # sns.heatmap(grid)
        # plt.show()
        # plt.savefig('seaborn_heatmap.png')
        # print(lamb_search,rho_search)
                
        # error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,5.6,0.007,26.7,self.iteration)

        # error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,1.4,0.007,0.0038*self.m,self.iteration)
        # error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,1.4,0.007,0.0038*self.m,self.iteration)
        # error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,1.4,0.007,0.0038*self.m,self.iteration)
        
        # x = []
        # v_2 = []
        # best_error = []
        # all_lambda = []
        # mu = 20
        # for i in range(2,98,2):
        #     graph = self.ring_graph(self.m,i)
        #     v_2.append(self.extra_setup_for_lambda_2_search(graph,i,mu))
        #     x.append(i)
        #     error = 1
        #     lambd = 1
        #     for j in range(1,56):
        #         error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,j*0.1,0.007,v_2[-1],self.iteration)
        #         if error_centralized_mc != None:
        #             if error_centralized_mc[-1] < error:
        #                 lambd = j*0.1
        #                 error = error_centralized_mc[-1]
        #     best_error.append(error)
        #     all_lambda.append(lambd)
        # plt.plot(x,v_2,label = "eta tau")
        # plt.plot(x,best_error, label = "error")
        # plt.plot(x,all_lambda, label = "lamdba")
        # plt.hlines(l1_error,0,100)
        # plt.legend()
        # plt.show()
        
        
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.06,0.0001,0.03,self.iteration,self.m)
        
        # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.008,5.05/self.m,self.iteration,graph,w_all)
        # extra = self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.02,0.00007/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_firm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        
        # self.pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,5.6/self.m,0.01,26.7/self.m,self.iteration,graph,w_all)
        # self.pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,3/self.m,0.02,0.11*self.m,self.iteration,graph,w_all)
        # self.pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,3/self.m,0.03,0.11*self.m,self.iteration,graph,w_all)

        # self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.4/self.m,0.01,0.09,self.iteration,graph,w_all)
        # self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all)
        self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,5.6/self.m,0.005,26.7,self.iteration,graph,w_all,1,1)
        self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,5.6/self.m,0.005,26.7,self.iteration,graph,w_all,0,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.4/self.m,0.01,0.11,self.iteration,graph,w_all,10,1)
        
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.4/self.m,1,0.0038,self.iteration,graph,w_all,1000,1)
        
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,1,0.09,self.iteration,graph,w_all,10,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,1,0.09,self.iteration,graph,w_all,1,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,1,0.09,self.iteration,graph,w_all,1,0.1)

        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,7)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,7)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,1)

        
        # self.pg_extra_scad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.12/self.m,0.02,2.51,self.iteration,graph,w_all)
        # self.pg_extra_scad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.11/self.m,0.02,2.51,self.iteration,graph,w_all)
        # self.pg_extra_scad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1/self.m,0.02,2.51,self.iteration,graph,w_all)
        # self.pg_extra_scad_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1/self.m,0.02,5,self.iteration,graph,w_all)
        
        
        # extra = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,3.3/self.m,0.02,1.9/self.m,self.iteration,graph,w_all)
        # extra = self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.58/self.m,0.05,2.8/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.58/self.m,0.05,2.8/self.m,self.iteration,graph,w_all)


        
        plt.xlabel("Iterations",fontsize=12)
        plt.ylabel("System Mismatch (dB)",fontsize=12)
        plt.grid(which = "major")
        # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # plt.axhline(y = error_centralized_dual_soft[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        plt.legend(fontsize=12)
        # plt.savefig('main performance comparison journal_2.pdf')
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