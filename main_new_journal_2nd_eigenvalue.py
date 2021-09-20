import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages
import copy





np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 10
        self.m = 50
        self.r_i = 3
        self.iteration = 1000
        self.sparsity_percentage = 0.2
        self.lamb = 0.31    
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.001
        self.w_noise = 30
        self.normal_distribution = True
        self.w_zero = True

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
        plt.rcParams["font.family"] = "Times New Roman" 
        plt.rcParams['text.usetex'] = True 
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 16 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        # pdf = PdfPages('journal_convergence_speed_second_eigenvalue.pdf')
        
        I = np.eye(self.m)
        Laplacian = self.r_i*I - (graph - I)
        eig = []
        smallest_eig = 100
        biggest_eig = 0
        for i in range(1000):
            graph_flag = True
            while graph_flag:
                graph = self.undirected_graph(self.m,self.r_i)
                graph_flag = self.disjoint_checker(graph,self.m)
            g_eig,g_vec = np.linalg.eig(I-(1/self.r_i)*(graph -I))
            g_eig_sort = sorted(list(g_eig))
            second_eig = (g_eig_sort[1]).real
            eig.append(second_eig)
            if second_eig < smallest_eig:
                smallest_eig =copy.deepcopy(second_eig)
                smallest_graph = copy.deepcopy(graph)
                print(i)
            if second_eig > biggest_eig:
                biggest_eig = copy.deepcopy(second_eig)
                biggest_graph = copy.deepcopy(graph)
                print(i)
        # plt.figure()
        # plt.hist(eig)
        # plt.figure()
        # self.show_graph(smallest_graph,self.r_i)
        # plt.figure()
        # self.show_graph(biggest_graph,self.r_i)
        self.pg_extra_mc_consensus_violation_fixed_tau_eta(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,smallest_graph,w_all,0,0.07,smallest_eig)
        self.pg_extra_mc_consensus_violation_fixed_tau_eta(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,smallest_graph,w_all,1,0.07,smallest_eig)
        self.pg_extra_mc_consensus_violation_fixed_tau_eta(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,biggest_graph,w_all,0,0.07,biggest_eig)
        self.pg_extra_mc_consensus_violation_fixed_tau_eta(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,biggest_graph,w_all,1,0.07,biggest_eig)
        plt.xlabel("Iterations",fontsize=16)
        plt.ylabel("System Mismatch (dB)",fontsize=16)
        plt.grid(which = "major")
        # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # plt.axhline(y
        #  = error_centralized_dual_soft[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        plt.legend(fontsize=16,loc = "best")
        plt.savefig('journal_convergence_speed_second_eigenvalue.pdf')
        plt.show()
        print(smallest_eig,biggest_eig)

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()