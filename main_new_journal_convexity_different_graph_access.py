import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages



np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 10
        self.m = 50
        self.r_i = 2
        self.iteration = 800
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
        plt.rcParams["font.family"] = "Times New Roman" 
        plt.rcParams['text.usetex'] = True 
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
        plt.rcParams['font.size'] = 16 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        
        trial_times = 1000
        min_eigenvalues = []
        bad_V = []
        good_V = []
        w,w_star,w_all,U_all,d_all,L2,graph_not = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)    

        for i in range(trial_times):
            w,w_star,w_all,U_not,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)    
            I,c_1,c_2,c_tilde,eta = self.extra_setup(graph,self.r_i,U_all,1,1,self.rho,1)
            V = I - c_2
            u_eig,u_vec = np.linalg.eig(U_all@U_all.T+V@V.T)
            min_eigenvalues.append(min(u_eig))
            v_eig,v_vec =np.linalg.eig(V@V.T)
            
            # if min(u_eig) < 0:
            #     bad_V.append(v_eig[0])
            # else:
            #     good_V.append(v_eig[0])
        print(set(min_eigenvalues))
        plt.hist(min_eigenvalues)


        plt.xlabel("Smallest Eigenvalue",fontsize=16)
        plt.ylabel("Frequency",fontsize=16)
        plt.grid(which = "major")
        # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # plt.axhline(y = error_centralized_dual_soft[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        # plt.legend(fontsize=12)
        plt.savefig('journal_convexity_with_CPP_static_u_SIP.pdf')
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