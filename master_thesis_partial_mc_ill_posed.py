import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages





np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 10
        self.m = 9
        self.r_i = 4
        self.iteration = 20000
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
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 16 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        


        lambda_array = np.array([0.09,0.04,0.03,0.02,0.03,0.01,0.01,0.01,0.2])
        rho_array = np.array([0.7,0.7,0.4,1.0,0.01,1,1,1,0.01])
        # self.pg_extra_partial_mc_each_param(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lambda_array,0.027,rho_array,self.iteration,graph,w_all,0.05)
        
        for i in range(1000):
            print(i)
            w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
            w_l1_error, wl1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.04/self.m,1,0.09,self.iteration,graph,w_all,0,0.75,False,False)
            w_mc_error = self.prox_dgd(U_all, d_all, w_star, L2, self.N, self.m, self.r_i, 0.7/self.m, 1, 35/self.m, self.iteration, graph, w_all,False)
            w_amc_error, wamc = self.pg_extra_approximate_mc(U_all, d_all, w_star, L2, self.N, self.m, self.r_i, 0.04/self.m, 0.027, 10**-8/self.m, self.iteration, graph, w_all, 0.05,False)
            w_pmc_error, wpmc = self.pg_extra_partial_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.05/self.m,0.1,9/self.m,self.iteration,graph,w_all,0.75,False,False)
            if i == 0:
                w_l1_error_array = np.array(w_l1_error).reshape(1, self.iteration)
                w_mc_error_array = np.array(w_mc_error).reshape(1, self.iteration)
                w_amc_error_array = np.array(w_amc_error).reshape(1, self.iteration)
                w_pmc_error_array = np.array(w_pmc_error).reshape(1, self.iteration)
            else:
                w_l1_error_array = np.vstack((w_l1_error_array,np.array(w_l1_error).reshape(1, self.iteration)))
                w_mc_error_array = np.vstack((w_mc_error_array,np.array(w_mc_error).reshape(1, self.iteration)))
                w_amc_error_array = np.vstack((w_amc_error_array,np.array(w_amc_error).reshape(1, self.iteration)))
                if w_pmc_error != None:
                    w_pmc_error_array = np.vstack((w_pmc_error_array,np.array(w_pmc_error).reshape(1, self.iteration)))
        l1_average = w_l1_error_array.mean(axis = 0)
        mc_average = w_mc_error_array.mean(axis = 0)
        amc_average = w_amc_error_array.mean(axis = 0)
        pmc_average = w_pmc_error_array.mean(axis = 0)
        plt.plot(list(range(self.iteration)),l1_average, label='PG-EXTRA with ' + r"$\ell_1$"+' penalty')
        plt.plot(list(range(self.iteration)),mc_average, label='Prox DGD with MC penalty')
        plt.plot(list(range(self.iteration)),amc_average, label="PG-EXTRA with AMC penalty")
        plt.plot(list(range(self.iteration)),pmc_average, label="PG-EXTRA with Projective MC penalty")
        # best_error = 0
        # optimal_lambda = 0
        # optimal_rho = 0
        # for i in range(10):
        #     for j in range(10):
        #         lamb = 0.01 * i
        #         rho = (1+j) * 10 **(-9)
        #         error_pdgd, wamc = self.pg_extra_approximate_mc(U_all, d_all, w_star, L2, self.N, self.m, self.r_i, lamb/self.m, 0.027, rho/self.m, self.iteration, graph, w_all, 0.05)
        #         if error_pdgd[-1] < best_error:
        #             best_error = error_pdgd[-1]
        #             optimal_lambda = lamb
        #             optimal_rho = rho
        # print(optimal_lambda, optimal_rho, best_error)

        plt.xlabel("Iterations",fontsize=16)
        plt.ylabel("System Mismatch (dB)",fontsize=16)
        plt.grid(which = "major")
        plt.legend(fontsize=16)
        plt.savefig('master_thesis_pmc.pdf')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()