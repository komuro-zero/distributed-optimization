import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib



np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 10
        self.m = 60
        self.r_i = 3
        self.iteration = 5000
        self.sparsity_percentage = 0.3
        self.lamb = 0.31
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.000
        self.w_noise = 30
        self.normal_distribution = True
        self.w_zero = True

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
        # u_eig,u_vec = np.linalg.eig(U_all.T@U_all)
        # print(min(u_eig))
        # plt.rcParams['text.usetex'] = True 
        # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] 
        # plt.rcParams['font.family'] = 'sans-serif'
        # plt.rcParams['font.sans-serif'] = 'Helvetica'
        # plt.rcParams["font.family"] = "Times New Roman"
        # 
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['text.usetex'] = True 
        pdf = PdfPages('main performance comparison SSP.pdf')
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 10 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        print(matplotlib.get_cachedir())

        # error_mc = []
        # smallest_error = 0
        # optimal_lamb= 0
        # optimal_rho = 0
        # for i in range(100):
        #     for j in range(100):
        #         lamb = 0.01 + 0.01*i
        #         rho = 0.1 + 0.1*j
        #         error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,lamb,0.001,rho,self.iteration)
        #         if error_centralized_mc[-1] < smallest_error:
        #             optimal_lamb = lamb
        #             optimal_rho = rho
        #             smallest_error = error_centralized_mc[-1]
        #         print(optimal_lamb,optimal_rho)
        # print(optimal_lamb,optimal_rho)

        # error_mc = []
        # smallest_error = 0
        # optimal_lamb= 0
        # optimal_rho = 0
        # for i in range(100):
        #     for j in range(100):
        #         lamb = 0.01 + 0.01*i
        #         rho = 0.1 + 0.1*j
        #         error_approximate_mc,wcmc = self.centralized_mc_twin_with_amc(U_all,d_all,w,w_star,L2,lamb,0.02,rho,self.iteration,self.m)
        #         if error_approximate_mc[-1] < smallest_error:
        #             optimal_lamb = lamb
        #             optimal_rho = rho
        #             smallest_error = error_approximate_mc[-1]
        #         print(optimal_lamb,optimal_rho)
        # print(optimal_lamb,optimal_rho)


        # smallest_error = 0
        # optimal_lamb= 0
        # for i in range(100):
        #     lamb = 0.01 + 0.01*i
        #     error_centralized_L1,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,lamb,0.001,self.iteration)
        #     if error_centralized_L1[-1] < smallest_error:
        #         optimal_lamb = lamb
        #         smallest_error = error_centralized_L1[-1]
        #     print(optimal_lamb)
        # print(optimal_lamb)
        amc_list = []
        mc_list = []
        l1_list = []
        trials = 400
        I = np.eye(self.N)
        I_list = []
        i =0
        while i <trials:
            B = (3.4/0.99)**0.5
            w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
            if self.centralized_convexity_checker(B,0.99,U_all,len(w)):
                error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.99,0.001,3.4,self.iteration)
                # error_l1,wl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.26,10,self.iteration)

                # error_centralized_mc_twin,wcmc_twin = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.46,0.001,1.1,10000,self.m)
                error_extra,w_extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.02,self.rho,self.iteration,graph,w_all)
                error_l1,w_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1/self.m,0.02,0.09,self.iteration,graph,w_all,0,1)
                # self.pg_extra_mc_nonconvex_twin(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.4/self.m,0.02,17/self.m,self.iteration,graph,w_all)
                # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.4/self.m,0.02,17/self.m,self.iteration,graph,w_all)
                # error_amc,wamc  = self.pg_extra_approximate_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.59/self.m,0.04,1.5/self.m,self.iteration,graph,w_all,1)
                error_amc,wamc  = self.pg_extra_approximate_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1/self.m,0.04,10**-30/self.m,self.iteration,graph,w_all,1)
                amc_list.append(error_amc[-1])
                mc_list.append(error_centralized_mc[-1][0])
                l1_list.append(error_l1[-1])
                I_approximation = U_all.T@U_all/self.m
                I_list.append(np.linalg.norm(I-I_approximation))
                i+=1
        print("amc difference from wstar", sum(amc_list)/trials)
        print("mc difference from wstar", sum(mc_list)/trials)
        print("l1 difference",sum(l1_list)/trials)
        print("I approximation", sum(I_list)/trials)
        # I = np.eye(self.N)
        # I_approximation = U_all.T@U_all/self.m
        # print("w difference", np.linalg.norm(wcmc-wamc))
        # print("approximation validity", np.linalg.norm(I-I_approximation))
        # print("amc difference from wstar", error_amc[-1])
        # print("mc difference from wstar", error_centralized_mc[-1][0])
        # print("l1 difference",error_l1[-1])
        # print("extra error",error_extra[-1])
        # plt.xlabel("Iterations",fontsize=12)
        # plt.ylabel("System Mismatch (dB)",fontsize=12)
        # plt.grid(which = "major")
        # # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # # plt.axhline(y = error_centralized_mc_twin[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        # plt.legend(fontsize=12)
        # plt.savefig('SSP_system_mismatch_main_result.pdf')
        # plt.show()
        # x  = range(len(w_star))
        # plt.plot(x,extra,label = "extra")
        # plt.plot(x,wcl1,label = "L1")
        # plt.plot(x,wcmc,label = "mc")
        # plt.plot(x,amc_w,label = "amc")
        # plt.plot(x,w_star,color = "black")
        # plt.legend()
        # plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()