import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages





np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 10
        self.m = 100
        self.r_i = 5
        self.iteration = 2000
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
        # u_eig,u_vec = np.linalg.eig(U_all.T@U_all)
        # m_eig,m_vec = np.linalg.eig(((graph + np.eye(len(graph)))/2))
        # print(min(m_eig))
        # step=[]
        # x=[]
        # eig = []
        # for i in range(2,self.m,1):
        #     I,c_1,c_2,c_tilde,eta=self.extra_setup(graph,i,U_all,1,1,1,1)
        #     m_eig,m_vec = np.linalg.eig(c_tilde)
        #     x.append(i)
        #     eig_val,eig_vec = np.linalg.eig((I-c_2).T@(I-c_2))
        #     list_val = list(eig_val)
        #     list_val.sort()
        #     eig.append(list_val[1])
        #     if i == 2:
        #         second_smallest = list_val[1]
        #         step.append(2*min(m_eig)/(max(u_eig)+1/second_smallest))
        #     else:
        #         mu = 1/list_val[1]
        #         step.append(2*min(m_eig)/(max(u_eig)+mu))
        # step = step/step[-1]
        # plt.plot(x,eig,label = "2nd smallest eigenvalue")
        # # plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        # plt.xlabel("Neighboring nodes",fontsize=12)
        # plt.ylabel("2nd smallest eigenvalue/ step size",fontsize=12)
        # plt.plot(x,step,label = "step size")
        # # plt.ylabel("Magnitude",fontsize=12)
        # plt.grid(which = "major")
        # plt.legend(fontsize=12)
        # plt.savefig('journal_2nd_eigenvalue.pdf')
        # plt.show()

        # plt.rcParams['text.usetex'] = True 
        # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] 
        # plt.rcParams['font.family'] = 'sans-serif'
        # plt.rcParams['font.sans-serif'] = 'Helvetica'
        # plt.rcParams["font.family"] = "Times New Roman" 
        # pdf = PdfPages('main performance comparison journal.pdf')
        # # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # # plt.axhline(y = error_centralized_dual_soft[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        # self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.01,self.iteration)
        # error_l1,wl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.19,0.01,self.iteration)
        # error_centralized_mc,wcmc = self.centralized_partial_mc(U_all,d_all,w,w_star,L2,0.11,0.01,7.1,self.iteration,self.m)
        # error_centralized_scad,wcmc = self.centralized_scad(U_all,d_all,w,w_star,L2,0.08,0.01,4.9,self.iteration)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,0,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,1,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,20,1)
        
        error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.9,0.01,44,10000)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,0.9/self.m,4,44/self.m,2500,graph,w_all,0,0.08)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,0.9/self.m,1,44/self.m,10000,graph,w_all,0,0.3)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,0.9/self.m,4,44/self.m,2500,graph,w_all,0,0.15)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,0.9/self.m,2,44/self.m,2500,graph,w_all,1,0.12)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,0.9/self.m,2,44/self.m,2500,graph,w_all,10,0.08)
        
        # self.pg_extra_partial_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.66/self.m,0.027,44/self.m,self.iteration,graph,w_all,0.05)
        # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.66/self.m,0.01,44/self.m,self.iteration,graph,w_all)
        # self.pg_extra_approximate_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.45/self.m,0.027,1.2/self.m,self.iteration,graph,w_all,0.05)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.66/self.m,1,44/self.m,self.iteration,graph,w_all,0,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.66/self.m,1,44/self.m,self.iteration,graph,w_all,10,0.03)
        # self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.25/self.m,0.027,0.09,self.iteration,graph,w_all,0,0.75)
        
        # print(U_all.T@U_all)
        # thresh = 0
        # for i in range(50):
        #     for j in range(50):
        #         lamb = 0.01 + 0.1*i
        #         rho =  0.01 + 0.01*j
        #         error,w = self.pg_extra_approximate_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb/self.m,0.027,rho/self.m,self.iteration,graph,w_all)
        #         if error != None:
        #             if error[-1] < thresh:
        #                 thresh = error[-1]
        #                 optimal_lamb = lamb
        #                 optimal_rho = rho
        #         print(i,j)
        # print(optimal_lamb,optimal_rho)

        thresh = 0
        for i in range(100):
            for j in range(100):
                lamb = 0.01 + 0.1*i
                rho =  0.01 + 0.01*j
                error,w = self.pg_extra_partial_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb/self.m,0.027,rho/self.m,self.iteration,graph,w_all,1)
                if error != None:
                    if error[-1] < thresh:
                        thresh = error[-1]
                        optimal_lamb = lamb
                        optimal_rho = rho
                print(i,j)
        print(optimal_lamb,optimal_rho)

        # thresh = 0
        # for i in range(50):
        #     for j in range(50):
        #         lamb = 0.01 + 0.1*i
        #         rho =  0.01 + 0.01*j
        #         error = self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb/self.m,0.02,rho/self.m,self.iteration,graph,w_all)
        #         if error != None:
        #             if error[-1] < thresh:
        #                 thresh = error[-1]
        #                 optimal_lamb = lamb
        #                 optimal_rho = rho
        #         print(i,j)
        # print(optimal_lamb,optimal_rho)

        # self.pg_extra_scad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.08/self.m,0.027,4.9,self.iteration,graph,w_all)
        # smallest_error = 0
        # optimal_beta = 0
        # optimal_tau = 0
        # optimal_inner_eta = 0
        # for tau in range(10):
        #     for inner_eta in range(10):
        #         extra,error_next = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,0.001,2/self.m,self.iteration,graph,w_all,0.01+0.01*inner_eta,0.1+0.1*tau,0.5,1)
        #         if error_next[-1] < smallest_error:
        #             smallest_error = error_next[-1]
        #             optimal_inner_eta = 0.01+0.01*inner_eta
        #             optimal_tau = 0.1+0.1*tau
        # print(optimal_tau,optimal_inner_eta)
        # smallest_error = 0
        # for i in range(10):
        #     for j in range(10):
        #         lamb = 0.1+0.1*i
        #         rho = 0.1 + 0.1*j
        #         extra,error = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb/self.m,0.007,rho/self.m,self.iteration,graph,w_all,0.1,0.5,1)
        #         if smallest_error > error[-1]:
        #             smallest_error = error[-1]
        #             optimal_lamb = lamb
        #             optimal_rho = rho
        # print(optimal_lamb,optimal_rho)
        # extra = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,0.005,2/self.m,self.iteration,graph,w_all,0.1,0.5,1)
        # extra = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,0.1,2/self.m,self.iteration,graph,w_all,10,10)
        # next_w,error = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,0.01,2/self.m,self.iteration,graph,w_all,0.1,1)
        
        
        # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,0.006,2/self.m,self.iteration,graph,w_all)
        # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,0.005,2/self.m,self.iteration,graph,w_all)
        
        
        # self.pg_extra_partial_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1/self.m,1,0.1/self.m,self.iteration,graph,w_all)

        # for i in range(10):
        #     for j in range(10):
        #         lamb = 0.1+0.1*i
        #         rho = 0.1+0.1*j
        #         self.pg_extra_partial_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb/self.m,1,rho/self.m,self.iteration,graph,w_all)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,1,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,20,0.07)
        # thresh = 0
        # for i in range(100):
        #     lamb = 0.01 + 0.01*i
        #     error_l1,wl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,lamb,0.01,self.iteration)
        #     if error_l1[-1] < thresh:
        #         thresh = error_l1[-1]
        #         optimal_lamb = lamb
        # print(optimal_lamb)
        # thresh = 0
        # for i in range(100):
        #     for j in range(100):
        #         lamb = 0.01 + 0.01*i
        #         rho =  1 + 1*j
        #         error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,lamb,0.01,rho,300)
        #         if error_centralized_mc != None:
        #             if error_centralized_mc[-1] < thresh:
        #                 thresh = error_centralized_mc[-1]
        #                 optimal_lamb = lamb
        #                 optimal_rho = rho
        #         print(i,j,optimal_lamb,optimal_rho)
        # print(optimal_lamb,optimal_rho)
        # error_centralized_smc,wcsmc = self.centralized_mc_twin_with_amc(U_all,d_all,w,w_star,L2,0.45,0.001,1.2,300,self.m)

        # error_mc = []
        # smallest_error = 0
        # optimal_lamb= 0
        # optimal_rho = 0
        # for i in range(100):
        #     for j in range(100):
        #         lamb = 0.01 + 0.01*i
        #         rho = 0.1 + 0.1*j
        #         error_centralized_smc,wcsmc = self.centralized_mc_twin_with_amc(U_all,d_all,w,w_star,L2,lamb,0.001,rho,300,self.m)
        #         if error_centralized_smc[-1] < smallest_error:
        #             optimal_lamb = lamb
        #             optimal_rho = rho
        #             smallest_error = error_centralized_smc[-1]
        #         print(i,j,optimal_lamb,optimal_rho)
        # print(optimal_lamb,optimal_rho)
        # thresh = 0
        # for i in range(100):
        #     for j in range(100):
        #         lamb = 0.01 + 0.01*i
        #         rho =  0.1 + 0.1*j
        #         error_centralized_mc,wcmc = self.centralized_partial_mc(U_all,d_all,w,w_star,L2,lamb,0.01,rho,self.iteration,self.m)
        #         if error_centralized_mc != None:
        #             if error_centralized_mc[-1] < thresh:
        #                 thresh = error_centralized_mc[-1]
        #                 optimal_lamb = lamb
        #                 optimal_rho = rho
        #         print(i,j)
        # print(optimal_lamb,optimal_rho)

        # thresh = 0
        # for i in range(100):
        #     for j in range(100):
        #         lamb = 0.01 + 0.01*i
        #         rho =  0.1 + 0.1*j
        #         error_scad,w_scad = self.centralized_scad(U_all,d_all,w,w_star,L2,lamb,0.01,rho,self.iteration)
        #         if error_scad != None:
        #             if error_scad[-1] < thresh:
        #                 thresh = error_scad[-1]
        #                 optimal_lamb = lamb
        #                 optimal_rho = rho
        #         print(i,j)
        # print(optimal_lamb,optimal_rho)

        # error_centralized_mc,wcmc = self.centralized_scad(U_all,d_all,w,w_star,L2,lamb,0.01,rho,self.iteration)

        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,4.6/self.m,1,8.7/self.m,self.iteration,graph,w_all,2,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,4.6/self.m,1,8.7/self.m,self.iteration,graph,w_all,5,1)
        # self.pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.5/self.m,0.02,8/self.m,self.iteration,graph,w_all)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.5/self.m,1,8/self.m,self.iteration,graph,w_all,0.1,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.5/self.m,1,8/self.m,self.iteration,graph,w_all,0.01,1)
        # self.pg_extra_scad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.036,0.02,0.001,self.iteration,graph,w_all)


        # error_centralized_mc,wcmc = self.centralized_scad(U_all,d_all,w,w_star,L2,0.04,0.01,2,self.iteration)
        # error_centralized_dual_soft,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,2.2,0.01,0.09*self.m,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,2,0.01,8,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,2,0.01,5.05,self.iteration,self.m)

        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.06,0.0001,0.03,self.iteration,self.m)

        # extra = self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.02,0.00007/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_firm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.035,5.05/self.m,self.iteration,graph,w_all)
        


        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,1,0.09,self.iteration,graph,w_all,10,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,1,0.09,self.iteration,graph,w_all,1,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,1,0.09,self.iteration,graph,w_all,1,0.1)

        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,7)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,7)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,0.1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.2/self.m,0.02,0.09,self.iteration,graph,w_all,1)

        
        # self.pg_extra_scad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.11/self.m,0.02,2.51,self.iteration,graph,w_all)
        # self.pg_extra_scad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1/self.m,0.02,2.51,self.iteration,graph,w_all)
        # self.pg_extra_scad_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1/self.m,0.02,5,self.iteration,graph,w_all)
        
        
        # extra = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,3.3/self.m,0.02,1.9/self.m,self.iteration,graph,w_all)
        # extra = self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.58/self.m,0.05,2.8/self.m,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2.58/self.m,0.05,2.8/self.m,self.iteration,graph,w_all)


        # self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.023,1,self.iteration,graph,w_all)
        # extra = self.dista(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.01,1,self.iteration,graph,w_all,0.5)
        # extra = self.FDDA(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*2.5,0.019,1,self.iteration,graph,w_all)
        # extra = self.FDDA(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*2.5,0.02,1,self.iteration,graph,w_all)
        # extra = self.FDDA(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*2.5,0.023,1,self.iteration,graph,w_all)
        # extra = self.dfbbs(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.023,1,self.iteration,graph,w_all)
        # extra = self.pad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.007,0.01,0.07,self.iteration,graph,w_all,0.08)
        # self.distributed_proximal_gradient_algorithm(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.2/self.m,0.036,0.01,self.iteration,graph,w_all)

        # extra = self.APM(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0000001,0.2,1,self.iteration,graph,w_all)
        # extra = self.ADDA(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.54,1,self.iteration,graph,w_all)
        # self.pg_extra_l1_projection(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.095,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1_projection(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,0.255,1,self.iteration,graph,w_all)
        # self.atc_pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.056/self.m,0.25,1,self.iteration,graph,w_all)
        # self.pg_extra_l1_step_size(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,L2*1.95,1.305,1,self.iteration,graph,w_all)



        # extra = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.085/self.m,0.085,0.001,self.iteration,graph,w_all)
        # self.pxg_extra_mc_convex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2/self.m,0.088,0.0001,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.141,0.034,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.061/self.m,0.1405,0.0016,self.iteration,graph,w_all)
        # self.atc_pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2005/self.m,0.24,0.034,self.iteration,graph,w_all)

        plt.xlabel("Iterations",fontsize=16)
        # plt.ylabel("System Mismatch (dB)",fontsize=16)
        plt.ylabel("Distance to optimal vector (dB)",fontsize=16)
        # plt.ylabel("Disagreement (dB)",fontsize=12)
        plt.grid(which = "major")
        # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # plt.axhline(y = thresh,linestyle="dashed",label = "Approximate MC penalty",color = "blue")
        # plt.axhline(y = error_centralized_dual_soft[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        # plt.legend(fontsize=16,bbox_to_anchor=(1.05, 0.5, 0.5, .100), borderaxespad=0.,ncol=1,mode="expand")
        plt.legend(fontsize=16)
        plt.savefig('journal_system_mismatch_with_l1_and_amc.pdf')
        # plt.savefig('journal_disagreement_neighbor_size.pdf')
        plt.tight_layout()
        # plt.savefig('journal_global_convergence_3.pdf')
        # plt.subplots_adjust(left = 0.1)
        plt.show()
        # x  = range(len(w_star))
        # plt.plot(x,next_w,label = "next")
        # # plt.plot(x,extra,label = "extra")
        # # plt.plot(x,wcl1,label = "L1")
        # # plt.plot(x,wcmc,label = "mc")
        # plt.plot(x,w_star,color = "black")
        # plt.legend()
        # plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()