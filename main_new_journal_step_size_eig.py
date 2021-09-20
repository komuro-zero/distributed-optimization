import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages
from statistics import mean




np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 10
        self.m = 50
        self.r_i = 5
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
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
        u_eig,u_vec = np.linalg.eig(U_all.T@U_all)
        # m_eig,m_vec = np.linalg.eig(((graph + np.eye(len(graph)))/2))
        # print(min(m_eig))
        step=[]
        x=[]
        eig = []
        for i in range(2,self.m,1):
            prox_eig = []
            for j in range(100):
                I,c_1,c_2,c_tilde,eta=self.extra_setup(graph,i,U_all,1,1,1,1)
                m_eig,m_vec = np.linalg.eig(c_tilde)
                eig_val,eig_vec = np.linalg.eig((I-c_2).T@(I-c_2))
                list_val = list(eig_val)
                list_val.sort()
                prox_eig.append(list_val[1])
            x.append(i)
            eig.append(mean(prox_eig))
            if i == 2:
                second_smallest = list_val[1]
                step.append(2*min(m_eig)/(max(u_eig)+1/second_smallest))
            else:
                mu = 1/list_val[1]
                step.append(2*min(m_eig)/(max(u_eig)+mu))
            print(i)
        # step = step/step[-1]
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 10 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        plt.grid(which = "major")
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x,step,"C1",label = "step size")
        ax2 = ax1.twinx()
        ax2.plot(x,eig,"C0",label ="2nd smallest eigenvalue of "+ r"$VV^{\top}$")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='lower right')
        
        # pdf = PdfPages('main performance comparison journal.pdf')
        ax1.set_xlabel("Neighboring nodes",fontsize=12)
        ax1.set_ylabel("2nd smallest eigenvalue of "+ r"$VV^{\top}$")
        ax2.set_ylabel("Step size")
        ax1.grid(True)
        # # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # # plt.axhline(y = error_centralized_dual_soft[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        # plt.legend(fontsize=12)
        plt.savefig('journal_2nd_eigenvalue.pdf')
        # plt.show()
        
        # self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.01,self.iteration)
        # error_l1,wl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.19,0.01,self.iteration)
        # error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.9,0.01,2,self.iteration)
        # error_centralized_scad,wcmc = self.centralized_scad(U_all,d_all,w,w_star,L2,0.08,0.01,4.9,self.iteration)
        # self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.19/self.m,0.027,0.09,self.iteration,graph,w_all)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,0,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,1,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,20,1)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,0,0.15)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,wcmc,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,10,0.08)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,0,0.15)
        # self.pg_extra_mc_consensus_violation(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.9/self.m,1,2/self.m,self.iteration,graph,w_all,10,0.08)
        # self.pg_extra_scad(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.08/self.m,0.027,4.9/self.m,self.iteration,graph,w_all)
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
        #         lamb = 0.1 + 0.1*i
        #         rho =  0.1 + 0.1*j
        #         error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,lamb,0.01,rho,self.iteration)
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


        # error_centralized_mc,wcmc = self.centralized_scad(U_all,d_all,w,w_star,L2,0.04,0.01,2,self.iteration)
        # error_centralized_dual_soft,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,2.2,0.01,0.09*self.m,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,2,0.01,8,self.iteration,self.m)
        # error,wcmc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,2,0.01,5.05,self.iteration,self.m)

        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.06,0.0001,0.03,self.iteration,self.m)

        # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.008,5.05/self.m,self.iteration,graph,w_all)
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

        # extra = self.NEXT_with_MC_penalty(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.041/self.m,1.3,0.00145,self.iteration,graph,w_all,0.009,1.1,1,1)


        # extra = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.085/self.m,0.085,0.001,self.iteration,graph,w_all)
        # self.pxg_extra_mc_convex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2/self.m,0.088,0.0001,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.22/self.m,0.141,0.034,self.iteration,graph,w_all)
        # self.distributed_proximal_gradient_algorithm_approximate_MC(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.061/self.m,0.1405,0.0016,self.iteration,graph,w_all)
        # self.atc_pg_extra_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.2005/self.m,0.24,0.034,self.iteration,graph,w_all)

        # plt.xlabel("Iterations",fontsize=12)
        # plt.ylabel("Second Smallest Eigenvalue/ Step Size (dB)",fontsize=12)
        # plt.grid(which = "major")
        # # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # # plt.axhline(y = error_centralized_dual_soft[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # # pdf.savefig()
        # plt.legend(fontsize=12)
        # plt.savefig('journal_global_convergence_3.pdf')
        # plt.show()
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