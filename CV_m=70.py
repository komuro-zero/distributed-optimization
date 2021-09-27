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
        self.m = 70
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

        
        w_star_all =np.loadtxt(f"./samples/sample_wstar_100_m={self.m}")
        U_all_all =np.loadtxt(f"./samples/sample_U_all_100_m={self.m}")
        d_all_all =np.loadtxt(f"./samples/sample_d_all_100_m={self.m}")
        graph_all =np.loadtxt(f"./samples/sample_graph_100_m={self.m}")
        
        data_number = 100
        train_number = 90
        test_number = data_number - train_number
        batch_number = 10
        list_prox = list(range(100))
        mc_test_error = []
        amc_test_error = []
        l1_test_error = []
        # for i in range(10):
        #     testing_list = list_prox[i*(self.m):(i+1)*(self.m)]
        #     first_list = list_prox[0:i*(self.m)]
        #     second_list= list_prox[(i+1)*self.m:100]
        #     all_list = first_list + second_list
        #     print(testing_list)
        #     print(all_list)
        # print(len(w_star_all[:,99]))
        for i in range(batch_number):
            train_w_star_1 = w_star_all[:,0:i*(test_number)]
            train_w_star_2 = w_star_all[:,(i+1)*test_number:100]
            train_w_star_list = np.append(train_w_star_1,train_w_star_2)
            test_w_star_list = w_star_all[:,i*(test_number):(i+1)*test_number]
            train_d_all_1 = d_all_all[:,0:i*(test_number)]
            train_d_all_2 = d_all_all[:,(i+1)*test_number:100]
            train_d_all_list = np.append(train_d_all_1 , train_d_all_2)
            test_d_all_list = d_all_all[:,i*(test_number):(i+1)*test_number]
            train_U_all_1 = U_all_all[:,0:i*(test_number)*self.m]
            train_U_all_2 = U_all_all[:,(i+1)*test_number*self.m:100*self.m]
            train_U_all_list = np.append(train_U_all_1 , train_U_all_2)
            test_U_all_list = U_all_all[:,i*(test_number)*self.m:(i+1)*test_number*self.m]
            train_graph_1 = graph_all[:,0:i*(test_number)*self.m]
            train_graph_2 = graph_all[:,(i+1)*test_number*self.m:100*self.m]
            train_graph_list = np.append(train_graph_1 , train_graph_2)
            test_graph_list = graph_all[:,i*(test_number)*self.m:(i+1)*test_number*self.m]
            mc_error_best = 0
            amc_error_best = 0
            l1_error_best = 0
            mc_optimal_lambda = 0
            mc_optimal_rho = 0
            amc_optimal_lambda = 0
            amc_optimal_rho = 0
            l1_optimal_lambda = 0
            for delta_lambda in range(100):
                for delta_rho in range(100):
                    mc_lamb = (delta_lambda+1)*0.01
                    mc_rho = (delta_rho+1)*0.1
                    amc_lamb = (delta_lambda+1)*0.01
                    amc_rho = (delta_rho+1)*0.1
                    l1_lamb = (delta_lambda+1)*0.01
                    mc_error_one_train = []
                    amc_error_one_train = []
                    l1_error_one_train = []
                    for train in range(train_number):
                        one_train_w_star = train_w_star_list[train*self.N:(train+1)*self.N]
                        one_train_w_star = one_train_w_star.reshape(self.N,1)
                        one_train_d_all = train_d_all_list[train*self.m:(train+1)*self.m]
                        one_train_d_all = one_train_d_all.reshape(self.m,1)
                        one_train_U_all = train_U_all_list[train*self.m*self.N:(train+1)*self.m*self.N]
                        one_train_U_all = one_train_U_all.reshape(self.m,self.N)
                        one_train_graph = train_graph_list[train*self.m*self.m:(train+1)*self.m*self.m]
                        one_train_graph = one_train_graph.reshape(self.m,self.m)
                        L2 = np.linalg.norm(one_train_w_star)
                        
                        error_centralized_mc,wcmc = self.centralized_mc(one_train_U_all,one_train_d_all,w,one_train_w_star,L2,0.039,0.001,0.0013,self.iteration)
                        error_l1,w_l1=self.pg_extra_l1(one_train_U_all,one_train_d_all,one_train_w_star,L2,self.N,self.m,self.r_i,0.039/self.m,0.02,0.09,self.iteration,one_train_graph,w_all,0,1)
                        error_amc,wamc = self.pg_extra_approximate_mc(one_train_U_all,one_train_d_all,one_train_w_star,L2,self.N,self.m,self.r_i,0.041/self.m,0.04,0.087/self.m,self.iteration,one_train_graph,w_all,1)
                        if error_centralized_mc[-1] < mc_error_best:
                            mc_error_best = error_centralized_mc[-1]
                            mc_optimal_lambda = mc_lamb
                            mc_optimal_rho = mc_rho
                        if error_amc[-1] < amc_error_best:
                            amc_error_best = error_amc[-1]
                            amc_optimal_lambda = amc_lamb
                            amc_optimal_rho = amc_rho
                        if error_l1[-1] < l1_error_best:
                            l1_error_best = error_l1[-1]
                            l1_optimal_lambda = l1_lamb
                        print("batch number : ",i,"\n train number : ",train ,"\n delta lambda : ",delta_lambda,"\n delta rho : ",delta_rho)
            for test in range(test_number):
                print("batch number : ",i,"\n test number : ",test , "\n delta lambda : ",delta_lambda,"\n delta rho : ",delta_rho)

                one_test_w_star = test_w_star_list[test*self.N:(test+1)*self.N]
                one_test_w_star = one_test_w_star.reshape(self.N,1)
                one_test_d_all = test_d_all_list[test*self.m:(test+1)*self.m]
                one_test_d_all = one_test_d_all.reshape(self.m,1)
                one_test_U_all = test_U_all_list[test*self.m*self.N:(test+1)*self.m*self.N]
                one_test_U_all = one_test_U_all.reshape(self.m,self.N)
                one_test_graph = test_graph_list[test*self.m*self.m:(test+1)*self.m*self.m]
                one_test_graph = one_test_graph.reshape(self.m,self.N)
                L2 = np.linalg.norm(one_train_w_star)


                error_centralized_mc,wcmc = self.centralized_mc(one_test_U_all,one_test_d_all,w,one_test_w_star,L2,mc_optimal_lambda,0.001,mc_optimal_rho,self.iteration)
                error_l1,w_l1=self.pg_extra_l1(one_test_U_all,one_test_d_all,one_test_w_star,L2,self.N,self.m,self.r_i,l1_optimal_lambda/self.m,0.02,0.09,self.iteration,one_test_graph,w_all,0,1)
                error_amc,wamc = self.pg_extra_approximate_mc(one_test_U_all,one_test_d_all,one_test_w_star,L2,self.N,self.m,self.r_i,amc_optimal_lambda/self.m,0.04,amc_optimal_rho/self.m,self.iteration,one_test_graph,w_all,1)
                mc_test_error.append(error_centralized_mc[-1])
                amc_test_error.append(error_amc[-1])
                l1_test_error.append(error_l1[-1])

        print(sum(mc_test_error)/len(mc_test_error))                
        print(sum(amc_test_error)/len(amc_test_error))                
        print(sum(l1_test_error)/len(l1_test_error))                
        # amc_list = []
        # mc_list = []
        # l1_list = []
        # trials = 400
        # i =0
        # I_list = []
        # I = np.eye(self.N)
        # while i <trials:
        #     B = (0.0013/0.039)**0.5
        #     w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
        #     if self.centralized_convexity_checker(B,0.039,U_all,len(w)):
        #         # w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
        #         error_centralized_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.039,0.001,0.0013,self.iteration)
        #         # error_l1,wl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.039,10,self.iteration)

        #         # error_centralized_mc_twin,wcmc_twin = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.02,0.001,0.03,10000,self.m)
        #         # error_extra,w_extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.02,self.rho,self.iteration,graph,w_all)
        #         error_l1,w_l1=self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.039/self.m,0.02,0.09,self.iteration,graph,w_all,0,1)
        #         # self.pg_extra_mc_nonconvex_twin(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.4/self.m,0.02,17/self.m,self.iteration,graph,w_all)
        #         # self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.4/self.m,0.02,17/self.m,self.iteration,graph,w_all)
        #         error_amc,wamc = self.pg_extra_approximate_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.041/self.m,0.04,0.087/self.m,self.iteration,graph,w_all,1)
        #         amc_list.append(error_amc[-1])
        #         mc_list.append(error_centralized_mc[-1][0])
        #         l1_list.append(error_l1[-1])
        #         I_approximation = U_all.T@U_all/self.m
        #         I_list.append(np.linalg.norm(I-I_approximation))
        #         i+=1
        # print("amc difference from wstar", sum(amc_list)/trials)
        # print("mc difference from wstar", sum(mc_list)/trials)
        # print("l1 difference",sum(l1_list)/trials)
        # print("I approximation",sum(I_list)/trials)
        

        

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()