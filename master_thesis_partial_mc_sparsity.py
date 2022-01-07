"""experiment for partial mc"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from modules.distributed_regression import update_functions


"""
-30.01575276179356-31.223088916352264
"""


np.random.seed(0)
class DistributedUpdates(update_functions):
    def __init__(self):
        """initializing setting parameters"""
        self.N = 10
        self.m = 9
        self.r_i = 4
        self.iteration = 3000
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
        """running the experiments"""
        # w, w_star, w_all, U_all, d_all, L2, graph = self.make_variables_noise_after_2(self.N, self.m, self.r_i,\
        #      self.sparsity_percentage, self.how_weakly_sparse, self.w_noise, self.normal_distribution, self.w_zero)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 16  #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

        ################################################################
        animation_flag = False
        ################################################################

        lambda_array = np.array([0.05/self.m]*self.m)
        rho_array = np.array([9/self.m]*self.m)
        lambda_array[0] = 10**-16/self.m
        lambda_array[1] = 0.11/self.m
        lambda_array[2] = 0.06/self.m
        lambda_array[3] = 0.55/self.m
        lambda_array[4] = 10**-14/self.m
        lambda_array[5] = 0.07/self.m
        lambda_array[6] = 0.015/self.m
        lambda_array[7] = 0.06/self.m
        lambda_array[8] = 0.015/self.m
        rho_array[8] = 8/self.m
        l1_list = [0.04/self.m]*self.m
        l1_list[0] = 0.0009
        l1_list[1] = 0.006
        l1_list[2] = 0.006
        l1_list[3] = 0.0009
        l1_list[4] = 0.006
        l1_list[5] = 0.006
        l1_list[6] = 0.0009
        l1_list[7] = 0.007
        l1_list[8] = 0.006
        # node = 8
        # best_error_1 = 0
        # best_error_2 = 0
        # best_error_3 = 0
        # optimal_lambda_1 = 0
        # optimal_lambda_2 = 0
        # optimal_lambda_3 = 0
        # for i in range(10):
        #     lamb = 10 ** -(i)
        #     l1_list[node] = lamb
        #     error, w_l1 = self.pg_extra_l1_each_param(U_all, d_all, w_star, L2, self.N, self.m,self.r_i, l1_list, 1, 0.09, self.iteration, graph, w_all, 0, 0.05,not animation_flag, animation_flag)
        #     if error[-1] < best_error_1:
        #         best_error_1 = error[-1]
        #         optimal_lambda_1 = lamb
        # print(optimal_lambda_1,best_error_1)
        # for i in range(10):
        #     if i < 5:
        #         lamb = optimal_lambda_1* 0.5 + optimal_lambda_1 * 0.1 *(i)
        #     elif i >= 5:
        #         lamb = optimal_lambda_1*(1 + i)
        #     l1_list[node] = lamb
        #     error, w_l1 = self.pg_extra_l1_each_param(U_all, d_all, w_star, L2, self.N, self.m,self.r_i, l1_list, 1, 0.09, self.iteration, graph, w_all, 0, 0.05,not animation_flag, animation_flag)
        #     if error[-1] < best_error_2:
        #         best_error_2 = error[-1]
        #         optimal_lambda_2 = lamb
        # print(optimal_lambda_2,best_error_2)
        # error_l1, w_l1 = self.pg_extra_l1(U_all, d_all, w_star, L2, self.N, self.m,self.r_i, 0.04/self.m, 1, 0.09, self.iteration, graph, w_all, 0, 0.05, not animation_flag, animation_flag)
        # self.pg_extra_l1_each_param(U_all, d_all, w_star, L2, self.N, self.m,self.r_i, l1_list, 1, 0.09, self.iteration, graph, w_all, 0, 0.05, not animation_flag, animation_flag)
        # error_pmc, pmc_w = self.pg_extra_partial_mc(U_all, d_all, w_star, L2, self.N, self.m,self.r_i, 0.05/self.m, 0.1, 9/self.m, self.iteration, graph, w_all, 0.05, not animation_flag, animation_flag)
        # error_pmc_ip, pmc_w_ip = self.pg_extra_partial_mc_each_param(U_all, d_all, w_star, L2, self.N, self.m, self.r_i, lambda_array, 0.3, rho_array, self.iteration, graph, w_all, 0.05, not animation_flag, animation_flag)
        # print(error_l1[-1],error_pmc[-1],error_pmc_ip[-1])

        # error_l1, w_l1 = self.pg_extra_l1(U_all, d_all, w_star, L2, self.N, self.m,self.r_i, 0.04/self.m, 2, 0.09, self.iteration, graph, w_all, 0, 0.05, True, animation_flag)
        # error_pmc, pmc_w = self.pg_extra_partial_mc(U_all, d_all, w_star, L2, self.N, self.m, self.r_i, 0.05/self.m, 0.1, 8.9/self.m, self.iteration, graph, w_all, 0.05, True, animation_flag)
        # error_pmc, pmc_w = self.pg_extra_partial_mc(U_all, d_all, w_star, L2, self.N, self.m, self.r_i, 0.1/self.m, 0.1, 8.9/self.m, self.iteration, graph, w_all, 0.05, True, animation_flag)
        sparsity_list = []
        error_list = []
        for sparsity_i in range(10):
            best_error_2_list = []
            for i in range(100):
                sparsity = (sparsity_i + 1) * 5
                print(sparsity)
                w, w_star, w_all, U_all, d_all, L2, graph = self.make_variables_noise_after_2(self.N, self.m, self.r_i,\
                sparsity, self.how_weakly_sparse, self.w_noise, self.normal_distribution, self.w_zero)
                optimal_lamb_1 = 0
                optimal_lamb_2 = 0
                best_error = 0
                for i in range(12):
                    this_lamb = 10 ** (i - 10)
                    error_pmc, w_pmc = self.pg_extra_partial_mc(U_all, d_all, w_star, L2, self.N, self.m, self.r_i, this_lamb/self.m, 0.1, 8.9/self.m, self.iteration, graph, w_all, 0.05, True, False)
                    if error_pmc != None:
                        if error_pmc[-1] < best_error:
                            best_error = error_pmc[-1]
                            optimal_lamb_1 = this_lamb/self.m
                print(best_error,optimal_lamb_1)
                best_error_2 = 0
                for j in range(55):
                    if j < 50:
                        this_lamb = optimal_lamb_1 * 0.1 * (j + 1)
                    else:
                        this_lamb = optimal_lamb_1 * (j - 45)
                    print(this_lamb)
                    error_pmc, w_pmc = self.pg_extra_partial_mc(U_all, d_all, w_star, L2, self.N, self.m, self.r_i, this_lamb/self.m, 0.1, 8.9/self.m, self.iteration, graph, w_all, 0.05, False, False)
                    if error_pmc != None:
                        if error_pmc[-1] < best_error_2:
                            best_error_2 = error_pmc[-1]
                            optimal_lamb_2 = this_lamb/self.m
                print(best_error_2,optimal_lamb_2)
                best_error_2_list.append(best_error_2)
            sparsity_list.append(sparsity)
            error_list.append(sum(best_error_2_list)/len(best_error_2_list))
        plt.show()            
        plt.plot(sparsity_list, error_list,label="System Mismatch")

        if animation_flag:
            all_variables = []
            fig = plt.figure()

            for i in range(len(error)):

                w_all_next_l1 = w_l1[i]
                w_all_next_pmc = pmc_w[i]
                w_all_next_pmc_ip = pmc_w_ip[i]
                x_pmc = w_all_next_pmc[:, [0]].flatten()
                y_pmc = w_all_next_pmc[:, [1]].flatten()
                x_l1 = w_all_next_l1[:, [0]].flatten()
                y_l1 = w_all_next_l1[:, [1]].flatten()
                x_pmc_ip = w_all_next_pmc_ip[:, [0]].flatten()
                y_pmc_ip = w_all_next_pmc_ip[:, [1]].flatten()
                variable_pmc = plt.plot(x_pmc, y_pmc, "o", color="red", label="pmc")
                variable_l1 = plt.plot(x_l1, y_l1, "o", color="blue", label="l1")
                variable_pmc_ip = plt.plot(x_pmc_ip, y_pmc_ip, "o", color="green", label="pmc individual parameter")
                # if i == 0:
                #     plt.legend()
                all_variables.append(variable_l1+variable_pmc+variable_pmc_ip)
            ani = animation.ArtistAnimation(fig,all_variables,interval=1)
            plt.grid(which="major")
            plt.plot(w_star[1], w_star[2], "o", color="black", label="w*")
            plt.show()
        else:
            plt.xlabel("Sparsity Percentage (%)", fontsize=16)
            plt.ylabel("System Mismatch (dB)", fontsize=16)
            plt.grid(which="major")
            plt.legend(fontsize=16)
            plt.savefig('master_thesis_pmc_sparsity.pdf')
            plt.show()
            # plt.legend()

if __name__ == "__main__":
    SIMULATION = DistributedUpdates()
    SIMULATION.run()
