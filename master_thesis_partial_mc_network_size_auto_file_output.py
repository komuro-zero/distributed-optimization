"""experiment for partial mc"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from modules.distributed_regression import update_functions
import pickle

"""
-30.01575276179356-31.223088916352264
"""


# np.random.seed(0)
class DistributedUpdates(update_functions):
    def __init__(self):
        """initializing setting parameters"""
        self.N = 10
        self.m = 9
        self.r_i = 2
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

        network_list = []
        error_list = []
        error_list_l1 = []
        error_list_mc = []
        error_list_amc = []
        total_trials = 100
        network_size_iter = 7
        for network_size in range(network_size_iter):
            best_error_3_list = []
            best_error_3_list_l1 = []
            best_error_3_list_mc = []
            best_error_3_list_amc = []
            for trials in range(total_trials):
                print(network_size+1, f"/{network_size_iter}", trials+1, f"/{total_trials}")
                network_size_i = (3 + network_size)
                w, w_star, w_all, U_all, d_all, L2, graph = self.make_variables_noise_after_2(self.N, network_size_i, self.r_i,\
                self.sparsity_percentage, self.how_weakly_sparse, self.w_noise, self.normal_distribution, self.w_zero)
                optimal_lamb_1 = 0
                optimal_lamb_2 = 0
                optimal_lamb_3 = 0
                optimal_lamb_l1_1 = 0
                optimal_lamb_l1_2 = 0
                optimal_lamb_l1_3 = 0
                optimal_lamb_mc_1 = 0
                optimal_lamb_mc_2 = 0
                optimal_lamb_mc_3 = 0
                optimal_lamb_amc_1 = 0
                optimal_lamb_amc_2 = 0
                optimal_lamb_amc_3 = 0
                best_error = 0
                best_error_2 = 0
                best_error_3 = 0
                best_error_l1_1 = 0
                best_error_l1_2 = 0
                best_error_l1_3 = 0
                best_error_mc_1 = 0
                best_error_mc_2 = 0
                best_error_mc_3 = 0
                best_error_amc_1 = 0
                best_error_amc_2 = 0
                best_error_amc_3 = 0
                best_increment = 0
                best_increment_l1 = 0
                best_increment_amc = 0
                best_increment_mc = 0
                for i in range(12):
                    this_lamb = 10 ** (i - 10)
                    error_pmc, w_pmc = self.pg_extra_partial_mc(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb/network_size_i, 0.1, 8.9/network_size_i, self.iteration, graph, w_all, 0.05, False, False)
                    if error_pmc != None:
                        if error_pmc[-1] < best_error:
                            best_error = error_pmc[-1]
                            optimal_lamb_1 = this_lamb
                    w_l1_error, wl1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,network_size_i,self.r_i,this_lamb/network_size_i,1,0.09,self.iteration,graph,w_all,0,0.75,False,False)
                    if w_l1_error[-1] < best_error_l1_1:
                        best_error_l1_1 = w_l1_error[-1]
                        optimal_lamb_l1_1 = this_lamb
                    w_amc_error, wamc = self.pg_extra_approximate_mc(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb/network_size_i, 0.027, 10**-8/self.m, self.iteration, graph, w_all, 1,False)
                    if w_amc_error != None:
                        if w_amc_error[-1] < best_error_amc_1:
                            best_error_amc_1 = w_amc_error[-1]
                            optimal_lamb_amc_1 = this_lamb
                    w_mc_error = self.prox_dgd(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb/network_size_i, 1, 35/network_size_i ,self.iteration, graph, w_all,False)
                    if w_amc_error != None:
                        if w_mc_error[-1] < best_error_mc_1:
                            best_error_mc_1 = w_mc_error[-1]
                            optimal_lamb_mc_1 = this_lamb
                best_increment = 0
                for j in range(19):
                    if j < 10:
                        this_lamb = optimal_lamb_1 * 0.1 * (j + 1)
                        increment = j + 1
                    else:
                        this_lamb = optimal_lamb_1 * (j - 8)
                        increment = j - 8
                    error_pmc, w_pmc = self.pg_extra_partial_mc(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb/network_size_i, 0.1, 8.9/network_size_i, self.iteration, graph, w_all, 0.05, False, False)
                    if error_pmc != None:
                        if error_pmc[-1] < best_error_2:
                            best_error_2 = error_pmc[-1]
                            optimal_lamb_2 = this_lamb
                            best_increment = increment
                    if j < 10:
                        this_lamb_l1 = optimal_lamb_l1_1 * 0.1 * (j + 1)
                    else:
                        this_lamb_l1 = optimal_lamb_l1_1 * (j - 8)
                    w_l1_error, wl1 = self.pg_extra_l1(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb_l1/network_size_i, 1, 0.09, self.iteration, graph, w_all, 0, 0.75, False, False)
                    if w_l1_error[-1] < best_error_l1_2:
                        best_error_l1_2 = w_l1_error[-1]
                        optimal_lamb_l1_2 = this_lamb_l1
                        best_increment_l1 = increment
                    if j < 10:
                        this_lamb_amc = optimal_lamb_amc_1 * 0.1 * (j + 1)
                    else:
                        this_lamb_amc = optimal_lamb_amc_1 * (j - 8)
                    w_amc_error, wamc = self.pg_extra_approximate_mc(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb_amc/network_size_i, 0.027, 10**-8/self.m, self.iteration, graph, w_all, 1, False)
                    if w_amc_error != None:
                        if w_amc_error[-1] < best_error_amc_2:
                            best_error_amc_2 = w_amc_error[-1]
                            optimal_lamb_amc_2 = this_lamb_amc
                            best_increment_amc = increment
                    if j < 10:
                        this_lamb_mc = optimal_lamb_mc_1 * 0.1 * (j + 1)
                    else:
                        this_lamb_mc = optimal_lamb_mc_1 * (j - 8)
                    w_mc_error = self.prox_dgd(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb_mc/network_size_i, 1, 35/network_size_i ,self.iteration, graph, w_all,False)
                    if w_mc_error != None:
                        if w_mc_error[-1] < best_error_mc_2:
                            best_error_mc_2 = w_mc_error[-1]
                            optimal_lamb_mc_2 = this_lamb_mc
                            best_increment_mc = increment
                if best_increment_mc == 0:
                    best_increment_mc = 1
                for k in range(19):
                    this_lamb = (optimal_lamb_2 - optimal_lamb_2/best_increment) + (k + 1) * 0.1 * optimal_lamb_2/best_increment
                    error_pmc, w_pmc = self.pg_extra_partial_mc(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb/network_size_i, 0.1, 8.9/network_size_i, self.iteration, graph, w_all, 0.05, False, False)
                    if error_pmc != None:
                        if error_pmc[-1] < best_error_3:
                            best_error_3 = error_pmc[-1]
                            optimal_lamb_3 = this_lamb/network_size_i
                    this_lamb_l1 = (optimal_lamb_l1_2 - optimal_lamb_l1_2/best_increment_l1) + (k + 1) * 0.1 * optimal_lamb_l1_2/best_increment_l1
                    w_l1_error, wl1 = self.pg_extra_l1(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb_l1/network_size_i, 1, 0.09, self.iteration, graph, w_all, 0, 0.75, False, False)
                    if w_l1_error[-1] < best_error_l1_3:
                        best_error_l1_3 = w_l1_error[-1]
                        optimal_lamb_l1_3 = this_lamb_l1
                    if best_increment_amc != 0:
                        this_lamb_amc = (optimal_lamb_amc_2 - optimal_lamb_amc_2/best_increment_amc) + (k + 1) * 0.1 * optimal_lamb_amc_2/best_increment_amc
                        w_amc_error, wamc = self.pg_extra_approximate_mc(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb_amc/network_size_i, 0.027, 10**-8/self.m, self.iteration, graph, w_all, 0.05,False)
                        if w_amc_error != None:
                            if w_amc_error[-1] < best_error_amc_3:
                                best_error_amc_3 = w_amc_error[-1]
                                optimal_lamb_amc_3 = this_lamb_amc
                    this_lamb_mc = (optimal_lamb_mc_2 - optimal_lamb_mc_2/best_increment_mc) + (k + 1) * 0.1 * optimal_lamb_mc_2/best_increment_mc
                    w_mc_error = self.prox_dgd(U_all, d_all, w_star, L2, self.N, network_size_i, self.r_i, this_lamb_mc/network_size_i, 1, 35/network_size_i ,self.iteration, graph, w_all,False)
                    if w_mc_error != None:
                        if w_mc_error[-1] < best_error_mc_3:
                            best_error_mc_3 = w_mc_error[-1]
                            optimal_lamb_mc_3 = this_lamb_mc
                print("pmc",best_error_3,optimal_lamb_3,optimal_lamb_3*network_size_i)
                print("l1",best_error_l1_3,optimal_lamb_l1_3,optimal_lamb_l1_3*network_size_i)
                print("mc",best_error_mc_3,optimal_lamb_mc_3,optimal_lamb_mc_3*network_size_i)
                print("amc",best_error_amc_3,optimal_lamb_amc_3,optimal_lamb_amc_3*network_size_i)
                best_error_3_list.append(best_error_3)
                best_error_3_list_l1.append(best_error_l1_3)
                best_error_3_list_mc.append(best_error_mc_3)
                best_error_3_list_amc.append(best_error_amc_3)
            network_list.append(network_size_i)
            error_list.append(sum(best_error_3_list)/len(best_error_3_list))
            error_list_l1.append(sum(best_error_3_list_l1)/len(best_error_3_list_l1))
            error_list_mc.append(sum(best_error_3_list_mc)/len(best_error_3_list_mc))
            error_list_amc.append(sum(best_error_3_list_amc)/len(best_error_3_list_amc))
        plt.plot(network_list, error_list_l1, marker="D", markersize=6, markeredgewidth=3, label=r"$\ell_1$")
        plt.plot(network_list, error_list_mc, marker="D", markersize=6, markeredgewidth=3, label="MC")
        plt.plot(network_list, error_list_amc, marker="D", markersize=6, markeredgewidth=3, label="AMC")
        plt.plot(network_list, error_list, marker="D", markersize=6, markeredgewidth=3, label="PMC")
        plt.legend()
        plt.show()
        with open(f"network_l1_{total_trials}.txt","wb") as f:
            pickle.dump(error_list_l1,f)
        with open(f"network_pmc_{total_trials}.txt","wb") as f:
            pickle.dump(error_list,f)
        with open(f"network_amc_{total_trials}.txt","wb") as f:
            pickle.dump(error_list_amc,f)
        with open(f"network_mc_{total_trials}.txt","wb") as f:
            pickle.dump(error_list_mc,f)
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
            pass
            # plt.xlabel("Network size", fontsize=16)
            # plt.ylabel("System Mismatch (dB)", fontsize=16)
            # plt.grid(which="major")
            # plt.legend(fontsize=16)
            # plt.savefig('master_thesis_pmc_network_size.pdf')
            # plt.legend()
            # plt.show()

if __name__ == "__main__":
    SIMULATION = DistributedUpdates()
    SIMULATION.run()