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
        self.m = 20
        self.r_i = 3
        self.iteration = 10000
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
        # matplotlib.rcParams['ps.useafm'] = True
        # matplotlib.rcParams['pdf.use14corefonts'] = True
        # matplotlib.rcParams['text.usetex'] = True 
        plt.rcParams["font.family"] = "Times New Roman" 
        plt.rcParams['text.usetex'] = True 
        pdf = PdfPages('main performance comparison SSP.pdf')
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 16 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        
        x = [10,20,30,40,50,60,70,80,90]
        # y_I = [3.48,2.1,1.54,1.48,1.51,1.27,1.24,1.14,1.01,0.97]
        y_I = [3.27,2.44,1.86,1.65,1.46,1.30,1.21,1.15,1.08,1.03]
        # y_amc_star = [-27.27,-32.31,-37.15,-39.34,-39.45,-39.33,-42.19,-43.83,-43.01,-39.89]
        # y_mc_star = [-32.48,-32.44,-39.71,-38.54,-39.21,-40.92,-42.22,-45.31,-43.32,-27.10]
        # y_l1 = [-32.37,-32.15,-35.99,-37.65,-38.70,-39.64,-41.26,-42.00,-41.84,42.15]
        # y_amc_star_average = [-20.92,-29.94,-36.96,-42.04,-40.98,-40.18,-41.85,-43.15,-43.39]
        # y_mc_star_average = [-28.30,-32.13,-39.89,-39.90,-39.22,-39.96,-41.87,-47.75,-43.60]
        # y_l1_average = [-16.80,-27.54,-35.89,-39.74,-38.70,-38.76,-41.06,-40.91,-41.19]
        # y_amc_star_average_100 = [-20.84,-30.15,-36.78,-38.86,-39.45,-39.63,-42.19,-43.73,-43.31]
        # y_mc_star_average_100 = [-29.38,-32.41,-39.71,-38.54,-39.21,-40.92,-42.22,-45.31,-43.32]
        # y_l1_average_100 = [-16.65,-28.25,-34.79,-37.15,-38.69,-39.64,-41.26,-41.99,-41.84]
        y_amc_star_average_400 = [-20.34,-30.97,-37.14,-39.05,-39.19,-39.61,-41.85,-42.64,-43.27]
        y_mc_star_average_400 = [-29.43,-32.93,-39.91,-38.58,-38.96,-40.36,-41.83,-44.67,-43.47]
        y_l1_average_400 = [-15.81,-28.72,-35.30,-37.58,-38.42,-39.61,-40.86,-41.39,-42.12]
        fig = plt.figure()
        plot_1 = fig.add_subplot(111)
        
        # ln1 = plot_1.plot(x,y_amc_star,label= "System mismatch of AME penalty")
        # ln1 = plot_1.plot(x,y_mc_star,label= "System mismatch of ME penalty")
        # ln1 = plot_1.plot(x,y_l1,label= r"System mismatch of $\ell_1$ penalty")
        # ln1 = plot_1.plot(x,y_amc_star_average_100,label= "AMC penalty")
        # ln1 = plot_1.plot(x,y_mc_star_average_100,label= "MC penalty")
        # ln1 = plot_1.plot(x,y_l1_average_100,label= r"$\ell_1$ penalty")
        ln1 = plot_1.plot(x,y_amc_star_average_400,label= "AMC penalty")
        ln1 = plot_1.plot(x,y_mc_star_average_400,label= "MC penalty")
        ln1 = plot_1.plot(x,y_l1_average_400,label= r"$\ell_1$ penalty")
        plot_1.set_xlabel("network size",fontsize=16)
        plot_1.set_ylabel("System Mismatch (dB)",fontsize=16)
        plot_1.grid(which = "major")
        # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # plt.axhline(y = error_centralized_mc_twin[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        plot_1.legend(fontsize=16)
        plot_2 = plot_1.twinx()
        trial = 400
        all_I_validity = []
        I = np.eye(self.N)
        # for network_size in range(10,101,10):
        for network_size in range(10,100,10):
            I_approximation_list_per_network = []
            for trials in range(trial):
                w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,network_size,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
                # for i in range(network_size):
                #     U_all[i] = U_all[i]/np.linalg.norm(U_all[i])
                I_approximation = (U_all.T@U_all)/(network_size)
                approximation_validity = np.linalg.norm(I-I_approximation)/np.linalg.norm(I)
                I_approximation_list_per_network.append(approximation_validity)
            all_I_validity.append(sum(I_approximation_list_per_network)/len(I_approximation_list_per_network))
        ln2 = plot_2.plot(x,all_I_validity,color = "purple", label="Approximation Validity")
        plot_2.set_ylabel("Validity of approximation",fontsize=16)
        plt.savefig('system_mismatch_of_different_network_size_400_trials.pdf')
        plot_2.legend(fontsize=16,loc = "center right")
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