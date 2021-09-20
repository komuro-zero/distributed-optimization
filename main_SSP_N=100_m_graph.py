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
        self.iteration = 1000
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
        plt.rcParams["font.family"] = "Times New Roman" 
        plt.rcParams['text.usetex'] = True 
        # matplotlib.rcParams['ps.useafm'] = True
        # matplotlib.rcParams['pdf.use14corefonts'] = True
        # matplotlib.rcParams['text.usetex'] = True 
        pdf = PdfPages('main performance comparison SSP.pdf')
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 10 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        
        # x= [120,130,140,150,160,170,180,190,200]
        x= [120,140,160,180,200,400,600,800]
        y_w = [14.12,14.08,14.08,14.07,14.13,14.11,14.17,14.13]
        approx_validity = [9.02,8.35,7.79,7.37,7.01,5.05,4.08,3.54]
        y_amc = [-30.25,-32.76,-32.63,-33.97,-36.57,-37.77,-40.08,-41.67]
        y_wmc = [-31.97,-34.68,-36.29,-36.83,-39.61,-40.56,-42.30,-44.70]
        y_l1 = [-30.06,-32.24,-32.13,-32.75,-35.60,-36.37,-38.83,-40.33]
        y_extra = [-23.16,-26.31,-29.66,-29.12,-29.78,-34.28,-36.67,-38.11]
        plt.plot(x,y_w,label= "difference between estimates using amc and mc")
        plt.plot(x,approx_validity,label= "approximation validity")
        # plt.plot(x,y_amc,label= "difference of amc estimate and w*")
        # plt.plot(x,y_wmc,label= "difference of mc estimate and w*")
        # plt.plot(x,y_l1,label= "difference of l1 estimate and w*")
        # plt.plot(x,y_extra,label= "difference of extra estimate and w*")
        plt.xlabel("network size",fontsize=12)
        plt.ylabel("Mismatch of objective value (dB)/ Mismatch of points",fontsize=12)
        plt.grid(which = "major")
        # plt.axhline(y = error[-1],linestyle="dashed",label = "Centralized L1 penalty",color = "green")
        # plt.axhline(y = error_centralized_mc[-1],linestyle="dashed",label = "Centralized MC penalty",color = "green")
        # plt.axhline(y = error_centralized_mc_twin[-1],linestyle = "dashed",label = "Centralized Approximate MC penalty",color = "purple")
        # pdf.savefig()
        plt.legend(fontsize=12)
        plt.savefig('SSP_validity_AME.pdf')
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