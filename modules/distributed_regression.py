import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import copy
from modules.base_functions import base
import numpy.linalg as LA


class update_functions(base):
	def centralized_gradient_descent(self,Ut,d,w,w_star,L2,eta,iteration):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - eta*(np.dot(U,(np.dot(Ut,w)-d)))
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		plt.plot(times,error,label = "centralized gradient descent")
		return error,w

	def centralized_L1(self,Ut,d,w,w_star,L2,lamb,eta,iteration):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - eta*(np.dot(U,(np.dot(Ut,w)-d)))
			for j in range(len(w)):
				if w[j] > 0 and eta*lamb < abs(w[j]):
					w[j] -= eta*lamb
				elif w[j] < 0 and eta*lamb < abs(w[j]):
					w[j] += eta*lamb
				else:
					w[j] = 0
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		exec("plt.plot(times,error,label = 'centralized_L1')")
		return error,w

	def centralized_mc(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - eta*(np.dot(U,(np.dot(Ut,w)-d)))
			for j in range(len(w)):
				if abs(w[j]) < eta*lamb:
					w[j] = 0
				elif eta*lamb < abs(w[j]) and abs(w[j]) < lamb/rho:
					w[j] = w[j]*(abs(w[j])-eta*lamb)/(abs(w[j])*(1-eta*rho))
				elif lamb/rho <= abs(w[j]):
					w[j] = w[j]
				else:
					print("banana")
					print("eta*lamb",eta*lamb,"lamb/rho =",lamb/rho,"w = ",w[j])
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
			one_error = 0
		plt.plot(times,error,label = "centralized_mc")
		return error,w

	def one_gradient_descent(self,Ut,d,w,eta):
		U = Ut.T
		w = w - eta*((U*(np.dot(Ut,w)-d)))
		return w

	def all_gradient_descent(self,Ut,w_next,d,w_all,eta):
		for i in range(len(Ut)):
			w_next[i] = self.one_gradient_descent(Ut[i],d[i],w_all[i],eta)
		return w_next

	def one_L1(self,Ut,d,w,lamb,eta):
		U = Ut.T
		w = w - eta*((U*(np.dot(Ut,w)-d)))
		for j in range(len(w)):
			if w[j] > 0 and eta*lamb < abs(w[j]):
				w[j] -= eta*lamb
			elif w[j] < 0 and eta*lamb < abs(w[j]):
				w[j] += eta*lamb
			else:
				w[j] = 0
		return w
	
	def all_L1(self,Ut,w_next,d,w_all,lamb,eta):
		for i in range(len(Ut)):
			w_next[i] = self.one_L1(Ut[i],d[i],w_all[i],lamb,eta)
		return w_next

	def one_mc(self,Ut,d,w,lamb,eta,rho):
		U = Ut.T
		w = w - eta*((U*(np.dot(Ut,w)-d)))
		for j in range(len(w)):
			if abs(w[j]) <= eta*lamb:
				w[j] = 0
			elif eta*lamb < abs(w[j]) and abs(w[j]) < lamb/rho:
				w[j] = w[j]*(abs(w[j])-eta*lamb)/(abs(w[j])*(1-eta*rho))
			elif lamb/rho <= abs(w[j]):
				w[j] = w[j]
			else:
				print("banana")
		return w
	
	def all_mc(self,Ut,w_next,d,w_all,lamb,eta,rho):
		for i in range(len(Ut)):
			w_next[i] = self.one_mc(Ut[i],d[i],w_all[i],lamb,eta,rho)
		return w_next

	def distributed_gradient_descent(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,w_star,N,L2,m))
			w_all_next = self.all_gradient_descent(Ut,w_all_next,d,w_all_iter,eta)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'new distributed gradient descent')
		#return average_error,w_all

	def distributed_L1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,w_star,N,L2,m))
			w_all_next = self.all_L1(Ut,w_all_next,d,w_all_iter,lamb,eta)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'new L1')
		#return average_error,w_all

	def distributed_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,w_star,N,L2,m))
			w_all_next = self.all_mc(Ut,w_all_next,d,w_all_iter,lamb,eta,rho)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'distributed mc')
		return np.mean(w_all_iter,axis = 0)

	def distributed_mc_compare(self,Ut,d,wcmc,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		size = np.dot(wcmc.T,wcmc)[0][0]
		print(size)
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,wcmc,N,size,m))
			w_all_next = self.all_mc(Ut,w_all_next,d,w_all_iter,lamb,eta,rho)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'mc compare centralized with decentralized')
		return np.mean(w_all_iter,axis = 0)

	def pg_extra_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)

		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox = self.extra_share(Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,w_all_next,d,w_all_prox,lamb,eta,rho)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'extra mc')
		return np.mean(w_all_next,axis = 0)

	def extra(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_between = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_between = copy.deepcopy(w_all_next)
			w_all_next = self.extra_share(Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i)
			w_all_before = copy.deepcopy(w_all_between)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'extra')
		return np.mean(w_all_next,axis = 0)
	
	def extra_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_next - c_tilde@w_all_before - eta*(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))
	
	def pg_extra_l1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox = self.extra_share(Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,w_all_next,d,w_all_prox,lamb,eta)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'extra l1')
		return np.mean(w_all_next,axis = 0)
	
	def gradient(self,Ut,w,d):
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			gradient[i] = (((Ut[i]).T)*(np.dot(Ut[i],w[i])-d[i]))
		return gradient
	
	def one_extra_L1(self,Ut,d,w,lamb,eta):
		U = Ut.T
		for j in range(len(w)):
			if w[j] > 0 and eta*lamb < abs(w[j]):
				w[j] -= eta*lamb
			elif w[j] < 0 and eta*lamb < abs(w[j]):
				w[j] += eta*lamb
			else:
				w[j] = 0
		return w
	
	def all_extra_L1(self,Ut,w_next,d,w_all,lamb,eta):
		for i in range(len(Ut)):
			w_next[i] = self.one_L1(Ut[i],d[i],w_all[i],lamb,eta)
		return w_next

	def one_extra_mc(self,Ut,d,w,lamb,eta,rho):
		U = Ut.T
		for j in range(len(w)):
			if abs(w[j]) <= eta*lamb:
				w[j] = 0
			elif eta*lamb < abs(w[j]) and abs(w[j]) < lamb/rho:
				w[j] = w[j]*(abs(w[j])-eta*lamb)/(abs(w[j])*(1-eta*rho))
			elif lamb/rho <= abs(w[j]):
				w[j] = w[j]
			else:
				print("banana")
		return w
	
	def all_extra_mc(self,Ut,w_next,d,w_all,lamb,eta,rho):
		for i in range(len(Ut)):
			w_next[i] = self.one_mc(Ut[i],d[i],w_all[i],lamb,eta,rho)
		return w_next