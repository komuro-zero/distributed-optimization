#functions used in each simulations
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

#np.random.seed(0)

class functions():
	def w_star(self,N,sparse_percentage):
		w_star = randn(N,1)     #平均は0
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  #重複要素の削除
		return w_star
	
	def w_star_weakly_sparse(self,N,sparse_percentage,how_weak):
		w_star = randn(N,1)     #平均は0
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  #重複要素の削除
		w_star_prox = np.zeros((len(w_star),1))
		for i in range(len(w_star)):
			if w_star[i] == 0:
				w_star_prox[i] = randn()
		w_star_prox = w_star_prox/np.dot(w_star_prox.T,w_star_prox)
		w_star = w_star/np.dot(w_star.T,w_star)
		w_star += how_weak*w_star_prox
		return w_star
	
	def w_star_weakly_sparse_sample(self,N,sparse_percentage):
		w_star = self.w_star(N,sparse_percentage)
		for i in range(len(w_star)):
			if w_star[i] == 0:
				w_star[i] = 0.0001
			else:
				w_star[i] = 1
		return w_star

	def db(self,x, y):
		db = 10 * np.log10(x / y)    
		return db[0][0]

	def adjacency_matrix(self,m,r_i):
		"""
		disjointed example
		a = [1]*r_i
		b = [0]*(m-r_i)
		a.extend(b)
		c = [1]*r_i
		d = [0]*(m-r_i)
		d.extend(c)
		aa = [a]*int(m/2)
		dd = [d]*int(m/2)
		dd.extend(aa)
		result = np.array(dd)
		for i in range(len(result)):
			result[i][i]=1

		"""
		c = np.eye(m) 
		for j in range(m):
			counter = 0
			while counter != r_i:
				random_num = np.random.randint(0,m)
				if c[j][random_num] != 1:
					c[j][random_num] = 1
					counter += 1
		return c
	
	def find_ones(self,row):
		result = [i for i, element in enumerate(row) if element == 1]
		return result

	def all_connections(self,c_list):
		result = []
		for row in c_list:
			result.append(self.find_ones(row))
		return result

	def delete_self(self,all_connection):
		for i in range(len(all_connection)):
			all_connection[i].remove(i)
		return all_connection

	def disjoint_checker(self,c,m):
		c_list = c.tolist()
		all_connection = self.all_connections(c_list)
		adjacent_matrix = self.delete_self(all_connection)
		check_node = adjacent_matrix[0]
		connected_node = adjacent_matrix[0]
		last_length = len(connected_node)
		check_node_box = []
		count = 1

		flag = True
		while flag:
			for i in check_node:
				check_node_box[0:0] = (all_connection[i])
			common_part = list(set(check_node_box) & set(connected_node))
			check_node = list(set(check_node_box) - set(common_part))
			connected_node[0:0] = check_node_box
			connected_node = list(set(connected_node))
			check_node_box = []
			now_length = len(connected_node)
			print(connected_node)
			if now_length == last_length:
				flag = False
			count += 1
			if count > m or len(connected_node) == m:
				flag = False
			last_length = now_length
		if len(connected_node) == m:
			print("your graph is not disjoint")
		else:
			print("your graph is disjoint. connected nodes:",now_length)

	def w_average(self,w_all):
		result = [0]*len(w_all[0])
		for row in w_all:
			result += row
		result = result/len(w_all)
		return result
	
	def make_w(self,m,N):
		w = randn(N,1).T
		for i in range(m-1):
			w = np.concatenate((w,randn(N,1).T))
		return w

	def centralized_gradient_descent(self,Ut,d,w,w_star,L2,eta,iteration):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - 2*eta*(np.dot(U,(np.dot(Ut,w)-d)))
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
			w = w - 2*eta*(np.dot(U,(np.dot(Ut,w)-d)))
			for j in range(len(w)):
				if w[j] > 0 and lamb < abs(w[j]):
					w[j] -= lamb
				elif w[j] < 0 and lamb < abs(w[j]):
					w[j] += lamb
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
			w = w - 2*eta*(np.dot(U,(np.dot(Ut,w)-d)))
			for j in range(len(w)):
				if abs(w[j]) < eta*lamb:
					w[j] = 0
				elif eta*lamb < abs(w[j]) and abs(w[j]) < lamb/rho:
					w[j] = w[j]*(abs(w[j])-eta*lamb)/(abs(w[j])*(1-eta*rho))
				elif lamb/rho <= abs(w[j]):
					w[j] = w[j]
				else:
					print("banana")
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
			one_error = 0
		plt.plot(times,error,label = "centralized_mc")
		return error,w

	def one_gradient_descent(self,Ut,d,w,eta):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
		return w

	def one_L1(self,Ut,d,w,lamb,eta):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
		for j in range(len(w)):
			if w[j] > 0 and lamb < abs(w[j]):
				w[j] -= lamb
			elif w[j] < 0 and lamb < abs(w[j]):
				w[j] += lamb
			else:
				w[j] = 0
		return w

	def one_mc(self,Ut,d,w,lamb,eta,rho):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
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

	def distributed_gradient_descent(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration,c,w_all_f):
		average_error = [0]
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_gradient_descent(u_%d,d_%d,w_%d,eta)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i+1)
		exec("w_all_f = w_all")
		exec("w_average = self.w_average(w_all)")
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		plt.plot(times,average_error,label = 'distributed gradient descent')
		plt.legend()
		return average_error,w_all_f

	def distributed_L1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,iteration,c,w_all_f):	
		average_error = [0]
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)"%(i,i))
		average_error[0] = average_error[0]/m

		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_L1(u_%d,d_%d,w_%d,lamb,eta)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i+1)
		exec("w_average = self.w_average(w_all)")
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		plt.plot(times,average_error,label = 'distributed L1')
		plt.legend()
		return average_error,w_average

	def distributed_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all_f):
		average_error = [0]
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = [self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)]" % (i,i,i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_mc(u_%d,d_%d,w_%d,lamb,eta,rho)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i+1)
		exec("w_average = self.w_average(w_all)")
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		for i in range(1,m+1):
			exec("plt.plot(times,error_w_%d)"%(i))
		plt.plot(times,average_error,label = 'distributed mc',color = "black")
		plt.legend()
		return average_error,w_average

	def distributed_gradient_descent_wj(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration,c,w_all_f,wj):
		average_error = [0]
		L2wj = np.dot(wj.T,wj)
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel()),L2wj)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_gradient_descent(u_%d,d_%d,w_%d,eta)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel())" % (j,j,j))
				#exec("print(one_error_%d)"%(j))
				exec("error_w_%d.append(self.db(one_error_%d,L2wj))" % (j,j))
			times.append(i+1)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		#exec("plt.plot(times,average_error,label = 'wj distributed gradient descent')")
		return average_error,w_average

	def distributed_L1_wj(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,iteration,c,w_all_f,wj):	
		average_error = [0]
		L2wj = np.dot(wj.T,wj)
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel()),L2wj)"%(i,i))
		average_error[0] = average_error[0]/m

		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_L1(u_%d,d_%d,w_%d,lamb,eta)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2wj))" % (j,j))
			times.append(i+1)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		exec("plt.plot(times,average_error,label = 'wj distributed L1')")
		return average_error

	def distributed_mc_wj(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all_f,wj):
		average_error = [0]
		L2wj = np.dot(wj.T,wj)
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel()),L2wj)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_mc(u_%d,d_%d,w_%d,lamb,eta,rho)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel())" % (j,j,j))
				
				exec("error_w_%d.append(self.db(one_error_%d,L2wj))" % (j,j))
			times.append(i+1)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		exec("plt.plot(times,average_error,label = 'wj distributed mc')")
		return average_error
