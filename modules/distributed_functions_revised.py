#functions used in each simulations
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import numpy.linalg as LA
import copy

#np.random.seed(0)

class functions():
	def w_star(self,N,sparse_percentage):
		w_star = randn(N,1)     
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  
		return w_star
	
	def w_star_weakly_sparse(self,N,sparse_percentage,how_weak):
		w_star = randn(N,1)     
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  
		w_star_prox = np.zeros((len(w_star),1))
		for i in range(len(w_star)):
			if w_star[i] == 0:
				w_star_prox[i] = randn()
		w_star_prox = w_star_prox/np.linalg.norm(w_star_prox)
		w_star = w_star/np.linalg.norm(w_star)
		w_star += how_weak*w_star_prox
		w_star = w_star/np.linalg.norm(w_star)		
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
		return db

	def rho_checker(self,rho,lamb,eta):
		if lamb/rho <= eta*lamb:
			print("faulty rho")
			exit()
	
	def U_eigenvalue(self,U_all):
		square = U_all.T@U_all
		eigenvalue = LA.eig(square)[0]
		smallest_eigen = min(eigenvalue)
		biggest_eigen = max(eigenvalue)
		return smallest_eigen,biggest_eigen
	
	def params_checker(self,rho,lamb,eta,U_all,B,m,N,graph):
		small_eig, big_eig = self.U_eigenvalue(U_all)
		if rho > small_eig:
			print("rho is bigger than the smallest eigen",small_eig)
			exit()
		elif eta > 2/big_eig:
			print("eta may be too big")
		self.rho_checker(rho,lamb,eta)
		self.lipschitz_checker(U_all,B,m,eta,lamb)
		self.distributed_lipschitz_checker(U_all,B,m,eta,N,lamb)
		self.disjoint_checker(graph,m)
	
	def lipschitz_checker(self,U,B,m,eta,lamb):
		X = U@U.T-(B**2)*lamb*np.eye(m)
		U, s, V = np.linalg.svd(X)
		lpz = 2/max(s)
		if lpz < eta:
			print(f"eta must be smaller than {lpz/2}")
			exit()
	
	def lipschitz_checker_L1(self,U,m,eta,lamb):
		X = U@U.T
		U, s, V = np.linalg.svd(X)
		lpz = 2/max(s)
		if lpz < eta:
			print(f"eta must be smaller than {lpz/2}")
			exit()
	
	def distributed_lipschitz_checker(self,U,B,m,eta,N,lamb):
		for u in U:
			ut = np.reshape(u,(len(u),1)) 
			um = np.reshape(u,(1,len(u)))
			x = ut@um-(B**2)*lamb*np.eye(N)
			U, s, V = np.linalg.svd(x)
			lpz = 2/max(s)
			if lpz < eta:
				print(f"eta must be smaller than {lpz}")
				exit()

	def centralized_convexity_checker(self,B,lamb,U,N):
		B2 = B*B*np.eye(N)
		x = (1/lamb)*np.dot(U.T,U)-B2
		small_eig =  min(LA.eig(x)[0])
		if small_eig < 0:
			print(f"your smallest eigenvalue is {small_eig}. it is nonconvex.")
		else:
			print("your function is centrally convex. go fuck yourself")

	def distributed_convexity_checker(self,B,lamb,U,N):
		for u in U:
			B2 = B*B*np.eye(N)
			ut = np.reshape(u,[N,1])
			u = np.reshape(u,[1,N])
			x = (N/lamb)*np.dot(ut,u)-B2
			small_eig =  min(LA.eig(x)[0])
			if small_eig < 0:
				#print(f"your smallest eigenvalue is {small_eig}. it is nonconvex.")
				pass
			else:
				print("your function is convex. go fuck yourself")
				exit()

	def directed_graph(self,m,r_i):
		c = np.eye(m) 
		for j in range(m):
			counter = 0
			while counter != r_i:
				random_num = np.random.randint(0,m)
				if c[j][random_num] != 1:
					c[j][random_num] = 1
					counter += 1
		return c
	
	def disjointed_directed_graph(self,m,r_i):
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
		return result
	
	def make_range(self,m):
		result =[]
		for i in range(m):
			result.append(i)
		return result

	def make_checklist(self,r_i,m):
		untidy_result = self.make_range(m)
		for i in range(r_i-1):
			untidy_result[0:0] = self.make_range(m)
		return sorted(untidy_result)

	def horizontal_checker(self,graph,place,i,r_i):
		count = 0
		for row in graph:
			if row[i] == 1:
				count += 1
		if count == r_i+1:
			result = False
		elif count < r_i+1:
			result = True
		else:
			print(graph)
			print("faulty horizontal number")
			exit()
		if sum(graph[place]) <= r_i:
			result = True
		else:
			result = False
		return result

	def undirected_graph(self,m,r_i):
		graph = np.eye(m)
		checklist = self.make_checklist(r_i,m)
		for i in range(m):
			connected = sum(x == 1 for x in graph[i])
			while connected < r_i:
				select = np.random.randint(len(checklist))
				connect_node = checklist[select]
				if graph[i][connect_node] != 1 and self.horizontal_checker(graph,connect_node,i,r_i):
					graph[i][connect_node] = 1
					connected += 1
					graph[connect_node][i] =1
					checklist.pop(select)
		for row in graph:
			if sum(row) == r_i +1:
				print("good")
			else:
				print("bad",row)
				exit()
		return graph

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

	def disjoint_checker(self,graph,m):
		c_list = graph.tolist()
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
			if now_length == last_length:
				flag = False
			count += 1
			if count > m or len(connected_node) == m:
				flag = False
			last_length = now_length
		if len(connected_node) == m:
			print("Your graph is not disjoint")
		else:
			print("Your graph is disjoint. connected nodes:",now_length)

	def w_average(self,w_all,w_star):
		result = [0]*len(w_all[0])
		for row in w_all:
			result += row
		result = result/len(w_all)
		x = range(len(result))
		#fig = plt.figure()
		#plt.plot(x,w_star,color = "black",label = "w_star")
		#plt.plot(x,result,color = "red",label = "w_average")
		#plt.show()
		return result
	
	def make_w(self,m,N):
		w = randn(N,1).T
		for i in range(m-1):
			w = np.concatenate((w,randn(N,1).T))
		return w

	def error_distributed(self,w_all,w_star,N,L2,m):
		each_errors = []
		for w in w_all:
			w = np.reshape(w,(N,1))
			each_errors.append(self.db(((np.reshape(w-w_star,(1,N)))@(w-w_star))[0][0],L2))
		this_error = sum(each_errors)/m
		return this_error

	def make_variables(self,N,m,sparsity_percentage,how_weakly_sparse,w_noise):
		w = randn(N,1)
		w_star = self.w_star_weakly_sparse(N,sparsity_percentage,how_weakly_sparse)
		#w_star = self.w_star(N,sparsity_percentage)
		U_all = randn(m,N)
		w_star_noise = w_star +randn(N,1)*(10**-(w_noise/10))
		#w_star_noise = w_star_noise/np.dot(w_star_noise.T,w_star_noise)
		d_all = np.dot(U_all,w_star_noise)
		L2 = np.dot(w_star.T,w_star)

		return w,w_star,U_all,d_all,L2

	def make_variables_noise_after(self,N,m,r_i,sparsity_percentage,how_weakly_sparse,w_noise):
		w = randn(N,1)
		w_star = self.w_star_weakly_sparse(N,sparsity_percentage,how_weakly_sparse)
		U_all = randn(m,N)
		d_all = np.dot(U_all,w_star)
		d_all += d_all*randn(m,1)*(10**-(w_noise/10))
		L2 = np.dot(w_star.T,w_star)[0][0]
		graph = self.undirected_graph(m,r_i)
		w_all = self.make_w(m,N)
		return w,w_star,w_all,U_all,d_all,L2,graph

	def make_variables_no_noise(self,N,m,r_i,sparsity_percentage,how_weakly_sparse,w_noise):
		w = randn(N,1)
		w_star = self.w_star(N,sparsity_percentage)
		print(w_star)
		U_all = randn(m,N)
		d_all = np.dot(U_all,w_star)
		L2 = np.dot(w_star.T,w_star)[0][0]
		graph = self.undirected_graph(m,r_i)
		w_all = self.make_w(m,N)
		return w,w_star,w_all,U_all,d_all,L2,graph
	
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
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
		return w

	def all_gradient_descent(self,Ut,w_next,d,w_all,eta):
		for i in range(len(Ut)):
			w_next[i] = self.one_gradient_descent(Ut[i],d[i],w_all[i],eta)
		return w_next

	def one_L1(self,Ut,d,w,lamb,eta):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
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
		print(c)
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,w_star,N,L2,m))
			w_all_next = self.all_mc(Ut,w_all_next,d,w_all_iter,lamb,eta,rho)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
			print(w_all_iter)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'new mc')
		#return average_error,w_all

	def distributed_mc_compare(self,Ut,d,wcmc,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		size = np.dot(wcmc.T,wcmc)[0][0]
		print(size)
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(int(iteration/2)):
			average_error.append(self.error_distributed(w_all_iter,wcmc,N,size,m))
			w_all_next = self.all_mc(Ut,w_all_next,d,w_all_iter,lamb,eta,rho)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		for i in range(int(iteration/2)):
			average_error.append(self.error_distributed(w_all_iter,wcmc,N,size,m))
			w_all_next = self.all_mc(Ut,w_all_next,d,w_all_iter,lamb,eta,rho)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'mc compare centralized with decentralized')
		#return average_error,w_all