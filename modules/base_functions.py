#functions used in each simulations
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import numpy.linalg as LA
import copy
import random 
from statistics import mean
import networkx as nx
import math



# np.random.seed(0)

class base():
	def w_star(self,N,sparse_percentage):
		w_star = np.random.randn(N,1)     
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  
		return w_star

	def w_sin(self,w,a):
		w_copy = copy.deepcopy(w)
		for i in range(len(w_copy)):
			if abs(w[i]) > a:
				w_copy[i] = 0
			else:
				w_copy[i] = 0.5*math.pi*(math.sin(math.pi*w[i]/a))/a
		return w_copy
	
	def w_star_weakly_sparse(self,N,sparse_percentage,how_weak):
		w_star = np.random.randn(N,1)     
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  
		w_star_prox = np.zeros((len(w_star),1))
		for i in range(len(w_star)):
			if w_star[i] == 0:
				w_star_prox[i] = np.random.randn()
		if np.linalg.norm(w_star_prox) > 0:
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
		print("b should be smaller than",1/((eta*lamb)**0.5))
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
		small_eig, big_eig = self.U_eigenvalue(U_all.T@U_all)
		self.lipschitz_checker(U_all,B,m,eta,lamb)
		print(f"convexity condition of mc: b ({B}) should be smaller than ",(small_eig/lamb)**0.5)
		print(f"convexity condition of mc: lambda ({lamb}) should be smaller than ",(small_eig/(B**2)))
		print("rho must be smaller than",small_eig)
		if rho > small_eig:
			print("rho is bigger than the smallest eigen",small_eig,"rho = ",rho)
			exit()
		print(f"distributed approximate mc: {lamb*B**2}<1 should satisfy")
		if lamb*B**2 >= 1:
			exit()
		self.rho_checker(rho,lamb,eta)
		self.distributed_lipschitz_checker(U_all,B,m,eta,N,lamb,(1/2)*(np.eye(len(graph))+graph))
		self.disjoint_checker(graph,m)
	
	def lipschitz_checker(self,U,B,m,eta,lamb):
		X = U@U.T
		U, s, V = np.linalg.svd(X)
		lpz = 2/max(s)
		if lpz < eta:
			print(f"centralized condition: eta must be smaller than {lpz}")
			# exit()
	
	def lipschitz_checker_L1(self,U,m,eta,lamb):
		X = U@U.T
		U, s, V = np.linalg.svd(X)
		lpz = 2/max(s)
		if lpz < eta:
			print(f"L1 eta must be smaller than {lpz/2}")
			exit()
	
	def distributed_lipschitz_checker(self,U,B,m,eta,N,lamb,c):
		for u in U:
			ut = np.reshape(u,(len(u),1)) 
			um = np.reshape(u,(1,len(u)))
			lip =(B**2)*lamb/2+ abs(um@ut-lamb*(B**2)/2)
			U, s, V = np.linalg.svd(c)
			lpz = 2*min(s)/lip
			if lpz[0][0] < eta:
				print(f"distributed condition: eta {eta} must be smaller than {lpz}")

	def centralized_convexity_checker(self,B,lamb,U,N):
		B2 = B*B*np.eye(N)
		x = (1/lamb)*np.dot(U.T,U)-B2
		small_eig =  min(LA.eig(x)[0])
		if small_eig < 0:
			print(f"your smallest eigenvalue is {small_eig}. it is nonconvex.")
			return False
		else:
			# print("your function is centrally convex. go fuck yourself")
			return True

	def distributed_convexity_checker(self,B,lamb,U,N):
		for u in U:
			B2 = B*B*np.eye(N)
			ut = np.reshape(u,[N,1])
			u = np.reshape(u,[1,N])
			x = (N/lamb)*np.dot(ut,u)-B2
			small_eig =  min(LA.eig(x)[0])
			if small_eig < 0:
				print(f"your smallest eigenvalue is {small_eig}. it is nonconvex.")
				pass
			else:
				print("your function is convex. go fuck yourself")
				exit()
		print("your distributed function is nonconvex")

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
	
	def pg_extra_step_size(self,Ut,M_tilde):
		L_max = 0
		for u in Ut:
			ut = np.reshape(u,(len(u),1)) 
			um = np.reshape(u,(1,len(u)))
			lip = abs(um@ut)
			if lip > L_max:
				L_max  = lip[0][0]
		small_eig, big_eig = self.U_eigenvalue(M_tilde)
		return 2*small_eig/L_max
	
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
	
	def 	_range(self,m):
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
		elif sum(graph[place]) == r_i + 1:
			result = False
		else:
			print(graph)
			exit()
		return result

	def ring_graph(self,m,r_i):
		graph = np.eye(m)
		for i in range(m):
			for j in range(int(r_i/2)):
				if i+j+1 < m:
					graph[i][i+j+1] = 1
					graph[i+j+1][i] = 1
				else:
					graph[i][i+j+1-m] = 1
					graph[i+j+1-m][i] = 1
		return graph

	def undirected_graph(self,m,r_i):
		r_i += 1
		graph = np.eye(m)
		list_num = list(range(m))
		vertex_dict = dict(zip(list_num,[{i} for i in list_num]))
		i = 0
		while i < r_i:
			for edge_num in range(len(graph)):
				count = 0
				while len(vertex_dict[edge_num]) < i + 1:
					rand = random.randint(0,m-1)
					if rand != edge_num and rand not in vertex_dict[edge_num] and len(vertex_dict[rand]) < r_i:
						vertex_dict[edge_num].add(rand)
						vertex_dict[rand].add(edge_num)
					count += 1
					if count > 500:
						vertex_dict = dict(zip(list_num,[{i} for i in list_num]))
						i = -1
			i += 1
		for i in vertex_dict:
			for j in vertex_dict[i]:
				graph[i][j] = 1
		return graph
	
	def undirected_graph_new(self,m,r_i):
		graph = np.eye(m)
		for i in range(m):
			for j in range(int(r_i/2)):
				if i + j+1 < m:
					graph[i][i+j+1] = 1
					graph[i+j+1][i] = 1
				else:
					graph[i][j+i-m+1] = 1
					graph[j+i-m+1][i] = 1
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
			return False
		else:
			print("Your graph is disjoint. connected nodes:",now_length)
			return True

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
	
	def make_w(self,m,N,w_zero):
		if w_zero:
			w = np.zeros((N,1)).T
			for i in range(m-1):
				w = np.concatenate((w,np.zeros((N,1)).T))
		else:
			w = np.random.randn(N,1).T
			for i in range(m-1):
				w = np.concatenate((w,np.random.randn(N,1).T))
		return w

	def error_distributed(self,w_all,w_star,N,L2,m):
		each_errors = []
		for w in w_all:
			w = np.reshape(w,(N,1))
			each_errors.append(((np.reshape(w-w_star,(1,N)))@(w-w_star))[0][0])
		try:
			this_error = self.db(mean(each_errors),L2)
			return this_error
		except:
			print("error in system mismatch")
			print(w_all)
			return 0
	
	def error_distributed_2(self,w_all,w_star,N,L2,m):
		each_errors = []
		for w in w_all:
			w = np.reshape(w,(N,1))
			each_errors.append(self.db(((np.reshape(w-w_star,(1,N)))@(w-w_star))[0][0],L2))
		this_error = mean(each_errors)
		return this_error

	def error_consensus(self,w_all,N,L2,m,c_2):
		error = w_all - c_2@w_all
		this_error = self.db(np.linalg.norm(error, ord=2)**2,L2)
		return this_error

	def make_variables(self,N,m,sparsity_percentage,how_weakly_sparse,w_noise):
		w = np.random.randn(N,1)
		w_star = self.w_star_weakly_sparse(N,sparsity_percentage,how_weakly_sparse)
		#w_star = self.w_star(N,sparsity_percentage)
		U_all = np.random.randn(m,N)
		w_star_noise = w_star +np.random.randn(N,1)*(10**-(w_noise/10))
		#w_star_noise = w_star_noise/np.dot(w_star_noise.T,w_star_noise)
		d_all = np.dot(U_all,w_star_noise)
		L2 = np.dot(w_star.T,w_star)

		return w,w_star,U_all,d_all,L2
	
	def make_variables_2(self,N,m,r_i,sparsity_percentage,how_weakly_sparse,w_noise):
		w = np.random.randn(N,1)
		w_star = self.w_star(N,sparsity_percentage)
		U_all = np.random.randn(m,N)
		d_all = np.dot(U_all,w_star)
		L2 = np.dot(w_star.T,w_star)[0][0]
		graph = self.undirected_graph_new(m,r_i)
		w_all = self.make_w(m,N)
		return w,w_star,w_all,U_all,d_all,L2,graph

	def make_variables_noise_after(self,N,m,r_i,sparsity_percentage,how_weakly_sparse,w_noise):
		w = np.random.randn(N,1)
		w_star = self.w_star_weakly_sparse(N,sparsity_percentage,how_weakly_sparse)
		U_all = np.random.randn(m,N)
		d_all = np.dot(U_all,w_star)
		variance_n = np.var(d_all)
		variance_s = variance_n*10**(-w_noise/10)
		d_all += np.random.normal(loc= 0,scale = variance_s**(0.5),size = (m,1))
		L2 = np.dot(w_star.T,w_star)[0][0]
		graph = self.undirected_graph(m,r_i)
		w_all = self.make_w(m,N,False)
		return w,w_star,w_all,U_all,d_all,L2,graph
	
	def make_variables_noise_after_2(self,N,m,r_i,sparsity_percentage,how_weakly_sparse,w_noise,normal_distribution,w_zero):
		if w_zero:
			w = np.zeros((N,1))
		else:
			w = np.random.randn(N,1)
		w_star = self.w_star_weakly_sparse(N,sparsity_percentage,how_weakly_sparse)
		if normal_distribution:
			U_all = np.random.randn(m,N)
		else:
			random_matrix = np.random.rand(m,N)
			standard_deviation = np.std(random_matrix)
			average_random_matrix = np.average(random_matrix)
			U_all_not_standard = random_matrix-average_random_matrix*np.ones((m,N))
			U_all = U_all_not_standard/standard_deviation
		d_all = np.dot(U_all,w_star)
		if w_noise != 0:
			variance_n = np.var(d_all)
			variance_s = variance_n*10**(-w_noise/10)
			d_all += np.random.normal(loc= 0,scale = variance_s**(0.5),size = (m,1))
		L2 = np.dot(w_star.T,w_star)[0][0]
		graph_flag = True
		i = 0
		while graph_flag:
			graph = self.undirected_graph(m,r_i)
			graph_flag = self.disjoint_checker(graph,m)
			i += 1
		w_all = self.make_w(m,N,w_zero)
		return w,w_star,w_all,U_all,d_all,L2,graph
	
	def make_variables_noise_after_2_deviated_average(self,N,m,r_i,sparsity_percentage,how_weakly_sparse,w_noise,normal_distribution,w_zero,average_of_u):
		if w_zero:
			w = np.zeros((N,1))
		else:
			w = np.random.randn(N,1)
		w_star = self.w_star_weakly_sparse(N,sparsity_percentage,how_weakly_sparse)
		if normal_distribution:
			U_all = np.random.randn(m,N)
		else:
			random_matrix = np.random.rand(m,N)
			standard_deviation = np.std(random_matrix)
			U_all_not_standard = random_matrix/(standard_deviation) 
			average_random_matrix = np.average(U_all_not_standard)
			U_all = U_all_not_standard - (average_random_matrix-average_of_u)*np.ones((m,N))
			average_random_matrix = np.average(U_all)
			standard_deviation = np.std(U_all)
			print(standard_deviation,average_random_matrix)
		d_all = np.dot(U_all,w_star)
		if w_noise != 0:
			variance_n = np.var(d_all)
			variance_s = variance_n*10**(-w_noise/10)
			d_all += np.random.normal(loc= 0,scale = variance_s**(0.5),size = (m,1))
		L2 = np.dot(w_star.T,w_star)[0][0]
		graph_flag = True
		while graph_flag:
			graph = self.undirected_graph(m,r_i)
			graph_flag = self.disjoint_checker(graph,m)
		w_all = self.make_w(m,N,w_zero)
		return w,w_star,w_all,U_all,d_all,L2,graph
	
	def show_graph(self,graph,r_i):
		graph_edge = []
		graph_node = []
		print(len(graph))
		for i in range(len(graph)):
			graph_node.append(str(i+1))
			j = copy.deepcopy(i) + 1
			while j < len(graph):
				if graph[i][j] == 1:
					graph_edge.append((str(i+1),str(j+1)))
				j += 1
		G = nx.Graph()
		G.add_nodes_from(graph_node)
		G.add_edges_from(graph_edge)
		pos = nx.spring_layout(G,k = 5,seed=1)
		nx.draw_networkx_edges(G, pos, edge_color='y')
		nx.draw_networkx_nodes(G, pos, node_color='r', alpha=0.5)
		nx.draw_networkx_labels(G, pos, font_size=10)
		plt.axis('off')

	def make_variables_noise_after_3(self,N,m,r_i,sparsity_percentage,how_weakly_sparse,w_noise,normal_distribution,w_zero):
		if w_zero:
			w = np.zeros((N,1))
		else:
			w = np.random.randn(N,1)
		w_star = self.w_star_weakly_sparse(N,sparsity_percentage,how_weakly_sparse)
		if normal_distribution:
			U_all = np.random.randn(m,N)
		else:
			U_all = np.random.rand(m,N)
		d_all = np.dot(U_all,w_star)
		variance_n = np.var(d_all)
		variance_s = variance_n*10**(-w_noise/10)
		d_all += np.random.normal(loc= 0,scale = variance_s**(0.5),size = (m,1))
		L2 = np.dot(w_star.T,w_star)[0][0]
		graph = self.ring_graph(m,r_i)
		w_all = self.make_w(m,N,w_zero)
		return w,w_star,w_all,U_all,d_all,L2,graph

	def make_variables_no_noise(self,N,m,r_i,sparsity_percentage,how_weakly_sparse,w_noise):
		w = np.random.randn(N,1)
		w_star = self.w_star(N,sparsity_percentage)
		U_all = np.random.randn(m,N)
		d_all = np.dot(U_all,w_star)
		L2 = np.dot(w_star.T,w_star)[0][0]
		graph = self.undirected_graph(m,r_i)
		w_all = self.make_w(m,N)
		return w,w_star,w_all,U_all,d_all,L2,graph

	def one_gradient_descent(self,Ut,d,w,eta):
		U = Ut.T
		return ((U*(np.dot(Ut,w)-d)))

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

	def distributed_gradient_descent_1(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,w_star,N,L2,m))
			w_all_next = c@w_all_next - eta*self.all_gradient_descent(Ut,w_all_next,d,w_all_iter,eta)
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
	
	