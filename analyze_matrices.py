import os, sys, glob
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pylab as plt
import brain_graphs

matrix = '/home/despoB/arjun/100307-RELATIONAL-match.npy'
known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
known_membership[known_membership==-1] = 0
names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)

def plot_corr_matrix(matrix=np.load(matrix), membership=known_membership):
	swap_dict = {}
	index = 0
	corr_mat = np.zeros((matrix.shape))
	new_names = []
	x_ticks = []
	y_ticks = []
	for i in np.unique(membership):
		for node in np.where(membership==i)[0]:
			swap_dict[node] = index
			index = index + 1
			new_names.append(names[node])
	y_names = []
	x_names = []
	old_name = 0
	for i,name in enumerate(new_names):
		if name == old_name:
			continue
		old_name = name
		y_ticks.append(i)
		x_ticks.append(len(new_names)-i)
		y_names.append(name)
		x_names.append(name)
	for i in range(len(swap_dict)):
		for j in range(len(swap_dict)):
			corr_mat[swap_dict[i],swap_dict[j]] = matrix[i,j]
			corr_mat[swap_dict[j],swap_dict[i]] = matrix[j,i]
	membership.sort()
	sns.set(context="paper", font="monospace")
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(12, 9))
	# Draw the heatmap using seaborn
	y_names.reverse()
	sns.heatmap(corr_mat,square=True,yticklabels=y_names,xticklabels=x_names,linewidths=0.0,)
	ax.set_yticks(x_ticks)
	ax.set_xticks(y_ticks)
	# Use matplotlib directly to emphasize known networks
	networks = membership
	for i, network in enumerate(networks):
		if network != networks[i - 1]:
			ax.axhline(len(networks) - i, c='black',linewidth=2)
			ax.axvline(i, c='black',linewidth=2)
	f.tight_layout()
	plt.show()

def build_igraph(sub, task, membership=known_membership):
	# build the graphs for a task (one per condition, the correlation matrices should already exist
	mat_files = glob.glob('/home/despoB/arjun/{s}-{t}*'.format(s=sub,t=task))
	if not mat_files:
		print('No correlation matrices found for {s}-{t}'.format(s=sub,t=task))
		return
	else:
		for f in mat_files:
			cond_suffix = f.split('-')[-1]
			matrix = np.load(f)
			g = brain_graphs.matrix_to_igraph(matrix, cost=.1)
			partition = brain_graphs.brain_graph(VertexClustering(g, membership=membership))
			outfile = os.path.join('graphs', '{s}-{t}-{cs}'.format(s=sub, t=task, cs=cond_suffix))
			print outfile
			np.savez(outfile, graph=g, partition=partition)

if __name__ == "__main__":
	sub = str(sys.argv[1])
        task=sys.argv[2]
	build_igraph(sub, task)
