#!/home/despoB/mb3152/anaconda/bin/python
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
known_membership[known_membership==-1] = 0
names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)

def analyze_igraph(sub, task, membership=known_membership):
	# build the graphs for a task (one per condition, the correlation matrices should already exist
	gfile_pre = os.path.join('/home/despoB/arjun','graphs', '{s}-{t}-*'.format(s=sub, t=task))
	gfiles = glob.glob(gfile_pre)
	if not gfiles:
		print('No graphs found for {s}-{t}'.format(s=sub,t=task))
		return
	else:
		n_conds = len(gfiles)
		edge_counts = dict()
		for f in gfiles: #for each condition within a subject and task
			cond_suffix = f.split('-')[-1][:-4]
			with np.load(f) as data:
				g = data['graph'].item()
				p = data['partition'].item()
				n_verts = len(g.vs)
				mod_edge_counts = np.zeros(shape=(n_verts,2),dtype=np.int16)
				# this structure will hold the within-module edge counts in column 1 (index 0)
				# between-mod counts are in column 2 (index 1)
				W = 0
				B = 1
				for e in g.es:
					if known_membership[e.source] == known_membership[e.target]: # within-module
						C = W
					else: #between-module edge. update the other column
						C = B
					mod_edge_counts[e.source,C] = mod_edge_counts[e.source,C]+1
					mod_edge_counts[e.target,C] = mod_edge_counts[e.target,C]+1
				totals = mod_edge_counts.sum(axis=0)
				print('{c}: {wm} within, {bm} between'.format(c=cond_suffix, wm=totals[W], bm=totals[B]))
				edge_counts[cond_suffix] = (totals[W], totals[B])
		return edge_counts

# plotting function
def plot_corr_matrix(matrix, membership=known_membership):
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
	sns.heatmap(corr_mat,square=True,xticklabels=x_names,yticklabels=y_names,linewidths=0.0)
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
	plt.close()

def wb_edges(g, known_membership, W, B):
	n_verts = len(g.vs)
	# this structure will hold the within-module edge counts in column 1 (index 0)
	# between-mod counts are in column 2 (index 1)
	mod_edge_counts = np.zeros(shape=(n_verts,2),dtype=np.int16)
	for e in g.es:
		if known_membership[e.source] == known_membership[e.target]: # within-module
			C = W
		else: #between-module edge. update the other column
			C = B
		mod_edge_counts[e.source,C] = mod_edge_counts[e.source,C]+1
		mod_edge_counts[e.target,C] = mod_edge_counts[e.target,C]+1
	return mod_edge_counts	

def edge_vec(g):
	from itertools import combinations
	from scipy.misc import comb

	n_rois = len(g.vs)
	n_poss_edges = int(comb(n_rois, 2))
	evec = np.zeros(shape=n_poss_edges)

	for i,x in enumerate(combinations(xrange(264),2)):
		if g.get_eid(x[0], x[1], directed=False, error=False):
			evec[i] = 1

	return evec.T

def select_edges(subjects, task, known_membership=known_membership):
	# this function is only meant to be run via sge
	# use a classifier to differentiate the 'task engaged' and 'other' conditions, then return the most informative edges
	from scipy.misc import comb
	from itertools import combinations
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import cross_val_score
	
	# only process tasks with 2 opposed conditions, record which is 'on' here
	ok_tasks = {'EMOTION':'fear', 'RELATIONAL':'relation', 'WM':'2bk', 'SOCIAL':'social', 'LANGUAGE':'story'}
	if task in ok_tasks.keys():
		n_conds = 2
		on_cond = ok_tasks[task]
	else: #unknown task?
		raise Error("only handling tasks with 'on' and 'control' conditions: %s"%ok_tasks)

	n_subs = len(subjects)
	n_rois = len(known_membership)
	n_poss_edges = int(comb(n_rois, 2)) # 264 choose 2, no replacement allowed. gives 0,1 -> 262,263 in that order. 

	print('Called select_edges(), {ns} subs, task: {t}'.format(ns=n_subs,t=task))

	edge_counts = np.zeros(shape=(n_subs, n_poss_edges)) # n_subs x n_poss_edges because we want to have n_samples x n_features to classify 
	other_edge_counts = np.zeros(shape=(n_subs*10, n_poss_edges)) # there will be more examples from the other tasks, so preallocating big
	n_other_egs = 0 # will keep track of how many examples from the other condition there will be

	for i, sub in enumerate(subjects):
		gfile_pre = os.path.join('/home/despoB/arjun','graphs', '{s}-{t}-*'.format(s=sub, t=task))
		gfiles = glob.glob(gfile_pre)
		if not gfiles:
			print('No graphs found for {s}-{t}'.format(s=sub,t=task))
			continue
		elif len(gfiles) != n_conds:
			print('Wrong number of files for {s}-{t}! Found {nf}, should have {nc}.'.format(s=sub,t=task,nf=len(gfiles),nc=n_conds))
			continue
		else: #all the condition graphs exist, so we will use this on_cond graph and the graph culled from all other tasks to do edge selection
			on_cond_file = gfile_pre[:-1] + on_cond + '*'
			ocfile = glob.glob(on_cond_file)
			if len(ocfile) == 1:
				f = ocfile[0]
				with np.load(f) as data:
					g = data['graph'].item()
					edge_counts[i,:] = edge_vec(g)
			else:
				raise IOError('On-condition file %s does not exist or is not unique!'%ocfile)

			# now let's get the data for all the other tasks
			other_task_pre = os.path.join('/home/despoB/arjun','graphs', '{s}-*'.format(s=sub, t=task))
			ofiles = [file for file in glob.glob(other_task_pre) if file not in gfiles]
			for k, f in enumerate(sorted(ofiles)):
				with np.load(f) as data:
					g = data['graph'].item()
					other_edge_counts[n_other_egs,:] = edge_vec(g)
					n_other_egs = n_other_egs + 1
					if n_other_egs == other_edge_counts.shape[0]: # if we're in danger of overrunning the preallocated array, resize
						other_edge_counts.resize(other_edge_counts.shape[0]*2, n_poss_edges)

		print 'finished sub {i}, {n} other examples so far'.format(i=i,n=n_other_egs)

	other_edge_counts = other_edge_counts[~np.all(other_edge_counts == 0, axis=1)] # remove all-zero rows of other_edge_counts - these would be the extra allocated rows we preallocated
	task_target = np.ones(shape=(edge_counts.shape[0],1))
	other_target = np.zeros(shape=(other_edge_counts.shape[0],1))

	# make final structures for classifier, raveled targets and n_samples x n_features
	targets = np.ravel(np.vstack([task_target, other_target]))
	edges = np.vstack([edge_counts, other_edge_counts])
	
	# now train a logit classifier, using the comb(n_rois, 2) possible non-directional edges and their prsence/absence in various observations from each condition
	# to predict which task condition (engaged condition of task of interest and all conditions of all tasks, as a comparison) is engaged. use crossvalidation.
	model = LogisticRegression(C=.1)
	#scores = cross_val_score(model, edges, targets, cv=5)
	#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	# make a nice matrix of edge weights
	ws = np.empty(shape=(n_rois, n_rois), dtype=float)
	# we will refit on the whole data now, assuming xval performance was good
	weights = model.fit(edges, targets).coef_
	# reshape the weights into a half-matrix n_rois x n_rois
	for j, x in enumerate(combinations(xrange(n_rois),2)):
		ws[x[0],x[1]] = weights[0,j]
	
	return ws

def edge_stats(task):
	ec_file = '/home/despoB/arjun/{t}_RL-edges.npz'.format(t=task)
	d = np.load(ec_file)
	ws = d['ws']
	tps = d['tps']
	print ws, ws.shape
	print known_membership, np.unique(known_membership), known_membership.shape, names

def task_performance(subjects,task):
	all_performance = []
	for subject in subjects:
		try:
			df = pd.read_csv('/home/despoB/mb3152/scanner_performance_data/%s_tfMRI_%s_Stats.csv' %(subject,task))
		except:
			all_performance.append(np.nan)
			continue
		if task[:-3] == 'WM':
			performance = np.mean(df['Value'][[24,27,30,33]])
		if task[:-3] == 'RELATIONAL':
			performance = df['Value'][1]
		if task[:-3] == 'LANGUAGE':
			performance = df['Value'][1]
		if task[:-3] == 'SOCIAL':
			performance = np.mean([df['Value'][0],df['Value'][5]])
		all_performance.append(performance)
	return np.array(all_performance)
			
if __name__ == "__main__":
	subjects = os.listdir('/home/despoB/connectome-data/')
	#sub = str(sys.argv[1])
	task=sys.argv[1]
	ws = select_edges(subjects, task[:-3])
	tps = task_performance(subjects, task)
	#print tps
	outfile = '/home/despoB/arjun/{t}-edges'.format(t=task)
	np.savez(outfile, ws=ws, tps=tps)
