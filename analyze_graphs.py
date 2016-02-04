#!/home/despoB/mb3152/anaconda/bin/python
import brain_graphs
import os, sys, glob
import numpy as np
import pandas as pd

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

def edge_counts(subjects, task, known_membership=known_membership):
	
	from itertools import combinations_with_replacement as combo_wr
	
	if task in ['EMOTION', 'RELATIONAL', 'WM', 'GAMBLING', 'SOCIAL', 'LANGUAGE']:
		n_conds = 2
	elif task=='MOTOR':
		n_conds = 3
	else: #unknown task?
		return

	n_subs = len(subjects)
	n_rois = len(known_membership)
	
	print('Called edge_counts(), {ns} subs, task: {t}'.format(ns=n_subs,t=task))
	edge_counts = np.zeros(shape=(n_subs, n_conds, n_rois, n_rois)) # subs x conds x regions -- the value stored in the z col is (proportion of within-mod edges relative to all edges, w/(w+b))
	other_edge_counts = np.zeros(shape=(n_subs, n_rois, 2)) # subs x rois x within|between - will hold the edge count data for all the other tasks (ignoring conditions)
	for i, sub in enumerate(subjects):
		gfile_pre = os.path.join('/home/despoB/arjun','graphs', '{s}-{t}-*'.format(s=sub, t=task))
		gfiles = glob.glob(gfile_pre)
		if not gfiles:
			print('No graphs found for {s}-{t}'.format(s=sub,t=task))
			continue
		elif len(gfiles) != n_conds:
			print('Wrong number of files for {s}-{t}! Found {nf}, should have {nc}.'.format(s=sub,t=task,nf=len(gfiles),nc=n_conds))
			continue
		else:
			# mods 1/26: also calculate the edge counts for this subject on all the other tasks, save this struct.
			# so we will eventually have edge counts for this task in 'on' condition (engages faculty), the 'off' condition, and all other tasks
			# first let's do this task:
			cond_order = []
			W = 0
			B = 1
			for j, f in enumerate(sorted(gfiles)): #sorted guarantees that the order of conditions will be the same each time (sub-task-cond, with sub and task the same)
				cond_suffix = f.split('-')[-1][:-4]
				cond_order.append(cond_suffix)
				with np.load(f) as data:
					g = data['graph'].item()
					p = data['partition'].item()
					mod_edge_counts = wb_edges(g, known_membership, W, B)
					tot_edges = np.sum(mod_edge_counts,1) # total # edges for each region
					mask = np.where(tot_edges==0)
					print i, j, mask
					prop_within_edges = np.true_divide(mod_edge_counts[:,W],tot_edges)
					edge_counts[i, j, :, :] = mod_edge_counts
			# now let's get the data for all the other tasks
			other_task_pre = os.path.join('/home/despoB/arjun','graphs', '{s}-*'.format(s=sub, t=task))
			ofiles = [file for file in glob.glob(other_task_pre) if file not in gfiles]
			for k, f in enumerate(sorted(ofiles)):
				with np.load(f) as data:
					g = data['graph'].item()
					other_mod_edge_counts = wb_edges(g, known_membership, W, B) # n_verts x 2
					other_edge_counts[i,:,:] = other_edge_counts[i,:,:] + other_mod_edge_counts


	return edge_counts, cond_order, other_edge_counts

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
	ecdata = edge_counts(subjects, task[:-3])
	ecs = ecdata[0]
	cond_order = ecdata[1]
	other_edge_counts = ecdata[2]
	tps = task_performance(subjects, task)
	#print tps
	outfile = '/home/despoB/arjun/{t}-edges'.format(t=task)
	np.savez(outfile, ecs=ecs, cond_order=cond_order, other_ecs=other_edge_counts, tps=tps)
