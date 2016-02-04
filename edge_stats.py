import brain_graphs
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def edge_stats(task):
	# membership
	known_membership = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[31].values)
	known_membership[known_membership==-1] = 0
	names = np.array(pd.read_csv('/home/despoB/mb3152/modularity/Consensus264.csv',header=None)[36].values)

	ec_file = '/home/despoB/arjun/{t}_RL-edges.npz'.format(t=task)
	d = np.load(ec_file)
	ecs = d['ecs']
	cond_order = d['cond_order']
	other_ecs = d['other_ecs']
	tps = d['tps']
	nsub = ecs.shape[0]
	ncond = ecs.shape[1]
	print ecs.shape # shape = n_sub x n_cond x n_regions x 2 (within|between)
	
	
	print cond_order
	print other_ecs, other_ecs.shape
	print tps, tps.shape
	for s in range(nsub): # we want to compute w/b for the specific modules we care about... these will be different in every task, and so have to be specified
		if ncond ==2:
			# 1/28: make a matrix of the within and between edge counts for the various communities (values of known_membership)
			c1 = ecs[s,0,:,:]
			c2 = ecs[s,1,:,:]
			c_other = other_ecs[s,:,:]
			#mask = np.logical_and(np.isfinite(c1), np.isfinite(c2))
			print s, 'c1: ', c1.shape, sum(c1), sum(np.isnan(c1)), 'c2: ', c2.shape, sum(np.isnan(c2)), sum(c2), 'c_other: ', c_other.shape, sum(np.isnan(c_other)), sum(c_other)

			#result = ttest_rel(c1[mask],c2[mask])
			#print result

			#for mod in np.unique(known_membership):
			#	print mod

	print known_membership, np.unique(known_membership), known_membership.shape, names
if __name__ == '__main__':
	task=sys.argv[1]
	edge_stats(task)
