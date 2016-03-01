#!/home/despoB/mb3152/anaconda/bin/python
import os, glob, sys

#tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
tasks = ['GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
#tasks = ['EMOTION']
#subjects = ['987983']
print sys.path
for task in tasks:
	task_full = '{t}_RL'.format(t=task)
	print 'counting edges for {t}...'.format(t=task)
	os.system("qsub -l mem_free=18G \
		-N hcp_edges\
		-j y \
		-o /home/despoB/arjun/parse_evs/sge/ \
		-e /home/despoB/arjun/parse_evs/sge/ \
		-V /home/despoB/arjun/parse_evs/analyze_graphs.py %s"%task_full)
	