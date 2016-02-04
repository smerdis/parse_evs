#!/usr/local/anaconda/bin/python
import os, glob

#tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
tasks = ['GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
#tasks = ['EMOTION']
subjects = os.listdir('/home/despoB/connectome-data/')
#subjects = ['100307']
for sub in subjects:
	for task in tasks:
		if glob.glob('/home/despoB/arjun/{s}-{t}*'.format(s=sub,t=task)):
			#print 'skipping {s}-{t} as it exists already...'.format(s=sub,t=task)
			continue
		else:
			print '{s}-{t}...'.format(s=sub,t=task)
			os.system("qsub -l mem_free=18G \
				-N hcp_matrices\
				-j y \
				-o /home/despoB/arjun/parse_evs/sge/ \
				-e /home/despoB/arjun/parse_evs/sge/ \
				-V /home/despoB/arjun/parse_evs/parse_evs.py %s %s" %(sub,task))
