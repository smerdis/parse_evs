#!/usr/local/anaconda/bin/python
import os

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
#tasks = ['EMOTION']
subjects = os.listdir('/home/despoB/connectome-data/')
#subjects = ['987983']
for sub in subjects:
	for task in tasks:
		os.system("qsub -l mem_free=18G \
			-N hcp_matrices\
			-j y \
			-o /home/despoB/arjun/parse_evs/sge/ \
			-e /home/despoB/arjun/parse_evs/sge/ \
			-V /home/despoB/arjun/parse_evs/parse_evs.py %s %s" %(sub,task))
