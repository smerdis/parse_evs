#!/home/despoB/mb3152/anaconda/bin/python
import os, glob, sys

#tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
tasks = ['GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
#tasks = ['EMOTION']
subjects = os.listdir('/home/despoB/connectome-data/')
#subjects = ['987983']
print sys.path
for sub in subjects:
	for task in tasks:
		mat_files = glob.glob('/home/despoB/arjun/{s}-{t}*'.format(s=sub,t=task))
		if mat_files:
			outfile_prefix = os.path.join('/home/despoB/arjun','graphs', '{s}-{t}-*'.format(s=sub, t=task))
			g_files = glob.glob(outfile_prefix)
			if not g_files:
				print 'generating igraph for {s}-{t} from files: {m}'.format(s=sub,t=task,m=mat_files)
				os.system("qsub -l mem_free=18G \
					-N hcp_igraphs\
					-j y \
					-o /home/despoB/arjun/parse_evs/sge/ \
					-e /home/despoB/arjun/parse_evs/sge/ \
					-V /home/despoB/arjun/parse_evs/analyze_matrices.py %s %s" %(sub,task))
			else: print '{s}-{t} completed already ({gf})'.format(s=sub,t=task,gf=g_files)
		else:
			print '{s}-{t} matrix doesnt exist...'.format(s=sub,t=task)
			os.system("qsub -l mem_free=18G \
                                -N hcp_matrices\
                                -j y \
                                -o /home/despoB/arjun/parse_evs/sge/ \
                                -e /home/despoB/arjun/parse_evs/sge/ \
                                -V /home/despoB/arjun/parse_evs/parse_evs.py %s %s" %(sub,task))
