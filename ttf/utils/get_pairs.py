import os
import pandas as pd
from data_utils import get_thematic_role

if __name__ == '__main__':
	dir='../../datasets/DTFit/original'
	files = [os.path.join(dir,f) for f in os.listdir(dir)]
	for f in files:
		df = pd.read_csv(f, sep='\t')
		trole = get_thematic_role(df.columns, 'sbj' in f.lower())
		if trole in ['SUBJECT', 'OBJECT']:
			groups = df.groupby(list(set(['SUBJECT', 'VERB', 'OBJECT']).difference(set([trole])))).groups
		elif trole =='VERB':
			pass
		else:
			groups = df.groupby(['SUBJECT', 'VERB', 'OBJECT']).groups

		resulting_idxs = []
		for idx in groups.values():
			if len(idx)% 2==0:
				m = int(len(idx)/2)
				for i,j in zip(idx[:m], idx[m:]):
					resulting_idxs.append((i,j))
			else:
				print(idx)
				t = [i for i in idx if df['typicality'][i]=='T']
				at = [i for i in idx if df['typicality'][i]=='AT']
				for i, j in zip(t,at):
					resulting_idxs.append((i, j))
		print(f, len(df), len(resulting_idxs))