from scipy.stats import spearmanr
import re

roles = ["LOCATION", "TIME", "RECIPIENT", "INSTRUMENT"]
acc_functions = {'simple': _simple_accuracy, 'diff': _accuracy_with_thresh}

def _simple_accuracy(df, group_dict):
	df_res = {}
	for tup, idx in group_dict.items():
		if len(idx) == 2:
			
			if df['typicality'][idx[0]] == 'T':
				if df['probability'][idx[0]] > df['probability'][idx[1]]:
					a = 1
				else:
					a = 0
			else:
				if df['probability'][idx[0]] < df['probability'][idx[1]]:
					a = 1
				else:
					a = 0

			df_res[tup] = {'idxs': idx, 'acc': a}

	return df_res


def _accuracy_with_thresh(df, group_dict):
	diffs = []
	pairs = []
	for tup, idx in group_dict.items():
		if len(idx) == 2:
			if df['typicality'][idx[0]] == 'T':
				a = df['probability'][idx[0]] - df['probability'][idx[1]]
			else:
				a = df['probability'][idx[1]] - df['probability'][idx[0]]
			pairs.append(tup)
			diffs.append(a)
	return pairs, diffs


#def _correlation(df, group_dict):

def rank(df, group_dict, delim='_'):
	#il rango dell’elemento tipico e atipico nella distribuzione 
	#di probabilità (p.es., elemento tipico è decimo elemento 
	#più probabile)
	for tup, idx in group_dict.items():
		if len(idx) == 2:
			fillers = df['fillers'][idx[0]].split(delim)
			if df['typicality'][idx[0]] == 'T':
				diff_rank = fillers.index(df[]) - df['probability'][idx[1]]
			else:
				diff_rank = df['probability'][idx[1]] - df['probability'][idx[0]]
			

def evaluation(data_path, etype, thresh):
	data = pd.read_csv(data_path, sep='\t')
	data_covered = data.dropna()

	if any(c in data.columns for c in roles):
		#{('spy', 'information', 'pass'): [1, 3], ('volunteer', 'food', 'bring'): [0, 2]}
		groups = df.groupby(['SUBJECT', 'VERB', 'OBJECT']).groups
	else:
		groups = df.groupby(['SUBJECT', 'VERB']).groups

	tuples, res = acc_functions[etype](data_covered, groups)
	if etype == 'diff':
		l_func = lambda x: 1 if x > thresh else 0
		accs = [l_func(i) for i in res]
	accuracy = sum(accs)/len(accs)

