from scipy.stats import spearmanr
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

roles = ["LOCATION", "TIME", "RECIPIENT", "INSTRUMENT"]

def _simple_accuracy(df, group_dict):
	pairs = []
	scores = []
	for tup, idx in group_dict.items():
		if len(idx) == 2:
			#print(df.loc[[idx[0]]])
			if df['typicality'][idx[0]] == 'T':
				if df['computed_score'][idx[0]] > df['computed_score'][idx[1]]:
					a = 1
				else:
					a = 0
			else:
				if df['computed_score'][idx[0]] < df['computed_score'][idx[1]]:
					a = 1
				else:
					a = 0
			pairs.append(tup)
			scores.append(a)

	return pairs, scores


def _accuracy_with_thresh(df, group_dict):
	diffs = []
	pairs = []
	for tup, idx in group_dict.items():
		if len(idx) == 2:
			if df['typicality'][idx[0]] == 'T':
				a = df['computed_score'][idx[0]] - df['computed_score'][idx[1]]
			else:
				a = df['computed_score'][idx[1]] - df['computed_score'][idx[0]]
			pairs.append(tup)
			diffs.append(a)
	return pairs, diffs


def _correlation(df, output_location, path_data):
	scores = df['mean_rat']
	probs = df['computed_score']
	print(spearmanr(scores, probs))
	scores_for_regr = np.array(scores).reshape(-1, 1)
	probs_for_regr = np.array(scores).reshape(-1, 1)
	regr = LinearRegression().fit(scores_for_regr, probs_for_regr)
	probs_predicted = regr.predict(scores_for_regr)
	plt.plot(scores_for_regr, probs_for_regr, 'o', color='black')
	plt.plot(scores_for_regr, probs_predicted, color='blue')
	plt.xlabel('human typicality scores')
	plt.ylabel('model probabilities')
	plt.title("Actuals vs Regression Line")
	plt.savefig(os.path.join(output_location,os.path.basename(path_data).split('.')[0]+"_correl.png"))


"""
def rank(df, group_dict, delim=';'):
	#il rango dell’elemento tipico e atipico nella distribuzione 
	#di probabilità (p.es., elemento tipico è decimo elemento 
	#più probabile)
	for tup, idx in group_dict.items():
		if len(idx) == 2:
			fillers = df['best_completions'][idx[0]].split(delim)
			if df['typicality'][idx[0]] == 'T':
				diff_rank = fillers.index(df[]) - df['probability'][idx[1]]
			else:
				diff_rank = df['probability'][idx[1]] - df['probability'][idx[0]]
"""

def evaluation(data_path, etype, thresh, output_plot):
	print("Exec functions..", data_path)
	acc_functions = {'simple': _simple_accuracy, 'diff': _accuracy_with_thresh, 'corr': _correlation}

	data = pd.read_csv(data_path, sep='\t')
	data_covered = data.dropna()

	if etype == 'corr':
		acc_functions[etype](data_covered, output_plot, data_path)
	else:
		if any(c in data.columns for c in roles):
			#{('spy', 'information', 'pass'): [1, 3], ('volunteer', 'food', 'bring'): [0, 2]}
			groups = data_covered.groupby(['SUBJECT', 'VERB', 'OBJECT']).groups
		else:
			# PAIRS
			if 'OBJECT' not in data.columns:
				groups = data_covered.groupby(['SUBJECT']).groups
			# TRIPLES
			else:
				groups = data_covered.groupby(['SUBJECT', 'VERB']).groups

		tuples, accs = acc_functions[etype](data_covered, groups)
		if etype == 'diff':
			l_func = lambda x: 1 if x > thresh else 0
			accs = [l_func(i) for i in accs]
		accuracy = sum(accs)/len(accs)
		print(accuracy)

if __name__ == '__main__':
   evaluation('../../datasets/DTFit/sdm_result/TypicalityRatings_Triples.sdm-res', 'corr', 0.1, '../../')
