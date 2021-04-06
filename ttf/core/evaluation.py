import itertools

from scipy.stats import spearmanr
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
import seaborn as sns

roles = ["LOCATION", "TIME", "RECIPIENT", "INSTRUMENT"]


def _simple_accuracy(df, selected_pairs, path_data, output_location):
	if os.path.basename(path_data).startswith("WangDurrettErk"):
		scores_plaus = []
		scores_implaus = []
		for idx in selected_pairs:
			if df['typicality'][idx[0]] == 'P':
				scores_plaus.append(df['computed_score'][idx[0]])
				scores_implaus.append(df['computed_score'][idx[1]])
			else:
				scores_plaus.append(df['computed_score'][idx[1]])
				scores_implaus.append(df['computed_score'][idx[0]])
		plt.boxplot([scores_plaus, scores_implaus])
		plt.xlabel('plausibility label')
		plt.ylabel('model probabilities')
		plt.title("Plausible vs implausible events")
		plt.savefig(os.path.join(output_location, os.path.basename(path_data) + ".png"))
		plt.close()

	pairs = []
	scores = []
	if path_data.endswith("sdm-res"):
		lc_scores = np.array([tup[1] if not math.isnan(tup[1]) else 0 for tup in df["LC_sim"].iteritems()])
		ac_scores = np.array([tup[1] if not math.isnan(tup[1]) else 0 for tup in df["AC_sim"].iteritems()])
		values_probs = []
		for lc, ac in zip(lc_scores, ac_scores):
			if (lc != 0) and (ac != 0):
				values_probs.append((lc + ac)/2)
			elif (lc == 0) and (ac == 0):
				values_probs.append(0)
			elif lc == 0:
				values_probs.append(ac)
			elif ac == 0:
				values_probs.append(lc)
		probs = pd.Series(data=values_probs, index=df['computed_score'].index)
	else:
		probs = df['computed_score']
	bline_scores = []
	for idx in selected_pairs:
		if len(idx) == 2:
			#print(df.loc[[idx[0]]])
			if df['typicality'][idx[0]] in ['T','P']:
				if probs[idx[0]] > probs[idx[1]]:
					a = 1
				else:
					a = 0
					print(os.path.basename(path_data))
					if (not os.path.basename(path_data).startswith("WangDurrettErk")) and ("sdm-res" in os.path.basename(path_data)):
						print("Error. Sentence typical: {}  Human score: {}   Score assigned: {}".format(df['sentence'][idx[0]], df['mean_rat'][idx[0]], df['computed_score'][idx[0]]))
						print("Sentence atypical: {}  Human score: {}   Score assigned: {}".format(df['sentence'][idx[1]], df['mean_rat'][idx[1]], df['computed_score'][idx[1]]))
						print()
					if (os.path.basename(path_data).startswith("WangDurrettErk")) and ("sdm-res" not in os.path.basename(path_data)):
						print("Error. Sentence typical: {}  Score assigned: {}".format(df['sentence'][idx[0]], df['computed_score'][idx[0]]))
						print("Sentence atypical: {}  Score assigned: {}".format(df['sentence'][idx[1]], df['computed_score'][idx[1]]))
						print()

				if "baseline_score" in list(df.columns):
					if df['baseline_score'][idx[0]]>df['baseline_score'][idx[1]]:
						b = 1
					else:
						b = 0
			else:
				if probs[idx[0]] < probs[idx[1]]:
					a = 1
				else:
					a = 0
					if (not os.path.basename(path_data).startswith("WangDurrettErk")) and (not os.path.basename(path_data).endswith("sdm-res")):
						print("Error. Sentence typical: {}  Human score: {}   Score assigned: {}".format(df['sentence'][idx[1]], df['mean_rat'][idx[1]], df['computed_score'][idx[1]]))
						print("Sentence atypical: {}  Human score: {}   Score assigned: {}".format(df['sentence'][idx[0]], df['mean_rat'][idx[0]], df['computed_score'][idx[0]]))
						print()
					if (os.path.basename(path_data).startswith("WangDurrettErk")) and (not os.path.basename(path_data).endswith("sdm-res")):
						print("Error. Sentence typical: {}  Score assigned: {}".format(df['sentence'][idx[1]], df['computed_score'][idx[1]]))
						print("Sentence atypical: {}  Score assigned: {}".format(df['sentence'][idx[0]], df['computed_score'][idx[0]]))
						print()
				if "baseline_score" in list(df.columns):
					if df['baseline_score'][idx[0]] < df['baseline_score'][idx[1]]:
							b = 1
					else:
						b = 0
			pairs.append(idx)
			scores.append(a)
			if "baseline_score" in list(df.columns):
				bline_scores.append(b)


	return pairs, scores, bline_scores


def _accuracy_with_thresh(df, selected_pairs):
	diffs = []
	pairs = []
	bline_diff = []
	for idx in pairs:
		if len(idx) == 2:
			if df['typicality'][idx[0]] in ['T','P']:
				a = df['computed_score'][idx[0]] - df['computed_score'][idx[1]]
				if "baseline_score" in list(df.columns):
					b = df['baseline_score'][idx[0]] - df['baseline_score'][idx[1]]
			else:
				a = df['computed_score'][idx[1]] - df['computed_score'][idx[0]]
				if "baseline_score" in list(df.columns):
					b = df['baseline_score'][idx[1]] - df['baseline_score'][idx[0]]
			pairs.append(idx)
			diffs.append(a)
			if "baseline_score" in list(df.columns):
				bline_diff.append(b)

	return pairs, diffs, bline_diff


def _correlation(df, output_location, path_data):
	scores = df['mean_rat']
	if path_data.endswith("sdm-res"):
		lc_scores = np.array([tup[1] if not math.isnan(tup[1]) else 0 for tup in pd.to_numeric(df["LC_sim"], errors='coerce').iteritems()])
		ac_scores = np.array([tup[1] if not math.isnan(tup[1]) else 0 for tup in pd.to_numeric(df["AC_sim"], errors='coerce').iteritems()])
		probs = []
		for score in lc_scores:
			print(score, type(score))
		for score in ac_scores:
			print(score, type(score))
		for lc, ac in zip(lc_scores, ac_scores):
			if (lc != 0) and (ac!= 0):
				probs.append((lc + ac)/2)
			elif (lc == 0) and (ac == 0):
				probs.append(0)
			elif lc == 0:
				probs.append(ac)
			elif ac == 0:
				probs.append(lc)
	else:
		probs = df['computed_score']
	print("Model:  ", spearmanr(scores, probs))
	if "baseline_score" in list(df.columns):
		bline_probs = df['baseline_score']
		print("Baseline:  ", spearmanr(scores, bline_probs))
	scores_for_regr = np.log(np.array(scores)).reshape(-1, 1)
	probs_for_regr = np.log(np.array(probs)).reshape(-1, 1)
	regr = LinearRegression().fit(scores_for_regr, probs_for_regr)
	probs_predicted = regr.predict(scores_for_regr)
	#----Analysis of errors
	probs_normalized = (np.array(np.log(probs)) - np.amin(np.array(np.log(probs)))) / (np.amax(np.array(np.log(probs))) - np.amin(np.array(np.log(probs))))
	print("Probs normalized: ", probs_normalized)
	probs_normalized_for_regr = np.array(probs_normalized).reshape(-1, 1)
	regr = LinearRegression().fit(scores_for_regr, probs_normalized_for_regr)
	probs_predicted = regr.predict(scores_for_regr)
	residuals = []
	for i in range(len(df)):
		residuals.append(abs(probs_predicted[i][0]) - probs_normalized_for_regr[i][0])
	print("Sum of residuals: ", np.sum(residuals))





	"""
	if "transformers" in os.path.basename(path_data):
		labels = df['typicality']
		residuals = {}
		for n_item in range(len(df)):
			residuals[list(df["sentence"])[n_item]] = (probs_for_regr[n_item][0] - probs_predicted[n_item][0], list(labels)[n_item], list(scores)[n_item], probs_predicted[n_item][0], probs_for_regr[n_item][0])
		residuals_positive = dict(sorted(residuals.items(), key=lambda x: x[1][0], reverse=True))
		residuals_negative = dict(sorted(residuals.items(), key=lambda x: x[1][0]))
		print("Positive residuals")
		print()
		for item in list(residuals_positive.keys())[0:25]:
			print("Sentence: {}    Human score: {}   Label: {}   Prob predicted: {}  Prob assigned: {}".format(item, residuals_positive[item][2], residuals_positive[item][1], residuals_positive[item][3], residuals_positive[item][4]))
		print()
		print()
		print("Negative residuals")
		print()
		for item in list(residuals_negative.keys())[0:25]:
			print("Sentence: {}    Human score: {}   Label: {}   Prob predicted: {}  Prob assigned: {}".format(item, residuals_negative[item][2], residuals_negative[item][1], residuals_negative[item][3], residuals_negative[item][4]))
		print()
		print()
	"""

	plt.plot(scores_for_regr, probs_for_regr, 'o', color='black')
	plt.plot(scores_for_regr, probs_predicted, color='blue')
	plt.xlabel('human typicality scores')
	plt.ylabel('model probabilities')
	plt.title("Actuals vs Regression Line")
	plt.savefig(os.path.join(output_location,os.path.basename(path_data)+".correl.png"))
	plt.close()
	if "baseline_score" in list(df.columns):
		scores_for_regr = np.log(np.array(scores)).reshape(-1, 1)
		b_probs_for_regr = np.log(np.array(bline_probs)).reshape(-1, 1)
		regr = LinearRegression().fit(scores_for_regr, b_probs_for_regr)
		b_probs_predicted = regr.predict(scores_for_regr)
		plt.plot(scores_for_regr, b_probs_for_regr, 'o', color='black')
		plt.plot(scores_for_regr, b_probs_predicted, color='blue')
		plt.xlabel('human typicality scores')
		plt.ylabel('model probabilities')
		plt.title("Actuals vs Regression Line")
		plt.savefig(os.path.join(output_location, os.path.basename(path_data) + "baseline" + ".correl.png"))
		plt.close()


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

def evaluation(data_path, etype, thresh, output_plot, selected_idxs):
	#todo: gestire caso in cui non si passa lista coppie
	print("Exec functions..", data_path)
	acc_functions = {'simple': _simple_accuracy, 'diff': _accuracy_with_thresh, 'corr': _correlation}

	data = pd.read_csv(data_path, sep='\t')
	data_covered = data[data.index.isin(list(itertools.chain(*selected_idxs)))]
	print("Coverage: {}/{}".format(len(data_covered), len(data)))
	if etype == 'corr':
		acc_functions[etype](data_covered, output_plot, data_path)
	else:
		tuples, accs, bline_accs = acc_functions[etype](data_covered, selected_idxs, data_path, output_plot)
		if etype == 'diff':
			l_func = lambda x: 1 if x > thresh else 0
			accs = [l_func(i) for i in accs]
		accuracy = sum(accs) / len(accs)
		print('Accuracy: {}'.format(accuracy))
		if "baseline_score" in list(data.columns):
			bline_accuracy = sum(bline_accs) / len(bline_accs)
			print('Baseline accuracy: {}'.format(bline_accuracy))
	"""
	if selected_idxs:
		not_given = set([i for i in data.index]).difference(set(selected_idxs))
		data_covered = data.drop(list(not_given))
	else:
		data_covered = data.dropna(subset=['computed_score'])
	print("Coverage: {}/{}".format(len(data_covered),len(data)))
	
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
				if 'sbj' in os.path.basename(data_path).lower():
					groups = data_covered.groupby(['VERB','SUBJECT']).groups

		tuples, accs, bline_accs = acc_functions[etype](data_covered, groups)
		if etype == 'diff':
			l_func = lambda x: 1 if x > thresh else 0
			accs = [l_func(i) for i in accs]
		accuracy = sum(accs)/len(accs)
		print('Accuracy: {}'.format(accuracy))
		if "baseline_score" in list(data.columns):
			bline_accuracy = sum(bline_accs)/len(bline_accs)
			print('Baseline accuracy: {}'.format(bline_accuracy))
	"""
if __name__ == '__main__':
	from ast import literal_eval
	common_idx = pd.read_csv('/home/giulia.rambelli/transformers_thematic_fit/ttf/utils/dtfit_common_pairs.txt', sep='\t')
	i = common_idx.loc[common_idx['name'] == 'TypicalityRatings_Instr'].index
	pairs = literal_eval(common_idx['pairs'][i].values[0])

	evaluation('../../datasets/DTFit/sdm_result/TypicalityRatings_Instr.sdm-res', 'simple', 0, '../../', pairs)
