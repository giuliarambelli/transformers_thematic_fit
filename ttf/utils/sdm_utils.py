import logging
import os

import numpy as np
import pandas as pd

from os_utils import get_filenames
from data_utils import load_vectors
from vector_utils import cosine

logger = logging.getLogger(__name__)


roles = ["LOCATION", "TIME", "RECIPIENT", "INSTRUMENT"]

def tup_to_string(row, columns, inverse=False):
	if any(c in columns for c in roles):
		# OTHER RELATIONS
		target_rel = list(set(columns).intersection(set(roles)))[0]
		item = "{}@N@SBJ {}@V@ROOT {}@N@OBJ".format(row['SUBJECT'], row['VERB'], row['OBJECT'])
	else:
		if 'OBJECT' in columns:
			# TRIPLES
			if inverse:
				target_rel = 'SBJ'
				item = "{}@V@ROOT {}@N@OBJ".format(row['VERB'], row['OBJECT'])
			else:
				target_rel = 'OBJ'
				item = "{}@N@SBJ {}@V@ROOT".format(row['SUBJECT'], row['VERB'])
		else:
			# PAIRS
			target_rel = 'ROOT'
			item = "{}@N@SBJ".format(row['SUBJECT'])
	return item, target_rel


def prepare_input_file(data_path, out_dir, sbj=False):
	#TODO: change input argument order?
	data = pd.read_csv(data_path, delimiter="\t")
	items = []
	for _, row in data.iterrows():
		item, target_rel = tup_to_string(row, data.columns, sbj)
		items.append(item)
	items = list(set(items))

	# write file
	df = pd.DataFrame(data={"item": items, "target-relation": [target_rel for _ in range(0, len(items))]})
	if sbj:
		out_path = os.path.join(out_dir, os.path.basename(data_path).split('.')[0] + '_SBJ.sdm')
	else:
		out_path = os.path.join(out_dir, os.path.basename(data_path).split('.')[0]+'.sdm')
	df.to_csv(out_path, sep="\t", index=False)


def compute_sdm_function(data_path, sdm_outpath, out_dir, vecs):
	# load dataset
	data_original = pd.read_csv(data_path, sep='\t')
	# load sdm vecs
	data_sdm = pd.read_csv(sdm_outpath, delimiter="\t")
	groups = data_sdm.groupby(['item']).groups

	sims_LC = []
	sims_AC = []
	scores = []

	for index, row in data_original.iterrows():
		item, target_rel = tup_to_string(row, data_original.columns)
		idx = groups[item][0]
		#print(idx)
		# GET LC and AC vecs for each example in the dataset

		try:
			idx = groups[item][0]
			#d = data_sdm["LC_vector"][idx]#.to_string()
			v_LC = np.array([float(x) for x in data_sdm["LC_vector"][idx].split()])
		except ValueError:
			v_LC = None
		try:
			v_AC = np.array([float(x) for x in data_sdm["AC_vector"][idx].split()])
		except ValueError:
			v_AC = None

		if target_rel == 'ROOT':
			target = row['VERB']
		elif target_rel == 'OBJ':
			target = row['OBJECT']
		elif target_rel == 'SBJ':
			target = row['SUBJECT']
		else:
			target = row[target_rel]
		try:
			v_target = vecs[(target, '_')]
			if v_LC is not None and v_AC is not None:
				sim_LC = cosine(v_target, v_LC)
				sim_AC = cosine(v_target, v_AC)
				sum = sim_LC + sim_AC
			elif v_LC is not None and v_AC is None:
				sim_LC = cosine(v_target, v_LC)
				sim_AC = None
				sum = sim_LC
			else:
				sim_LC = None
				sim_AC = None
				sum = None
			sims_LC.append(sim_LC)
			sims_AC.append(sim_AC)
			scores.append(sum)
		except KeyError:
			# target word in vector space vocabulary
			print(target)
			# sim_res.append('None\tNone\tNone')
			sims_LC.append(None)
			sims_AC.append(None)
			scores.append(None)
	print(len(sims_LC), len(sims_AC), len(scores), data_original.size)
	# write scores
	data_original['LC_sim'] = sims_LC
	data_original['AC_sim'] = sims_AC
	data_original['computed_score'] = scores
	out_path = os.path.join(out_dir, os.path.basename(data_path).split('.')[0] + '.sdm-res')
	data_original.to_csv(out_path, sep='\t', index=False)

