import os
import numpy as np
import pandas as pd
import logging

from vector_utils import cosine

logger = logging.getLogger(__name__)


roles = ["LOCATION", "TIME", "RECIPIENT", "INSTRUMENT"]

def tup_to_string(row, columns):
	if any(c in columns for c in roles):
		# OTHER RELATIONS
		target_rel = list(set(columns).intersection(set(roles)))[0]
		item = "{}@N@SBJ {}@V@ROOT {}@N@OBJ".format(row['SUBJECT'], row['VERB'], row['OBJECT'])
	else:
		if 'OBJECT' in data.columns:
			# TRIPLES
			target_rel = 'OBJ'
			item = "{}@N@SBJ {}@V@ROOT".format(row['SUBJECT'], row['VERB'])
		else:
			# PAIRS
			target_rel = 'ROOT'
			item = "{}@N@SBJ".format(row['SUBJECT'])
	return item, target_rel


def prepare_input_file(data_path, out_dir):
	#TODO: change input argument order?
	data = pd.read_csv(data_path, delimiter="\t")
	items = []
	for _, row in data.iterrows():
		item, target_rel = tup_to_string(row, data.columns)
		items.append(item)
	items = list(set(items))

	# write file
	df = pd.DataFrame(data={"item": items, "target-relation": [target_rel for _ in range(0, len(items))]})
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
			d= data_sdm["LC_vector"][idx]#.to_string()
			v_LC = np.array([float(x) for x in data_sdm["LC_vector"][idx].split()])
		except ValueError:
			v_LC = None
		try:
			v_AC = np.array([float(x) for x in data_sdm["AC_vector"][idx].split()])
		except ValueError:
			v_AC = None

		if target_rel == 'ROOT':
			target = data_original['VERB']
		elif target_rel == 'OBJ':
			target = data_original['OBJECT']
		else:
			target = data_original[target_rel]
		try:
			v_target = vecs[(target, 'N')]
			if v_LC is not None and v_AC is not None:
				sim_LC = cosine(v_target, v_LC)
				sim_AC = cosine(v_target, v_AC)
				sum = sim_LC + sim_AC
			else:
				# LC or AC vector are None
				if sim_LC is not None:
					sum = sim_LC
				else:
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
		# write scores
		data_original['LC_sim'] = sims_LC
		data_original['AC_sim'] = sims_AC
		data_original['computed_score'] = scores
		out_path = os.path.join(out_dir, os.path.basename(data_path).split('.')[0] + '.sdm-res')
		data_original.to_csv(out_path, sep='\t', index=False)

if __name__ == '__main__':
	compute_sdm_function('../../datasets/DTFit/original/TypicalityRatings_Instr.txt',
	                     '../../datasets/DTFit/sdm_result/TypicalityRatings_Instr.sdm.out',
	                     '../../datasets/DTFit/sdm_result/',
	                     '/home/giulia.rambelli/to_backup/spaces/wiki-news-300d-1M.vec')
	#prepare_input_file('/home/giulia.rambelli/transformers_thematic_fit/datasets/DTFit/original/TypicalityRatings_Loc.txt', '/home/giulia.rambelli/transformers_thematic_fit/datasets/DTFit/sdm/')