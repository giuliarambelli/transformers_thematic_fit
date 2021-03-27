import os
import pandas as pd
from data_utils import get_thematic_role


def get_common_indexes(dir_sdm, dir_transf, outdir):
	files_names = set([f.split('.')[0] for f in os.listdir(dir_sdm) if os.path.isfile(os.path.join(dir_sdm,f))])
	# WARNING: it does not deal with subdirectories!
	for fname in sorted(files_names):
		print(fname)
		# get files for the same dataset
		paths = [os.path.join(dir_transf, f) for f in os.listdir(dir_transf) if f.startswith(fname+'.')] + \
		        [os.path.join(dir_sdm, f) for f in os.listdir(dir_sdm) if  f.startswith(fname+'.')]
		if len(paths)==5:
			# get indices covered by each model
			covered_idxs = []
			for path in paths:
				covered_idxs.append(list(pd.read_csv(path, sep='\t').dropna(subset=['computed_score']).index))
			common_idxs =  set(covered_idxs[0]).intersection(*covered_idxs)

			#get pair T-AT
			# {('spy', 'information', 'pass'): [1, 3], ('volunteer', 'food', 'bring'): [0, 2]}
			df = pd.read_csv(paths[0], sep='\t')
			trole = get_thematic_role(df.columns, 'sbj' in fname.lower())
			if trole in ['SUBJECT', 'OBJECT']:
				groups = df.groupby(list(set(['SUBJECT', 'VERB', 'OBJECT']).difference(set([trole])))).groups
			else:
				groups = df.groupby(['SUBJECT', 'VERB', 'OBJECT']).groups

			resulting_idxs = []
			for idx in groups.values():
				if len(idx)==2:
					if idx[0] in common_idxs and idx[1] in common_idxs:
						resulting_idxs.extend(idx)
				else:
					if any(j in common_idxs for j in idx[:2]) and any(j in common_idxs for j in idx[2:]):
				        #if all(j in common_idxs for j in idx):
						resulting_idxs.extend(idx)
			resulting_idxs = sorted(resulting_idxs)
			#for i in resulting_idxs:
			#	print(i)
			print(fname, len(resulting_idxs))

if __name__ == '__main__':
	#get_common_indexes('/home/giulia.rambelli/transformers_thematic_fit/datasets/DTFit/sdm_result/',
	#                   '/home/giulia.rambelli/transformers_thematic_fit/datasets/DTFit/transformers/',
	#                   '.')
	get_common_indexes('/home/giulia.rambelli/transformers_thematic_fit/datasets/semantic_plausibility_data/sdm_result/',
                   '/home/giulia.rambelli/transformers_thematic_fit/datasets/semantic_plausibility_data/transformers/',
                   '.')