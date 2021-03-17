import os
import pandas as pd
from pyinflect import getInflection


def to_past(verb):
	return getInflection(verb, tag='VBD')

def to_present_3s(verb):
	return getInflection(verb, tag='VBZ')


def triple_to_sentence(tup, verb_tag):
	sbj, verb, obj = tup
	verb = getInflection(verb, tag=verb_tag)[0]
	s = 'The {} {} the {}'.format(sbj, verb, obj)
	return s


def to_sentences(data_path, out_dir, v_tag):
	data = pd.read_csv(data_path, sep='\t')

	sentences = []

	for index, row in data.iterrows(): 
		sentence = ''
		sentence+= triple_to_sentence((row['SUBJECT'], row['VERB'], row['OBJECT']), v_tag)
		if "LOCATION" in data.columns:
			sentence += ' {} the {}'.format(row['LM'], row['LOCATION'])
		elif "TIME" in data.columns:
			sentence += ' {} the {}'.format(row['LM'], row['TIME'])
		elif "RECIPIENT" in data.columns:
			sentence += ' to the {}'.format(row['RECIPIENT'])
		elif "INSTRUMENT" in data.columns:
			sentence += ' with the {}'.format(row['INSTRUMENT'])
		sentence += ' .'
		sentences.append(sentence)

	data['sentence'] = sentences

	out_path = os.path.join(out_dir, os.path.basename(data_path).split('.txt')[0]+'.sentences')
	data.to_csv(out_path, header=True, index=None, sep='\t', mode='w')

 