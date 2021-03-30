import os
import pandas as pd
from pyinflect import getInflection

from ttf.utils.data_utils import get_thematic_role

roles = ["LOCATION", "TIME", "RECIPIENT", "INSTRUMENT"]


def to_past(verb):
	return getInflection(verb, tag='VBD')

def to_present_3s(verb):
	return getInflection(verb, tag='VBZ')

def pair_to_sentence(tup, verb_tag):
	sbj, verb = tup
	verb = getInflection(verb, tag=verb_tag)[0]
	s = 'The {} {}'.format(sbj, verb)
	return s

def triple_to_sentence(tup, verb_tag):
	sbj, verb, obj = tup
	verb = getInflection(verb, tag=verb_tag)[0]
	s = 'The {} {} the {}'.format(sbj, verb, obj)
	return s

def triple_to_sentence_passive(tup, verb_tag):
	sbj, verb, obj = tup
	verb = '{} {}'.format(getInflection('be', tag=verb_tag)[0],getInflection(verb, tag='VBN')[0])
	s = 'The {} {} the {}'.format(obj, verb, sbj)
	return s


def _to_transitive(data, v_tag, role):
	sentences = []
	for index, row in data.iterrows():
		sentence = ''
		if 'OBJECT' not in data.columns:
			sentence += pair_to_sentence((row['SUBJECT'], row['VERB']), v_tag)
		else:
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
	return sentences
	
def _to_passive(data, v_tag, role):
	sentences = []
	for index, row in data.iterrows():
		sentence = ''
		if 'OBJECT' not in data.columns:
			sentence += pair_to_sentence((row['SUBJECT'], row['VERB']), v_tag)
		else:
			sentence+= triple_to_sentence_passive((row['SUBJECT'], row['VERB'], row['OBJECT']), v_tag)
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
	return sentences


def _to_cleft(data, v_tag, role):
	sentences = []
	for index, row in data.iterrows():
		sentence = ''
		#if 'OBJECT' not in data.columns:
		#	sentence += pair_to_sentence((row['SUBJECT'], row['VERB']), v_tag)
		be_v = getInflection('be', tag=v_tag)[0]
		verb = getInflection(row['VERB'], tag=v_tag)[0]
		if role=='SUBJECT':
			# It was the reporter who held the camera
			sentence = 'It {} the {} that {} the {}'.format(be_v, row['SUBJECT'], verb, row['OBJECT'])
		elif role =='OBJECT':
			# It was the camera that the reporter held
			sentence = 'It {} the {} that the {} {}'.format(be_v, row['OBJECT'], row['SUBJECT'], verb)
		else:
		#elif any(c in data.columns for c in roles):
			svo = triple_to_sentence((row['SUBJECT'], row['VERB'], row['OBJECT']), v_tag).lower()
			if "LOCATION" in data.columns:
				# It was on the ring that the boxer delivered the punch .
				sentence += 'It {} {} the {} that {}'.format(be_v, row['LM'], row['LOCATION'], svo)
			elif "TIME" in data.columns:
				# It was during the party that the waiter delivered the drink.
				sentence += 'It {} {} the {} that {}'.format(be_v, row['LM'], row['TIME'], svo)
			elif "RECIPIENT" in data.columns:
				# It was to the refugee that the volunteer brought the food
				sentence += 'It {} to the {} that {}'.format(be_v, row['RECIPIENT'], svo)
			elif "INSTRUMENT" in data.columns:
				# It was with the mop that the housemaid washed the floor
				sentence += ' It {} with the {} that {}'.format(be_v, row['INSTRUMENT'], svo)
		sentence += ' .'
		sentences.append(sentence)
	return sentences

def _to_questions(data, v_tag, role):
	sentences = []
	
	for index, row in data.iterrows():
		sentence = ''
		# if 'OBJECT' not in data.columns:
		#	sentence += pair_to_sentence((row['SUBJECT'], row['VERB']), v_tag)
		do_v = getInflection('do', tag=v_tag)[0]

		if role=='SUBJECT':
			# Which woman painted the toenail?
			sentence = 'Which {} {} the {}'.format(row['SUBJECT'], getInflection(row['VERB'], tag=v_tag)[0], row['OBJECT'])
		elif role =='OBJECT':
			# Which camera did the reporter hold? .
			sentence = 'Which {} the {} that the {} {}.'.format(row['OBJECT'], do_v, row['SUBJECT'], row['VERB'])
		else:
			svo = triple_to_sentence((row['SUBJECT'], row['VERB'], row['OBJECT']), 'VBZ').lower()
			if "LOCATION" in data.columns:
				#Which ring did the boxer deliver the punch on?
				sentence += 'Which {} did {} {}'.format(row['LOCATION'], svo, row['LM'])
			elif "TIME" in data.columns:
				# During which party did the waiter delivered the drink?
				sentence += '{} which {} did {}'.format(row['LM'], row['TIME'], svo)
			elif "RECIPIENT" in data.columns:
				# Which refugee did the volunteer bring the food to?
				sentence += 'Whhich {} did {} to'.format(row['RECIPIENT'], svo)
			elif "INSTRUMENT" in data.columns:
				# With which mop that the housemaid washed the floor
				sentence += ' With which {} did {}'.format(row['INSTRUMENT'], svo)
		sentence += ' ?'
		sentences.append(sentence)
	return sentences


def to_sentences(data_path, out_dir, v_tag, form):
	data = pd.read_csv(data_path, sep='\t')
	target_role = get_thematic_role(data, data_path)
	data['sentence'] = synform_functions[form](data, v_tag, target_role)

	out_path = os.path.join(out_dir, os.path.basename(data_path).split('.txt')[0]+'.sentences')
	data.to_csv(out_path, header=True, index=None, sep='\t', mode='w')

synform_functions = {'transitive': _to_transitive,
                     'passive': _to_passive,
                     'cleft': _to_cleft,
                     'question':_to_questions}#, 'violation':_to_violation }