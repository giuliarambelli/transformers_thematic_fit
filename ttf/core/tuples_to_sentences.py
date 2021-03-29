import os
import pandas as pd
from pyinflect import getInflection

roles = ["LOCATION", "TIME", "RECIPIENT", "INSTRUMENT"]


def to_past(verb):
	return getInflection(verb, tag='VBD')

def to_present_3s(verb):
	return getInflection(verb, tag='VBZ')

def triple_to_sentence_passive(tup, verb_tag):
	sbj, verb, obj = tup
	verb = '{} {}'.format( getInflection(verb, tag=verb_tag)[0],getInflection(verb, tag='VBN')[0])
	s = 'The {} {} the {}'.format(obj, verb, sbj)
	return s

#It is the award that the actor won
# it is the X that the gardner cut the grass with



def triple_to_sentence(tup, verb_tag):
	sbj, verb, obj = tup
	verb = getInflection(verb, tag=verb_tag)[0]
	s = 'The {} {} the {}'.format(sbj, verb, obj)
	return s


def pair_to_sentence(tup, verb_tag):
	sbj, verb = tup
	verb = getInflection(verb, tag=verb_tag)[0]
	s = 'The {} {}'.format(sbj, verb)
	return s


def _to_transitive(data, v_tag, sbj):
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


def _to_cleft(data, v_tag, sbj):
	sentences = []
	for index, row in data.iterrows():
		sentence = ''
		#if 'OBJECT' not in data.columns:
		#	sentence += pair_to_sentence((row['SUBJECT'], row['VERB']), v_tag)

		if sbj:
			verb = getInflection(verb, tag=v_tag)[0]
			sentence = 'It is the {} that {} the {} .'.format(row['SUBJECT'], verb, row['OBJECT'])
		elif any(c in data.columns for c in roles):
			svo = triple_to_sentence((row['SUBJECT'], row['VERB'], row['OBJECT']), v_tag).lower()
			if "LOCATION" in data.columns:
				sentence += 'It is the {} that {} {}'.format(row['LOCATION'], svo, row['LM'])
			elif "TIME" in data.columns:
				sentence += ' {} the {}'.format(row['LM'], row['TIME'])
			elif "RECIPIENT" in data.columns:
				sentence += 'It is the {} {} to. '.format(row['RECIPIENT'], svo)
			elif "INSTRUMENT" in data.columns:
				sentence += ' It is the {} {} with .'.format(row['INSTRUMENT'], svo)
		else:
			verb = getInflection(verb, tag=v_tag)[0]
			sentence += 'It is the {} that the {} {} .'.format(row['OBJECT'], verb, row['SUBJECT'])
		sentence += ' .'
		sentences.append(sentence)
	return sentences

def _to_questions(data, v_tag, sbj):
	sentences = []
	"""
	for index, row in data.iterrows():
		sentence = ''
		# if 'OBJECT' not in data.columns:
		#	sentence += pair_to_sentence((row['SUBJECT'], row['VERB']), v_tag)

		if sbj:
			#verb = getInflection(verb, tag=v_tag)[0]
			#sentence = 'It is the {} that {} the {} .'.format(row['SUBJECT'], verb, row['OBJECT'])
		elif any(c in data.columns for c in roles):
			svo = 'did the {} {} the {}'.format(row['SUBJECT'], row['VERB'], row['OBJECT'])
			if "LOCATION" in data.columns:
				sentence += 'Which {} that {} {}'.format(row['LOCATION'], svo, row['LM'])
			elif "TIME" in data.columns:
				sentence += ' {} the {}'.format(row['LM'], row['TIME'])
			elif "RECIPIENT" in data.columns:
				sentence += 'It is the {} {} to. '.format(row['RECIPIENT'], svo)
			elif "INSTRUMENT" in data.columns:
				sentence += ' It is the {} {} with .'.format(row['INSTRUMENT'], svo)
		else:
			verb = getInflection(verb, tag=v_tag)[0]
			sentence += 'Which {} did the {} {} ?'.format(row['OBJECT'], row['SUBJECT'], row['VERB'])
			# Which award did the actor win?
		sentence += ' .'
		sentences.append(sentence)
	return sentences
	"""



def to_sentences(data_path, out_dir, v_tag, form):
	data = pd.read_csv(data_path, sep='\t')
	data['sentence'] = synform_functions[form](data, v_tag, 'sbj' in data_path.lower())

	out_path = os.path.join(out_dir, os.path.basename(data_path).split('.txt')[0]+'.sentences')
	data.to_csv(out_path, header=True, index=None, sep='\t', mode='w')

synform_functions = {'transitive': _to_transitive, 'passive': _to_passive, 'cleft': _to_cleft,
	                 'question':_to_questions, 'violation':_to_violation }