import logging
import pandas as pd
from pickle import load, dump
logger = logging.getLogger(__name__)

roles = ["LOCATION", "TIME", "RECIPIENT", "INSTRUMENT"]


def load_data_sequences(path1, path2):
	df = pd.read_csv(path1, sep="\t", header=None, names=["dataset", "sequence", "sentence",
														  "id_verb", "id_dep", "fit_score"])
	fp = open(path2, "rb")
	tokenized_sentences = load(fp, encoding="utf-8")
	fp.close()
	return df, tokenized_sentences


def get_thematic_role(columns):
	if any(c in columns for c in roles):
		return list(set(roles).intersection(set(columns)))[0]
	else:
		if 'OBJECT' not in columns:
			return 'VERB'
		else:
			return 'OBJECT'




class VectorsDict(dict):

	def __init__(self, withPoS=False):
		self.withPoS = withPoS

	def __getitem__(self, item):
		form, pos = item
		if self.withPoS:
			# TODO: allow for different composition functions
			form = form+"/"+pos

		return super().__getitem__(form)


def _load_vocab(fpath):
	ret = []
	with open(fpath) as fin:
		for line in fin:
			line = line.strip().split()
			for el in line:
				ret.append(el)

	return ret


def _load_vectors_npy(vectors_fpath, withPoS, noun_set, len_vectors):

	vectors_vocab = vectors_fpath[:-4]+".vocab"

	vectors = np.load(vectors_fpath)
	vocab = _load_vocab(vectors_vocab)

	noun_vectors = VectorsDict(withPoS)

	for key, value in zip(vocab, vectors):

		if key in noun_set or not len(noun_set):
			noun_vectors[key] = value
			if len_vectors > -1:
				noun_vectors[key] = value[:len_vectors]

	logger.info("loaded {} vectors".format(len(noun_vectors)))
	return noun_vectors


def _load_vectors_from_text(vectors_fpath, withPoS, noun_set, len_vectors):

	noun_vectors = VectorsDict(withPoS)

	with open(vectors_fpath) as fin_model:
		n_words, len_from_file = fin_model.readline().strip().split()
		len_from_file = int(len_from_file)

		for line in fin_model:
			if len_vectors == -1:
				len_vectors = len_from_file

			line = line.strip().split()
			len_line = len(line)
			word = " ".join(line[:len_line-len_from_file])

			if word in noun_set or not len(noun_set):
				try:
					vector = [float(x) for x in line[-len_vectors:]]
					noun_vectors[word] = np.array(vector)
				except:
					logger.info("problem with vector for word {}".format(word))

	logger.info("loaded {} vectors".format(len(noun_vectors)))
	return noun_vectors


def load_vectors(vectors_fpath, withPoS=False, noun_set=set(), len_vectors=-1):

	if vectors_fpath.endswith(".npy"):
		ret = _load_vectors_npy(vectors_fpath, withPoS=withPoS, noun_set=noun_set, len_vectors=len_vectors)
	else:
		ret = _load_vectors_from_text(vectors_fpath, withPoS=withPoS, noun_set=noun_set, len_vectors=len_vectors)
	return ret