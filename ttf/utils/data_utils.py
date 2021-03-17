import pandas as pd
from pickle import load, dump

def load_data_sequences(path1, path2):
	df = pd.read_csv(path1, sep="\t", header=None, names=["dataset", "sequence", "sentence",
														  "id_verb", "id_dep", "fit_score"])
	fp = open(path2, "rb")
	tokenized_sentences = load(fp, encoding="utf-8")
	fp.close()
	return df, tokenized_sentences