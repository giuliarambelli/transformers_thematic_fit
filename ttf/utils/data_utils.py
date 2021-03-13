import pandas as pd

def load_data_sequences(path1):
	df = pd.read_csv(path1, sep="\t", header=None, names=["sentence", "id_dep", "human_score"])
	return df
