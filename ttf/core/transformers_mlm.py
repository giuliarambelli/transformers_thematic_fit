from ttf.utils.data_utils import load_data_sequences
from transformers import BertTokenizer, TFBertForMaskedLM, RobertaTokenizer, TFRobertaForMaskedLM, GPT2Tokenizer, \
    TFGPT2LMHeadModel
import pandas as pd
import numpy as np
import logging
import tokenizations   #   pip install pytokenizations  (https://pypi.org/project/pytokenizations/)
import tensorflow as tf  #  TensorFlow 2.0 is required (Python 3.5-3.7, Pip 19.0 or later)


logger = logging.getLogger(__name__)

dict_tokenizers = {"gpt2-medium": GPT2Tokenizer.from_pretrained('gpt2-medium'),
                   "roberta-large": RobertaTokenizer.from_pretrained('roberta-large')}


dict_mlm_models = {"gpt2-medium": TFGPT2LMHeadModel.from_pretrained('gpt2-medium'),
                   "roberta-large": TFRobertaForMaskedLM.from_pretrained('roberta-large')}

BATCH_SIZE = 256

N_PREDICTIONS = 5


class TransformerModel:
    def __init__(self, transf_model):
        self.model_name = transf_model
        self.tokenizer = dict_tokenizers[transf_model]
        self.mlm_model = dict_mlm_models[transf_model]

    def prepare_input(self, d_sequences):
        target_tokens = []
        sentences_with_mask = []
        dependents_indices = []
        for i in range(len(d_sequences)):
            sent = d_sequences["sentence"][i]#.replace(".", " .")  #  remove the replace if Giulia adjusts the input
            id_dep = d_sequences["id_dep"][i] - 1
            #  remove -1 if Giulia gives files where index starts from zero - da cambiare in base al nome della colonna
            # GIULIA: alla fine è sempre la parola in posizione -1, l'unico problema è con il soggetto, però fare una colonna in cui 
            # l'indice è sempre lo stesso per tutti forse è un po' ridondante
            target_token = sent.split(" ")[id_dep]
            #  check if target token is in dictionary - otherwise add None to the lists
            if self.model_name.startswith("bert"):
                if self.tokenizer.convert_ids_to_tokens(self.tokenizer.convert_tokens_to_ids(target_token)) == "[UNK]":
                    target_tokens.append(None)
                else:
                    target_tokens.append(target_token)
            if self.model_name.startswith("roberta"):
                if id_dep == 0:
                    if self.tokenizer.convert_ids_to_tokens(self.tokenizer.convert_tokens_to_ids(target_token)) == \
                            "<unk>":
                        target_tokens.append(None)
                    else:
                        target_tokens.append(target_token)
                else:
                    if self.tokenizer.convert_ids_to_tokens(self.tokenizer.convert_tokens_to_ids("Ġ"+target_token)) == \
                            "<unk>":
                        target_tokens.append(None)
                    else:
                        target_tokens.append("Ġ"+target_token)
            if self.model_name.startswith("gpt"):
                if id_dep == 0:
                    if self.tokenizer.convert_ids_to_tokens(
                            self.tokenizer.convert_tokens_to_ids(target_token)) == "<|endoftext|>":
                        target_tokens.append(None)
                    else:
                        target_tokens.append(target_token)
                else:
                    if self.tokenizer.convert_ids_to_tokens(
                            self.tokenizer.convert_tokens_to_ids("Ġ" + target_token)) == "<|endoftext|>":
                        target_tokens.append(None)
                    else:
                        target_tokens.append("Ġ" + target_token)
            #   mask the sentence
            list_words = []
            for w in range(len(sent.split(" "))):
                if w != id_dep:
                    list_words.append(sent.split(" ")[w])
                else:
                    if self.model_name.startswith("bert"):
                        list_words.append("[MASK]")
                    if self.model_name.startswith("roberta"):
                        list_words.append("<mask>")
                    if self.model_name.startswith("gpt"):
                        list_words.append(sent.split(" ")[w])  #  mask is not needed for gpt
            masked_sent = " ".join(list_words)
            sentences_with_mask.append(masked_sent)
            model_tokenization = self.tokenizer.tokenize(masked_sent)
            if self.model_name.startswith("bert"):
                dependent_index = model_tokenization.index("[MASK]") + 1  # take into account token [CLS]
            if self.model_name.startswith("roberta"):
                dependent_index = model_tokenization.index("<mask>") + 1
            if self.model_name.startswith("gpt"):
                our_tokenization = masked_sent.split(" ")
                other_tokens_2_model_tokens, model_tokens_2_other_tokens = tokenizations.\
                    get_alignments(our_tokenization, model_tokenization)
                dependent_index = other_tokens_2_model_tokens[id_dep][0] + 1
            dependents_indices.append(dependent_index)
            i += 1
        return target_tokens, sentences_with_mask, dependents_indices

    def compute_filler_probability(self, list_target_words, list_masked_sentences, list_dependents_indexes):
        if self.model_name.startswith("gpt"):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(["<|endoftext|>" + sent + "<|endoftext|>" for sent in list_masked_sentences],
                                    padding=True, return_tensors="tf")
            # it is necessary to add a token at the beginning of the sentence
        else:
            inputs = self.tokenizer(list_masked_sentences, padding=True, return_tensors="tf")
        probabilities_fillers = []
        predicted_fillers = []
        print("Executing model for batch...")
        print()
        outputs = self.mlm_model(inputs)[0]
        for batch_elem, target_word, dep_index in zip(range(outputs.shape[0]), list_target_words,
                                                      list_dependents_indexes):
            if target_word is None:
                probabilities_fillers.append(None)
                predicted_fillers.append(None)
            else:
                if (self.model_name.startswith("bert")) or (self.model_name.startswith("roberta")):
                    all_probabilities = tf.nn.softmax(outputs[batch_elem, dep_index]).numpy()
                if self.model_name.startswith("gpt"):
                    all_probabilities = tf.nn.softmax(outputs[batch_elem, dep_index - 1]).numpy()
                probabilities_fillers.append(all_probabilities[self.tokenizer.convert_tokens_to_ids(target_word)])
                idxs_predictions = (-(np.array(all_probabilities))).argsort()[:N_PREDICTIONS]
                predictions = self.tokenizer.convert_ids_to_tokens([int(index) for index in idxs_predictions])
                string_predicted_fillers = ""
                for word, index in zip(predictions, idxs_predictions):
                    string_predicted_fillers += word.replace("Ġ", "")+"_("+str(all_probabilities[index])+")"+";"
                predicted_fillers.append(string_predicted_fillers)
        return probabilities_fillers, predicted_fillers

    def compute_fillers_scores(self, data_sequences, batch_dimension=64):
        num_sentences = len(data_sequences)
        if num_sentences % batch_dimension == 0:
            num_batches = num_sentences // batch_dimension
        else:
            num_batches = num_sentences // batch_dimension + 1
        total_scores = []
        total_best_fillers = []
        for batch in range(num_batches):
            print()
            logger.info("Processing batch {} of {} . Progress: {} ...".format(batch + 1, num_batches,
                                                                              np.round((100 / num_batches) * (batch + 1)
                                                                                       , 2)))
            if batch != num_batches - 1:
                target_words, masked_sentences, positions_dependents = self.\
                    prepare_input(data_sequences[batch * batch_dimension: (batch + 1) * batch_dimension])
                scores = self.compute_filler_probability(target_words, masked_sentences, positions_dependents)
            else:
                target_words, masked_sentences, positions_dependents = self.\
                    prepare_input(data_sequences[batch * batch_dimension:])
                scores, best_fillers = self.compute_filler_probability(target_words, masked_sentences,
                                                                       positions_dependents)
            total_scores.extend(scores)
            total_best_fillers.extend(best_fillers)
        return total_scores, total_best_fillers


def build_model(path_data, thematic_role, output_directory, transformers, name):  
    data = load_data_sequences(path_data)  #  sentence, id_dep (da adattare id_dep al nome che verrà dato da Giulia)
    for transformer in transformers:
        model = TransformerModel(transformer)
        model_fillers_scores, model_completions = model.compute_fillers_scores(data, BATCH_SIZE)
        data["computed_score"] = model_fillers_scores
        data["best_completions"] = model_completions
        data.to_csv("{}/{}_transformers_mlm_{}_{}.txt".
                    format(output_directory, name, transformer, thematic_role), index=None, sep='\t', mode='a')


if __name__ == '__main__':
    build_model('test_input.txt', 'recipient', 'results', ['gpt2-medium'], 'vassallo')
