from ttf.utils.data_utils import load_data_sequences, get_thematic_role
from transformers import BertTokenizer, TFBertForMaskedLM, RobertaTokenizer, TFRobertaForMaskedLM, GPT2Tokenizer, \
    TFGPT2LMHeadModel
import pandas as pd
import numpy as np
import logging
import tokenizations   #   pip install pytokenizations  (https://pypi.org/project/pytokenizations/)
import tensorflow as tf  #  TensorFlow 2.0 is required (Python 3.5-3.7, Pip 19.0 or later)
import os.path



logger = logging.getLogger(__name__)

dict_tokenizers = {"bert-base-cased": BertTokenizer.from_pretrained('bert-base-cased'),
                   "bert-large-cased": BertTokenizer.from_pretrained('bert-large-cased'),
                   "roberta-large": RobertaTokenizer.from_pretrained('roberta-large'),
                   "gpt2-medium": GPT2Tokenizer.from_pretrained('gpt2-medium')}


dict_mlm_models = {"bert-base-cased": TFBertForMaskedLM.from_pretrained('bert-base-cased'),
                   "bert-large-cased": TFBertForMaskedLM.from_pretrained('bert-large-cased'),
                   "roberta-large": TFRobertaForMaskedLM.from_pretrained('roberta-large'),
                   "gpt2-medium": TFGPT2LMHeadModel.from_pretrained('gpt2-medium')}


BATCH_SIZE = 256

N_PREDICTIONS = 5


class TransformerModel:
    def __init__(self, transf_model):
        self.model_name = transf_model
        self.tokenizer = dict_tokenizers[transf_model]
        self.mlm_model = dict_mlm_models[transf_model]

    def prepare_input(self, d_sequences, th_role):
        target_tokens = []
        sentences_with_mask = []
        dependents_indices = []
        d_sequences = d_sequences.reset_index(drop=True)
        for i in range(len(d_sequences)):
            sent = d_sequences["sentence"][i]
            if th_role != "Agent":
                id_dep = len(sent.split(" ")) - 2
            else:
                id_dep = 1
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
                    string_predicted_fillers += word.replace("Ġ", "")+"_("+str(np.log(all_probabilities[index]))+")"+";"
                predicted_fillers.append(string_predicted_fillers)
        probabilities_baseline = []
        new_attention_mask = []
        for mask, id, sent in zip(inputs["attention_mask"], list_dependents_indexes, list_masked_sentences):
            mask_array = np.array([0 for elem in mask])
            mask_array[id] = 1
            new_attention_mask.append(tf.convert_to_tensor(mask_array))
        inputs["attention_mask"] = tf.convert_to_tensor(new_attention_mask)
        outputs = self.mlm_model(inputs)[0]
        for batch_elem, target_word, dep_index in zip(range(outputs.shape[0]), list_target_words,
                                                      list_dependents_indexes):
            if target_word is None:
                probabilities_baseline.append(None)
            else:
                if (self.model_name.startswith("bert")) or (self.model_name.startswith("roberta")):
                    all_probabilities = tf.nn.softmax(outputs[batch_elem, dep_index]).numpy()
                if self.model_name.startswith("gpt"):
                    all_probabilities = tf.nn.softmax(outputs[batch_elem, 0]).numpy()
                probabilities_baseline.append(all_probabilities[self.tokenizer.convert_tokens_to_ids(target_word)])
        return np.log(probabilities_fillers.float()), predicted_fillers, np.log(probabilities_baseline.float())


    def compute_fillers_scores(self, data_sequences, role, batch_dimension=64):
        num_sentences = len(data_sequences)
        if num_sentences % batch_dimension == 0:
            num_batches = num_sentences // batch_dimension
        else:
            num_batches = num_sentences // batch_dimension + 1
        total_scores = []
        total_best_fillers = []
        total_bline_scores = []
        for batch in range(num_batches):
            print()
            logger.info("Processing batch {} of {} . Progress: {} ...".format(batch + 1, num_batches,
                                                                              np.round((100 / num_batches) * (batch + 1)
                                                                                       , 2)))
            if batch != num_batches - 1:
                target_words, masked_sentences, positions_dependents = self.\
                    prepare_input(data_sequences[batch * batch_dimension: (batch + 1) * batch_dimension], role)
                scores, best_fillers, bline_scores = self.compute_filler_probability(target_words, masked_sentences, positions_dependents)
            else:
                target_words, masked_sentences, positions_dependents = self.\
                    prepare_input(data_sequences[batch * batch_dimension:], role)
                scores, best_fillers, bline_scores = self.compute_filler_probability(target_words, masked_sentences,
                                                                       positions_dependents)
            total_scores.extend(scores)
            total_best_fillers.extend(best_fillers)
            total_bline_scores.extend(bline_scores)
        return total_scores, total_best_fillers, total_bline_scores


def build_model(path_data, output_directory, transformers, baseline):
    #data = load_data_sequences(path_data) #GIULIA: la funzione prende 2 file, non avendo più il pile in pickle non dovremmo importare solo un file csv classico?
    data = pd.read_csv(path_data, sep='\t')
    #thematic_role = os.path.basename(path_data).split("_")[1].split(".")[0]  # funziona solo se lasciamo i nomi dei file con le frasi come sono adesso
    thematic_role = get_thematic_role(list(data.columns))
    for transformer in transformers:
        model = TransformerModel(transformer)
        model_fillers_scores, model_completions, baseline_scores = model.compute_fillers_scores(data, thematic_role, BATCH_SIZE)
        data["computed_score"] = model_fillers_scores
        data["best_completions"] = model_completions
        if baseline == "y":
            data["baseline_score"] = baseline_scores
        out_path = os.path.join(output_directory, os.path.basename(path_data).split('.')[0]+'_transformers_mlm_{}.txt'.format(transformer))
        data.to_csv(out_path, index=None, sep='\t', mode='a')
        #data.to_csv("{}/{}_transformers_mlm_{}_{}.txt".
        #            format(output_directory, name, transformer, thematic_role), index=None, sep='\t', mode='a')


if __name__ == '__main__':
    build_model('TypicalityRatings_Recipient.sentences', 'results', ['gpt2-medium'], 'vassallo')
