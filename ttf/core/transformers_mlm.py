# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:24:05 2020

@author: pedin
"""
from tfm.utils.data_utils import load_data_sequences
from transformers import XLNetTokenizer, TFXLNetLMHeadModel, RobertaTokenizer, TFRobertaForMaskedLM, BertTokenizer, TFBertForMaskedLM
import pandas as pd
import numpy as np
import logging
import tokenizations  #  pip install pytokenizations  (https://pypi.org/project/pytokenizations/)
import tensorflow as tf  #  TensorFlow 2.0 is required (Python 3.5-3.7, Pip 19.0 or later)


logger = logging.getLogger(__name__)

dict_tokenizers = {"xlnet-large-cased": XLNetTokenizer.from_pretrained("xlnet-large-cased"),
                   "roberta-large": RobertaTokenizer.from_pretrained('roberta-large'),
                   "bert-large-cased": BertTokenizer.from_pretrained('bert-large-cased')}


dict_mlm_models = {"xlnet-large-cased": TFXLNetLMHeadModel.from_pretrained('xlnet-large-cased'),
                   "roberta-large": TFRobertaForMaskedLM.from_pretrained('roberta-large'),
                   "bert-large-cased": TFBertForMaskedLM.from_pretrained('bert-large-cased')}

BATCH_SIZE = 256


class TransformerModel:
    def __init__(self, transf_model):
        self.model_name = transf_model
        self.tokenizer = dict_tokenizers[transf_model]
        self.mlm_model = dict_mlm_models[transf_model]

    def prepare_input(self, d_sequences, tokenized_data):
        target_tokens = []
        sentences_with_mask = []
        heads_indices = []
        dependents_indices = []
        lengths_for_each_sentence = []
        i = 0
        for sentence in tokenized_data:
            #  save targets
            if self.model_name.startswith("bert"):
                target_tokens.append(sentence[1][(d_sequences.iloc[i, 4])-1])
            if self.model_name.startswith("roberta"):
                if (d_sequences.iloc[i, 4])-1 == 0:
                    target_tokens.append(sentence[1][(d_sequences.iloc[i, 4])-1])
                else:
                    target_tokens.append("Ġ"+(sentence[1][(d_sequences.iloc[i, 4])-1]))
                #  since roberta and gpt tokens incorporate space when they occur in positions differents than the first
            if self.model_name.startswith("xlnet"):
                target_tokens.append(u"\u2581"+(sentence[1][(d_sequences.iloc[i, 4])-1]))
                #  since in sentencepiece tokenizer this symbol is used for whitespace
            #  sentence masking - necessary for passing inputs to the model
            if self.model_name.startswith("bert"):
                sentence[1][(d_sequences.iloc[i, 4]) - 1] = "[MASK]"
            if self.model_name.startswith("roberta") or self.model_name.startswith("xlnet"):
                # since roberta model does not recognize token "[MASK]"
                sentence[1][(d_sequences.iloc[i, 4]) - 1] = "<mask>"
            masked_sentence = " ".join([token for token in sentence[1]])  # for gpt the unmasked sentence is maintained
            sentences_with_mask.append(masked_sentence)
            #  save position head and dependent in model tokenization
            #  - necessary for reading output of the model and creating attention mask arrays
            model_tokenization = self.tokenizer.tokenize(masked_sentence)
            other_tokens_2_model_tokens, model_tokens_2_other_tokens = tokenizations.\
                get_alignments(sentence[1], model_tokenization)
            if self.model_name.startswith("xlnet"):
                head_index = [index for index in other_tokens_2_model_tokens[(d_sequences.iloc[i, 3]) - 1]]
            else:
                head_index = [index + 1 for index in other_tokens_2_model_tokens[(d_sequences.iloc[i, 3]) - 1]]
            heads_indices.append(head_index)
            if self.model_name.startswith("bert"):
                dependent_index = model_tokenization.index("[MASK]") + 1  # take into account token [CLS]
            if self.model_name.startswith("roberta"):
                dependent_index = model_tokenization.index("<mask>") + 1
            if self.model_name.startswith("xlnet"):
                dependent_index = model_tokenization.index("<mask>")  # since xlnet tokenizer does not add cls token at the beginning of the sequence
            dependents_indices.append(dependent_index)
            # save sentence length in model tokenization - necessary for avoiding masking [SEP] token in context setting
            length_sentence = len(model_tokenization) + 2  # take into account tokens [CLS] and [SEP]
            lengths_for_each_sentence.append(length_sentence)
            i += 1
        return target_tokens, sentences_with_mask, heads_indices, dependents_indices, lengths_for_each_sentence

    def compute_filler_probability(self, list_sentences, list_target_words, list_masked_sentences, list_heads_indexes,
                                   list_dependents_indexes, list_lengths_sentences, masking_setting):
        #  try:
        targets_indexes = self.tokenizer.convert_tokens_to_ids(list_target_words)
        if self.model_name.startswith("xlnet"):
            self.tokenizer.padding_side = "right"  #  since instances of xlnet tokenizer by default apply padding to the left
            inputs = self.tokenizer(list_masked_sentences, padding=True, return_tensors="tf")
        else:
            inputs = self.tokenizer(list_masked_sentences, padding=True, return_tensors="tf")
        #  a new attention mask is created depending on the setting (if setting = "standard" no substitution is performed)
        if self.model_name.startswith("xlnet"):  #  prevents the model from performing attention on the last token of the sequence
            new_attention_mask = []
            for mask, sentence_length in zip(inputs["attention_mask"], list_lengths_sentences):
                mask_array = np.array(mask)
                mask_array[sentence_length - 1] = 0
                new_attention_mask.append(tf.convert_to_tensor(mask_array))
            inputs["attention_mask"] = tf.convert_to_tensor(new_attention_mask)
        if masking_setting == "head":
            new_attention_mask = []
            for mask, index_head in zip(inputs["attention_mask"], list_heads_indexes):
                mask_array = np.array(mask)
                mask_array[index_head] = 0
                new_attention_mask.append(tf.convert_to_tensor(mask_array))
            inputs["attention_mask"] = tf.convert_to_tensor(new_attention_mask)
        if (masking_setting == "context") or (masking_setting == "control"):
            new_attention_mask = []
            for mask, head_index, dependent_index, masked_sentence, \
                length_sentence_model in zip(inputs["attention_mask"], list_heads_indexes, list_dependents_indexes,
                                             list_masked_sentences, list_lengths_sentences):
                mask = [0 for elem in mask]
                mask_array = np.array(mask)
                if masking_setting == "context":
                    mask_array[head_index] = 1
                mask_array[dependent_index] = 1
                if not self.model_name.startswith("xlnet"):
                    mask_array[0] = 1  # since xlnet tokenizer does not add cls token at the beginning of the sequence
                    mask_array[length_sentence_model - 1] = 1
                new_attention_mask.append(tf.convert_to_tensor(mask_array))
            inputs["attention_mask"] = tf.convert_to_tensor(new_attention_mask)
        probabilities_fillers = []
        print("Executing model for batch...")
        print()
        outputs = self.mlm_model(inputs)[0]
        list_for_indexing1 = [[n, list_dependents_indexes[n]] for n in range(len(list_dependents_indexes))]
        list_for_indexing2 = [[n, targets_indexes[n]] for n in range(len(targets_indexes))]
        selected_outputs = tf.gather_nd(outputs, list_for_indexing1)
        all_probabilities = tf.nn.softmax(selected_outputs)
        probabilities_targets = tf.gather_nd(all_probabilities, list_for_indexing2)
        for probability in probabilities_targets:
            probabilities_fillers.append(probability.numpy())  # class numpy.float32
        #  except:
        #  list_probs = [None]*len(targets)
        return probabilities_fillers

    def compute_fillers_scores(self, data_sequences, splitted_sentences, settings=['standard'], batch_dimension=64):
        num_sentences = len(data_sequences)
        if num_sentences % batch_dimension == 0:
            num_batches = num_sentences // batch_dimension
        else:
            num_batches = num_sentences // batch_dimension + 1
        #settings = ["standard", "head", "context", "control"]
        total_scores = {"standard": [], "head": [], "context": [], "control": []}
        for batch in range(num_batches):
            print()
            logger.info("Processing batch {} of {} . Progress: {} ...".format(batch + 1, num_batches,
                                                                        np.round((100 / num_batches) * (batch + 1), 2)))
            if batch != num_batches - 1:
                target_words, masked_sentences, positions_heads, positions_dependents, lengths_sentences = self.\
                    prepare_input(data_sequences[batch * batch_dimension : (batch + 1) * batch_dimension],
                                  splitted_sentences[batch * batch_dimension : (batch + 1) * batch_dimension])
                for setting in settings:
                    scores = self.compute_filler_probability([sentence[0] for sentence
                                                              in splitted_sentences[batch * batch_dimension :
                                                                                    (batch + 1) * batch_dimension]],
                                                             target_words, masked_sentences, positions_heads,
                                                             positions_dependents, lengths_sentences, setting)
                    total_scores[setting].extend(scores)
            else:
                target_words, masked_sentences, positions_heads, positions_dependents, lengths_sentences = self.\
                    prepare_input(data_sequences[batch * batch_dimension : ],
                                  splitted_sentences[batch * batch_dimension : ])
                for setting in settings:
                    scores = self.compute_filler_probability([sentence[0] for sentence in
                                                          splitted_sentences[batch * batch_dimension : ]],
                                                         target_words, masked_sentences, positions_heads,
                                                         positions_dependents, lengths_sentences, setting)
                    total_scores[setting].extend(scores)
        return total_scores



def build_model(path_data, path_tokenized_sentences, syn_rel, output_directory, settings, transformers, name):
    #TODO: model to perform
    #  Load data for each sequence (sentences, original scores) and a file containing tokenized sentences
    data, tokenizations = load_data_sequences(path_data, path_tokenized_sentences)
    #  Create Transformer-based models objects (models have also to be imported at line 9 and added in dict_tokenizers - line - and dict_mlm_models - line )
    #transformers = ["bert-large-cased", "roberta-large", "xlnet-large-cased"]  #  add or remove models here
    for transformer in transformers:
        model = TransformerModel(transformer)
        #settings = ["standard", "head", "context", "control"]
        model_fillers_scores = model.compute_fillers_scores(data, tokenizations, settings, BATCH_SIZE)
        for setting in settings:
            data, tokenizations = load_data_sequences(path_data, path_tokenized_sentences)
            data["computed_score"] = model_fillers_scores[setting]
            data.to_csv("{}/{}.transformers_mlm_{}_{}_{}".
                        format(output_directory, name, transformer, syn_rel, setting),
                        header=None, index=None, sep='\t', mode='a')



if __name__ == '__main__':
    build_model('SP', 'nsubj')
