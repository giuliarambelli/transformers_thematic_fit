from tfm.utils.data_utils import load_data_sequences
from transformers import XLNetTokenizer, TFXLNetLMHeadModel, RobertaTokenizer, TFRobertaForMaskedLM, BertTokenizer, TFBertForMaskedLM
import pandas as pd
import numpy as np
import logging
import tokenizations  #  pip install pytokenizations  (https://pypi.org/project/pytokenizations/)
import tensorflow as tf  #  TensorFlow 2.0 is required (Python 3.5-3.7, Pip 19.0 or later)


logger = logging.getLogger(__name__)

dict_tokenizers = {"bert-base-cased": BertTokenizer.from_pretrained('bert-base-cased'),
                   "bert-large-cased": BertTokenizer.from_pretrained('bert-large-cased'),
                   "roberta-large": RobertaTokenizer.from_pretrained('roberta-large'),
                   "gpt2-medium": GPT2Tokenizer.from_pretrained('gpt2-medium')}


dict_mlm_models = {"bert-base-cased": TFBertForMaskedLM.from_pretrained('bert-base-cased'),
                   "bert-large-cased": TFBertForMaskedLM.from_pretrained('bert-large-cased')
                   "roberta-large": TFRobertaForMaskedLM.from_pretrained('roberta-large'),
                   "gpt2-medium": TFGPT2LMHeadModel.from_pretrained('gpt2-medium')}

BATCH_SIZE = 256


class TransformerModel:
    def __init__(self, transf_model):
        self.model_name = transf_model  
        self.tokenizer = dict_tokenizers[transf_model]
        self.mlm_model = dict_mlm_models[transf_model]

    def prepare_input(self, d_sequences):  #  my dataset
      #   only a row - sentence, id to mask. Must find target token.
      target_tokens = []
      sentences_with_mask = []
      dependents_indices = []
      for i in range(len(d_sequences)):
        sent = d_sequences.iloc[i,0]
        id_dep = d_sequences.iloc[i,1] - 1   #  remove -1 if Giulia gives files where index starts from zero
        target_token = sent.split(" ")[id_dep]  
        #  check if target token is in dictionary - otherwise add None to the lists
        if self.model_name.startswith("bert"):
          if self.tokenizer.convert_ids_to_tokens(self.tokenizer.convert_tokens_to_ids(target_token)) == "[UNK]":
            target_tokens.append(None)
           else:
            target_tokens.append(target_token)
        if self.model_name.startswith("roberta"):
          if self.tokenizer.convert_ids_to_tokens(self.tokenizer.convert_tokens_to_ids(target_token)) == "<unk>":
            target_tokens.append(None)
           else:
            if id_dep == 0:  
              target_tokens.append(target_token)
            else:
              target_tokens.append("Ġ"+target_token)
        if self.model_name.startswith("gpt"):
          if self.tokenizer.convert_ids_to_tokens(self.tokenizer.convert_tokens_to_ids(target_token)) == "<|endoftext|>":
            target_tokens.append(None)
           else:
            if id_dep == 0:  
              target_tokens.append(target_token)
           else:
              target_tokens.append("Ġ"+target_token)
        #   mask the sentence
        list_words = []
        for w in range(len(sent.split(" "))):
          if sent.split(" ")[w] != id_dep:  
            list_words.append(sent.split(" ")[w])
          else:
            if self.model_name.startswith("bert"):
              list_words.append("[MASK]")
            if self.model_name.startswith("roberta"):
              list_words.append("<mask>")
            if self.model_name.startswith("gpt"):
              list_words.append(sent.split(" ")[w])  #  mask is not needed for gpt
        masked_sent = join(list_words)
        sentences_with_mask.append(masked_sent)
        model_tokenization = self.tokenizer.tokenize(masked_sent)
        if self.model_name.startswith("bert"):
          dependent_index = model_tokenization.index("[MASK]") + 1  # take into account token [CLS]
        if self.model_name.startswith("roberta"):
          dependent_index = model_tokenization.index("<mask>") + 1
        if self.model_name.startswith("gpt"):
          our_tokenization = masked_sent.split(" ")
          other_tokens_2_model_tokens, model_tokens_2_other_tokens = tokenizations.get_alignments(our_tokenization, model_tokenization)
          dependent_index = other_tokens_2_model_tokens[id_dep][0]  
        dependents_indices.append(dependent_index)
        i += 1  
        return target_tokens, sentences_with_mask, dependents_indices
      
      
      

    def compute_filler_probability(self, list_target_words, list_masked_sentences, list_dependents_indexes):
        inputs = self.tokenizer(list_masked_sentences, padding=True, return_tensors="tf")
        probabilities_fillers = []
        print("Executing model for batch...")
        print()
        outputs = self.mlm_model(inputs)[0]  #   (batch_size, sequence_length, config.vocab_size)
        if (self.model_name.startswith("bert")) or (self.model_name.startswith("roberta")):
          for batch_elem, target_word, dep_index in zip(range(outputs.shape(0)), list_target_words, list_dependents_indexes):
            if target_word == None:
              probabilities_fillers.append(None)
            else:
              all_probabilities = tf.nn.softmax(outputs[batch_elem, dep_index]).numpy()
              probabilities_fillers.append(all_probabilities[self.tokenizer.convert_tokens_to_ids(target_word)]) 
       if self.model_name.startswith("gpt"):   #  check for gpt
       return probabilities_fillers
        
        
        

    def compute_fillers_scores(self, data_sequences, batch_dimension=64):
        num_sentences = len(data_sequences)  #  num sentences in my dataset
        if num_sentences % batch_dimension == 0:
            num_batches = num_sentences // batch_dimension
        else:
            num_batches = num_sentences // batch_dimension + 1
        total_scores = []
        for batch in range(num_batches):
            print()
            logger.info("Processing batch {} of {} . Progress: {} ...".format(batch + 1, num_batches,
                                                                        np.round((100 / num_batches) * (batch + 1), 2)))
            if batch != num_batches - 1:  #  if batch is not the last batch
                target_words, masked_sentences, positions_dependents = self.\
                    prepare_input(data_sequences[batch * batch_dimension : (batch + 1) * batch_dimension])
                scores = self.compute_filler_probability(target_words, masked_sentences, positions_dependents)
            else:  #  if batch is the last batch 
                target_words, masked_sentences, positions_dependents = self.prepare_input(data_sequences[batch * batch_dimension : ])
                scores = self.compute_filler_probability(target_words, masked_sentences, positions_dependents)
            total_scores.extend(scores)
        return total_scores



def build_model(path_data, thematic_role, output_directory, transformers, name):  
    data = load_data_sequences(path_data)  #  sentence, id_dep, human_score
    for transformer in transformers:
        model = TransformerModel(transformer)
        model_fillers_scores = model.compute_fillers_scores(data, BATCH_SIZE) 
        data["computed_score"] = model_fillers_scores
        data.to_csv("{}/{}.transformers_mlm_{}_{}_{}".
                        format(output_directory, name, transformer, thematic_role),
                        header=None, index=None, sep='\t', mode='a')



if __name__ == '__main__':
    build_model('SP', 'nsubj')
