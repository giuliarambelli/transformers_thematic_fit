import argparse  
import os
import logging.config

from tfm.core import transformers_mlm as transformers_mlm_model

from tfm.utils import config as cutils
from tfm.utils import os_utils as outils
from tfm.utils import evaluation_utils as eutils

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)

models_dict = {'bertbase': 'bert-base-cased', 'bertlarge':'bert-large-cased', 'roberta':'roberta-large', 'gpt2':'gpt2-medium'}


def _run_transformers_mlm(args):
    data_sequences_path = args.input_file_data_sequences  #  dove trovare i dati
    tokenized_sentences_path = args.input_file_tokenized_sentences  #  dove trovare le frasi tokenizzate
    outdir = outils.check_dir(args.output_dir)  #  output location
    synrel = args.rel
    sets = args.settings
    models = [models_dict[m] for m in args.model]
    name = args.name

    if sorted(sets)[0]=='all':  #  sorted function to list of settings. If lists starts with all, execute for all settings
        sets = ['standard', 'head', 'context', 'control']
    if os.path.isfile(data_sequences_path) and os.path.isfile(tokenized_sentences_path):
        transformers_mlm_model.build_model(data_sequences_path, tokenized_sentences_path, synrel, outdir, sets, models,name)  #  don't check whether the input files exist
    else:
        logger.info('Input path {} or {} does not exist'.format(data_sequences_path, tokenized_sentences_path))


def main():
    parser = argparse.ArgumentParser(prog='tfm')
    subparsers = parser.add_subparsers() 

    # TRANSFORMERS MODELS - WORD PREDICTION TASK (BERT, RoBERTa, GPT2)
    parser_transformers_mlm = subparsers.add_parser('transformers-mlm',
                                                    help='Predict the probability of a role'
                                                         ' filler given a context')
    parser_transformers_mlm.add_argument('-o', '--output-dir', default='results/', help='output folder')  #  first argument is name or flags (can be two)
    parser_transformers_mlm.add_argument('-ids', '--input-file-data-sequences', required=True,
                                         help='path to data (human scores, sentences selected for each'
                                              ' sequence of the dataset')
    parser_transformers_mlm.add_argument('-its', '--input-file-tokenized-sentences', required=True,
                                         help='path to file with tokenization of selected sentences')  #  NO NEED FOR IT IN OUR NEW VERSION
    parser_transformers_mlm.add_argument('-r', '--rel', required=True, choices=['sbj', 'obj'],
                                         help='thematic role')  #  THE NUMBER OF CHOICES MUST BE AUGMENTED - A COSA SERVONO?  
    parser_transformers_mlm.add_argument('-s', '--settings', choices=['standard', 'head', 'context', 'control', 'all'], 
                                         default=['standard'], nargs='+', help='masked settings')
    parser_transformers_mlm.add_argument('-m','--model', choices=['bert-base', 'bert-large', 'roberta', 'gpt2'],
                                         nargs='+', default=['bert-large'], help='transformer models')
    parser_transformers_mlm.add_argument('-n', '--name', required=True, help='dataset name')  #  VERIFICARE LA SUA FUNZIONE ANCHE IN RELAZIONE A THEMATIC ROLE
    parser_transformers_mlm.set_defaults(func=_run_transformers_mlm)  


    args = parser.parse_args()
    if 'func' not in args:
        parser.print_usage()
        exit()
    args.func(args)


if __name__ == '__main__':
    main()



