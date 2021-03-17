import argparse
import os
import logging.config

#from ttf.core import transformers_mlm
from ttf.core.tuples_to_sentences import to_sentences 

from ttf.utils import config as cutils
from ttf.utils import os_utils as outils


config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)

models_dict = {'bert':'bert-large-cased', 'roberta':'roberta-large', 'xlnet':'xlnet-large-cased'}

def _tuples_to_sentences(args):
    data_paths = args.input_path
    out_path = outils.check_dir(args.output_dir)
    verb_inflection = args.verb_tense

    for input_file in outils.get_filenames(data_paths):
        to_sentences(input_file,out_path, verb_inflection)

"""
def _run_transformers_mlm(args):
    data_sequences_path = args.input_file_data_sequences
    tokenized_sentences_path = args.input_file_tokenized_sentences
    outdir = outils.check_dir(args.output_dir)
    synrel = args.rel
    sets = args.settings
    models = [models_dict[m] for m in args.model]
    name = args.name

    if sorted(sets)[0]=='all':
        sets = ['standard', 'head', 'context', 'control']
    if os.path.isfile(data_sequences_path) and os.path.isfile(tokenized_sentences_path):
        transformers_mlm_model.build_model(data_sequences_path, tokenized_sentences_path, synrel, outdir, sets, models,name)  #  don't check whether the input files exist
    else:
        logger.info('Input path {} or {} does not exist'.format(data_sequences_path, tokenized_sentences_path))
"""

def _evaluation(args):
    #TODO: load arguments
    print()


def main():
    parser = argparse.ArgumentParser(prog='ttf')
    subparsers = parser.add_subparsers()

    parser_build_sentences = subparsers.add_parser('build-sentences',
                                                    help='Transform tuples into sentences.')
    parser_build_sentences.add_argument('-o', '--output-dir', default='../data/', help='output folder')
    parser_build_sentences.add_argument('-i', '--input-path', nargs='+', help='input files')
    parser_build_sentences.add_argument('-v', '--verb-tense', choices=['VBD', 'VBZ'], help="inflection of the verb")
    parser_build_sentences.set_defaults(func=_tuples_to_sentences)
    """
    # TRANSFORMERS MODELS - WORD PREDICTION TASK (BERT, RoBERTa, XLnet)
    parser_transformers_mlm = subparsers.add_parser('transformers-mlm',
                                                    help='Predict the probability of a role'
                                                         ' filler given a context')
    parser_transformers_mlm.add_argument('-o', '--output-dir', default='results/', help='output folder')
    parser_transformers_mlm.add_argument('-ids', '--input-file-data-sequences', required=True,
                                         help='path to data (human scores, sentences selected for each'
                                              ' sequence of the dataset')
    parser_transformers_mlm.add_argument('-its', '--input-file-tokenized-sentences', required=True,
                                         help='path to file with tokenization of selected sentences')
    parser_transformers_mlm.add_argument('-r', '--rel', required=True, choices=['sbj', 'obj'],
                                         help='syntactic relation')
    parser_transformers_mlm.add_argument('-s', '--settings', choices=['standard', 'head', 'context', 'control', 'all'], 
                                         default=['standard'], nargs='+', help='masked settings')
    parser_transformers_mlm.add_argument('-m','--model', choices=['bert', 'roberta'],
                                         nargs='+', default=['bert'], help='transformer models')
    parser_transformers_mlm.add_argument('-n', '--name', required=True, help='dataset name')
    parser_transformers_mlm.set_defaults(func=_run_transformers_mlm)
    """

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_usage()
        exit()
    args.func(args)


if __name__ == '__main__':
    main()



