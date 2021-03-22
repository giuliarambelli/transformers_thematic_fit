import argparse
import os
import logging.config

from ttf.core import transformers_mlm
from ttf.core.tuples_to_sentences import to_sentences
from ttf.core.evaluation import evaluation

from ttf.utils import config as cutils
from ttf.utils import os_utils as outils

config_dict = cutils.load(os.path.join(os.path.dirname(__file__), 'logging_utils', 'logging.yml'))
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)

models_dict = {'bert-base': 'bert-base-cased', 'bert-large': 'bert-large-cased', 'roberta-large': 'roberta-large', 'gpt2-medium': 'gpt2-medium'}


def _tuples_to_sentences(args):
    data_paths = args.input_path
    out_path = outils.check_dir(args.output_dir)
    verb_inflection = args.verb_tense

    for input_file in outils.get_filenames(data_paths):
        to_sentences(input_file, out_path, verb_inflection)


def _run_transformers_mlm(args):
    data_sequences_path = args.input_file_data_sequences
    outdir = outils.check_dir(args.output_dir)
    models = [models_dict[m] for m in args.model]
    #name = args.name

    if os.path.isfile(data_sequences_path):
        transformers_mlm.build_model(data_sequences_path, outdir, models)  # don't check whether the input files exist
    else:
        logger.info('Input path {} does not exist'.format(data_sequences_path))


def _evaluation(args):
    data_paths = args.input_path
    eval_type = args.eval
    thresh = args.thresh
    for input_file in outils.get_filenames(data_paths):
        evaluation(input_file, eval_type, thresh)


def main():
    parser = argparse.ArgumentParser(prog='ttf')
    subparsers = parser.add_subparsers()

    parser_build_sentences = subparsers.add_parser('build-sentences',
                                                   help='Transform tuples into sentences.')
    parser_build_sentences.add_argument('-o', '--output-dir', default='../data/', help='output folder')
    parser_build_sentences.add_argument('-i', '--input-path', nargs='+', required=True, help='input files')
    parser_build_sentences.add_argument('-v', '--verb-tense', choices=['VBD', 'VBZ'], default='VBD',
                                        help="inflection of the verb")
    parser_build_sentences.set_defaults(func=_tuples_to_sentences)

    # TRANSFORMERS MODELS - WORD PREDICTION TASK (BERT base and large, RoBERTa large, GPT2 medium)
    parser_transformers_mlm = subparsers.add_parser('transformers-mlm',
                                                    help='Predict the probability of a role'
                                                         ' filler given a context')
    parser_transformers_mlm.add_argument('-o', '--output-dir', default='results/', help='output folder')
    parser_transformers_mlm.add_argument('-ids', '--input-file-data-sequences', required=True,
                                         help='path to data (human scores, sentences selected for each'
                                              ' sequence of the dataset')
    parser_transformers_mlm.add_argument('-m', '--model', choices=['bert-base', 'bert-large', 'roberta-large', 'gpt2-medium'],
                                         nargs='+', default=['bert-large'], help='transformer models')
    #parser_transformers_mlm.add_argument('-n', '--name', required=True, help='dataset name')
    parser_transformers_mlm.set_defaults(func=_run_transformers_mlm)

    parser_evaluation = subparsers.add_parser('evaluation', help='compute evaluation measures')
    parser_evaluation.add_argument('-i', '--input-path', nargs='+', required=True, help='input files')
    parser_evaluation.add_argument('-e', '--eval', choices=['simple', 'diff', 'corr'], required=True,
                                   help='output folder')
    parser_evaluation.add_argument('-t', '--thresh', default=0, help='threshold for probabilities difference')
    parser_evaluation.set_defaults(func=_evaluation)

    args = parser.parse_args()
    if 'func' not in args:
        parser.print_usage()
        exit()
    args.func(args)


if __name__ == '__main__':
    main()


