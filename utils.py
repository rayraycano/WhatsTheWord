import os
import json
from static import LOGDIR

def load_run_data(run_id, full_folderpath=None):
    """
    Loads the run hyperparameters and dictionary mappings
    :param run_id: name of run
    :param full_folderpath: optional specification of full folderpath to pull run from
    :return:
        run_config: dictionary[string]:value -- specifies run hyperparameters
        dictionary: dictionary[string]: number -- maps words to lookup number
        reversed_dictionary: dictionary[number]: string -- maps numbers to vocab words
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    run_id_path = os.path.join(dir_path, 'logs', run_id)
    print(dir_path)
    print(run_id_path)
    if not os.path.isdir(run_id_path):
        print("invalid run_id: {}".format(run_id_path))
        return None, None, None, None
    if full_folderpath is None:
        datefolder_path = max(os.listdir(run_id_path))
        print("Using run: {}".format(datefolder_path))
        full_folderpath = os.path.join(run_id_path, datefolder_path)
    with open(os.path.join(full_folderpath, 'config.json'), 'r') as c:
        run_config = json.load(c)
    with open(os.path.join(full_folderpath, 'metadata.tsv'), 'r') as m:
        reversed_dictionary = {i: w for i, w in enumerate(m.readlines())}
    dictionary = dict(zip(reversed_dictionary.values(), reversed_dictionary.keys()))

    return run_config, dictionary, reversed_dictionary, full_folderpath


def run_experiment(args, f):
    """
    Parses out the correct path to save files to and runs the experiment `f`
    :param args: arguments from the argument parser
    :param f: functiona that takes a datafile, run_id, logdir, and **kwargs
    :return: None
    """
    param_file = args.pfile
    with open(param_file, 'r') as p:
        param_dict = json.load(p)
        datafile = param_dict.pop('datafile')
        run_id = param_file.split('/')[-1].split('.')[0]
        params_idx = param_file.split('/').index('params')
        config_path = param_file.split('/')[params_idx + 1:-1]
        logdir = os.path.join(LOGDIR, *config_path)
        print('run_id: ' + run_id)
    f(datafile, run_id, logdir, **param_dict)