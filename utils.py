import os
import json
from static import LOGDIR
import copy
from datetime import datetime

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
    if full_folderpath is not None:
        run_id_path = os.path.join(dir_path, full_folderpath)
    else:
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
        print("run config: ", run_config)
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


def run_grid_search(args, f):
    """
    Runs grid search over the params in the params file.
    :param args:
    :param f:
    :return:
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
        configs = create_grid_list(param_dict)
        best_score = (-float('inf'),)
        results_path = os.path.join('results', run_id)
        results_filename = os.path.join(results_path, datetime.now().strftime('%Y%m%d-%H%M'))
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        results = []
        for c in configs:
            print("config")
            print(c)
            result, result_dict = f(datafile, run_id, logdir, **c)
            result_dict.update(c)
            results.append((result, result_dict))
            if result > best_score:
                best_score = result
                print('new best score: {}'.format(result_dict))
        results.sort()
        jsonResults = json.dumps([x[1] for x in results], indent=4)
        with open(results_filename, 'w') as f:
            f.write(jsonResults)


def create_grid_list(param_dict):
    configs = [{}]
    for key, value in param_dict.items():
        new_configs = []
        for v in value:
            for c in configs:

                new_config = copy.deepcopy(c)
                new_config[key] = v
                new_configs.append(new_config)
                # print(new_configs)
        configs = copy.deepcopy(new_configs)
    return configs
