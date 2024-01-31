import autograd.numpy as np
import os
from pathlib import Path
import re

def get_file_dir(): # dmdm dir
    return Path(os.path.dirname(os.path.realpath(__file__)))

# Functions to load data
def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt_y = data[0]
    inpt_rt = data[1]
    y = data[2]
    session = data[3]
    rt = data[4]
    stim_onset = data[5]
    return inpt_y, inpt_rt, y, session, rt, stim_onset

def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table

def load_animal_list(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list

# Function to load results
def load_glm_vectors(glm_vectors_file):
    container = np.load(glm_vectors_file)
    data = [container[key] for key in container]
    loglikelihood_train = data[0]
    recovered_weights = data[1]
    return loglikelihood_train, recovered_weights

def load_lapse_params(lapse_file):
    container = np.load(lapse_file, allow_pickle=True)
    data = [container[key] for key in container]
    lapse_loglikelihood = data[0]
    lapse_glm_weights = data[1]
    lapse_glm_weights_std = data[2],
    lapse_p = data[3]
    lapse_p_std = data[4]
    return lapse_loglikelihood, lapse_glm_weights, lapse_glm_weights_std, \
           lapse_p, lapse_p_std

def load_glmhmm_data(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    this_hmm_params = data[0]
    lls = data[1]
    return [this_hmm_params, lls]

def scan_glm_hmm_output(path, fname_header=''):
    matched_iter = []
    matched_file = []
    for dirpath, _, filenames in os.walk(str(path)):
        for filename in [f for f in filenames if f.startswith(fname_header)]:
            iter_num = re.findall('\d+', filename)
            matched_iter.append(iter_num)
            matched_file.append(str(Path(dirpath) / filename))

    if not matched_file:
        raise FileNotFoundError('No GLM-HMM output files found in ' + str(path))
    return matched_iter, matched_file

def get_file_name_for_best_model_fold(cvbt_folds_model, K, overall_dir: Path,
                                      best_init_cvbt_dict, model, fname_header):
    '''
    Get the file name for the best initialization for the K value specified
    :param cvbt_folds_model:
    :param K:
    :param models:
    :param overall_dir:
    :param best_init_cvbt_dict:
    :return:
    '''
    # Identify best fold for best model:
    # loc_best = K - 1
    loc_best = 0
    best_fold = np.where(cvbt_folds_model[loc_best, :] == \
                         max(cvbt_folds_model[loc_best, :]))[0][0]
    base_path = overall_dir / (model +'_K_' + str(K)) / ('fold_' + str(best_fold))
    key_for_dict = model +'_K_' + str(K) + '/fold_' + str(best_fold)
    best_iter = best_init_cvbt_dict[key_for_dict]
    raw_file = base_path / ('iter_' + str(best_iter)) / (fname_header + str(best_iter) + '.npz')
    return raw_file

def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model

def load_global_params(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params