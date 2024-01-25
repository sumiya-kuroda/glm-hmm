import autograd.numpy as np
import autograd.numpy.random as npr
import os
from pathlib import Path

npr.seed(65)

def get_file_dir(): # dmdm dir
    return Path(os.path.dirname(os.path.realpath(__file__)))

# Load data
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

# Loat results
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

def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model