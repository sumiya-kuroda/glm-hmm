import autograd.numpy as np
import os
from pathlib import Path
import re
from functools import reduce

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

def load_glmhmm_data(glmhmm_output, IsRegularization=False):
    # load hmm output files
    container = np.load(glmhmm_output, allow_pickle=True)
    data = [container[key] for key in container]
    this_hmm_params = data[0]
    lls = data[1]
    alpha = data[2]
    sigma = data[3]
    global_fit = data[4]
    if IsRegularization:
        sigma = data[3]
        global_fit = data[4]
        l = data[5]
        regulization = data[6]

        return this_hmm_params, lls, alpha, sigma, global_fit, l, regulization
    else:
        return this_hmm_params, lls, alpha, sigma, global_fit

def scan_glmhmm_output(path, fname_header=''):
    matched_iter = []
    matched_file = []
    for dirpath, _, filenames in os.walk(str(path)):
        for filename in [f for f in filenames 
                         if f.startswith(fname_header)]:
            all_num = re.findall('\d+', filename)
            matched_iter.append([int(i) for i in all_num if int(i) < 100][0]) # save iter num only
            matched_file.append(str(Path(dirpath) / filename))
    if not matched_file:
        raise FileNotFoundError('No GLM-HMM output files found in ' + str(path))
    return matched_iter, matched_file

def get_file_name_for_best_glmhmm_fold(cvbt_folds_model, model_idx, K, 
                                        alpha_idx=None, alpha=None, sigma_idx=None, sigma=None, overall_dir: Path = None,
                                        best_init_cvbt_dict=None, model=None, fname_header=None,
                                        global_fit=True, map_params=None, animal=None):
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
    # loc_best = 0
    best_fold = np.where(cvbt_folds_model[0, 0, model_idx, :] == \
                         max(cvbt_folds_model[0, 0, model_idx, :]))[0][0]
    base_path = overall_dir / (model +'_K_' + str(K)) / ('fold_' + str(best_fold))
    if not global_fit:
        sigma, alpha, _ = get_best_map_params(map_params, animal=animal, fold=best_fold, K=K)

    key_for_dict = model +'_K_' + str(K) + '/fold_' + str(best_fold) \
                        + '/alpha_' + str(alpha) + '/sigma_' + str(sigma)
    best_iter = best_init_cvbt_dict[key_for_dict]

    fname_tail = '_a' + str(int(alpha*100)) + '_s' +  str(int(sigma*100)) + '.npz'
    fpath = base_path / ('iter_' + str(best_iter)) / (fname_header + str(best_iter) + fname_tail)
    return fpath, best_fold

def get_file_name_for_best_glmhmm_fold_l2(cvbt_folds_model, model_idx, K, 
                                        alpha_idx=None, alpha=None, sigma_idx=None, sigma=None, overall_dir: Path = None,
                                        best_init_cvbt_dict=None, model=None, fname_header=None,
                                        global_fit=True, map_params=None, animal=None,
                                        lambda_value=None):
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
    # loc_best = 0
    best_fold = np.where(cvbt_folds_model[0, model_idx, :] == \
                         max(cvbt_folds_model[0, model_idx, :]))[0][0]
    base_path = overall_dir / (model +'_K_' + str(K)) / ('fold_' + str(best_fold))
    if map_params is not None:
        lambda_value, _ = get_best_l2_params(map_params, animal='global', fold=best_fold, K=K)

    key_for_dict = model +'_K_' + str(K) + '/fold_' + str(best_fold) \
        + '/alpha_' + str(alpha) + '/sigma_' + str(sigma) \
            + '/lambda_' + str(lambda_value) + '/fold_tuning_' + str(best_fold) 
    best_iter = best_init_cvbt_dict[key_for_dict]

    fname_tail = '_a' + str(int(alpha*100)) + '_s' +  str(int(sigma*100)) + '_l' +  str(int(lambda_value*100)) + '.npz'
    fpath = base_path / ('iter_' + str(best_iter)) / (fname_header + str(best_iter) + fname_tail)
    # fpath = base_path / ('iter_' + str(best_iter)) / (fname_header + str(best_iter) + fname_tail)
    return fpath, best_fold

def get_file_name_for_best_glmhmm_iter(K, 
                                        overall_dir,
                                        best_init_cvbt_dict=None, model=None, fname_header=None,
                                        global_fit=True, num_fold = 5):

    fpaths = []
    for fold in range(num_fold):
        base_path = overall_dir / (model +'_K_' + str(K)) / ('fold_' + str(fold))
        if global_fit:
            alpha = 1
            sigma = 100

        key_for_dict = model +'_K_' + str(K) + '/fold_' + str(fold) \
                            + '/alpha_' + str(alpha) + '/sigma_' + str(sigma)
        best_iter = best_init_cvbt_dict[key_for_dict]

        fname_tail = '_a' + str(int(alpha*100)) + '_s' +  str(int(sigma*100)) + '.npz'
        fpath = base_path / ('iter_' + str(best_iter)) / (fname_header + str(best_iter) + fname_tail)
        fpaths.append(fpath)
    return fpaths
def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model

def load_global_best_params(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params

def load_best_map_params(map_params_file):
    container = np.load(map_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    value = data[0]
    return value

def get_best_map_params(map_params, animal: str, fold: int, K: int):
    params_noanimal = map_params[:,0:-1].astype('float32')
    params_animal = map_params[:,-1]

    idx_K = np.where(params_noanimal[:,2] == K)
    idx_fold = np.where(params_noanimal[:,3] == fold)
    idx_animal = np.where(params_animal == animal)

    idx = reduce(np.intersect1d, (idx_K, idx_fold, idx_animal))[0]
    sigma = params_noanimal[idx, 0]
    alpha = params_noanimal[idx, 1]

    return sigma, alpha, idx

def get_best_l2_params(map_params, animal: str, fold: int, K: int):
    params_noanimal = map_params.astype('int32')
    # params_animal = map_params[:,-1]

    idx_K = np.where(params_noanimal[:,1] == K)
    idx_fold = np.where(params_noanimal[:,2] == fold)
    if animal == 'global' :
        idx = reduce(np.intersect1d, (idx_K, idx_fold))[0]
    else:
        pass
        # idx_animal = np.where(params_animal == animal)
        # idx = reduce(np.intersect1d, (idx_K, idx_fold, idx_animal))[0]
    lambda_value = params_noanimal[idx, 0]

    return lambda_value, idx