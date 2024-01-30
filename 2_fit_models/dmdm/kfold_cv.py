import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import glob
import re
import pandas as pd
import ssm
from data_labels import create_abort_mask
from data_io import load_glm_vectors, get_file_dir, load_glmhmm_data, scan_glm_hmm_output
from data_postprocessing_utils import partition_data_by_session

sys.path.append(str(get_file_dir() / '1_fit_glm'))
from GLM import glm

def get_best_iter(model: str,
                  C: int,
                  num_folds: int,
                  inpt_y: np.array, 
                  inpt_rt: np.array, 
                  y: np.array, 
                  session: list, 
                  rt: np.array, 
                  stim_onset: np.array,
                  session_fold_lookup_table,
                  results_dir: Path,
                  outcome_dict=None,
                  K_vals: list =None):
    '''
    Create a matrix of size num_models x num_folds containing
    normalized loglikelihood for both train and test splits
    '''

    y = y.astype('int')

    # Create vectors to save output
    if model == "GLM_y" or model == "Lapse_Model":
        num_models = 1
    elif model == "GLM_HMM_y":
        num_models = len(K_vals) # maximum number of latent states
        D = 1  # number of output dimensions
    cvbt_folds_model = np.zeros((num_models, num_folds))
    cvbt_train_folds_model = np.zeros((num_models, num_folds))

    # Save best initialization for each model-fold combination
    best_init_cvbt_dict = {}
    print("Retrieving best iter results for model = {}; num_folds = {}".format(str(model), str(num_folds)))
    for fold in tqdm(range(num_folds)):
        # Use fold != fold for trainig and fold == fold for test dataset
        test_data, train_data, M_y, M_rt, n_test, n_train = \
            prepare_data_for_cv(inpt_y, inpt_rt, y, session, rt, stim_onset, 
                                session_fold_lookup_table, fold)
        
        [test_inpt_y, test_inpt_rt, test_y, test_rt, test_stim_onset, test_mask, test_session] = test_data
        [train_inpt_y, train_inpt_rt, train_y, train_rt, train_stim_onset, train_mask, train_session] = train_data

        ll0 = calculate_baseline_test_ll(
            train_y[np.where(train_mask == 1)[0], :],
            test_y[np.where(test_mask == 1)[0], :], C)
        ll0_train = calculate_baseline_test_ll(
            train_y[np.where(train_mask == 1)[0], :],
            train_y[np.where(train_mask == 1)[0], :], C)
        
        if (model == "GLM_y"):
            # Load parameters. Initerization does not matter for GLM
            glm_weights_file = results_dir / 'GLM' / ('fold_' + str(fold)) / 'variables_of_interest_y_iter_0.npz'            
            
            # Instantiate a new GLM object with these parameters
            ll_glm = calculate_glm_test_loglikelihood(
                glm_weights_file, test_y[np.where(test_mask == 1)[0], :],
                test_inpt_y[np.where(test_mask == 1)[0], :], M_y, C, outcome_dict)
            ll_glm_train = calculate_glm_test_loglikelihood(
                glm_weights_file, train_y[np.where(train_mask == 1)[0], :],
                train_inpt_y[np.where(train_mask == 1)[0], :], M_y, C, outcome_dict)
            
            cvbt_folds_model[0, fold] = calculate_cv_bit_trial(
                ll_glm, ll0, n_test)
            cvbt_train_folds_model[0, fold] = calculate_cv_bit_trial(
                ll_glm_train, ll0_train, n_train)
            
        elif model == "GLM_HMM_y":
            D = 1
            test_inpt_y = np.hstack((test_inpt_y, np.ones((len(test_inpt_y), 1))))
            train_inpt_y = np.hstack((train_inpt_y, np.ones((len(train_inpt_y), 1))))

            # For GLM-HMM set values of y for violations to 2.  This value doesn't
            # matter (as mask will ensure that these y values do not contribute to
            # loglikelihood calculation
            test_y[np.where(test_mask == 0)[0], :] = 2
            train_y[np.where(train_mask == 0)[0], :] = 2

            # For GLM-HMM, need to partition data by session
            this_test_inputs, this_test_datas, this_test_masks = \
                partition_data_by_session(
                    test_inpt_y, test_y,
                    test_mask,
                    test_session)
            this_train_inputs, this_train_datas, this_train_masks = \
                partition_data_by_session(
                    train_inpt_y, train_y,
                    train_mask,
                    train_session)

            for model_idx, K in enumerate(K_vals):

                dir_to_check = results_dir / ('GLM_HMM_y_K_' + str(K)) / ('fold_' + str(fold))
                test_ll_vals_across_iters, init_ordering_by_train, file_ordering_by_train = \
                    calculate_glm_hmm_test_loglikelihood(dir_to_check, 
                                                         'glm_hmm_y_raw_parameters_itr_', 
                                                         this_test_datas, this_test_inputs, 
                                                         this_test_masks, K, D, M_y, C)
                train_ll_vals_across_iters, _, _ = \
                      calculate_glm_hmm_test_loglikelihood(dir_to_check, 
                                                           'glm_hmm_y_raw_parameters_itr_', 
                                                           this_train_datas, this_train_inputs, 
                                                           this_train_masks, K, D, M_y, C)
                test_ll_vals_across_iters = test_ll_vals_across_iters[
                    file_ordering_by_train]
                train_ll_vals_across_iters = train_ll_vals_across_iters[
                    file_ordering_by_train]
                ll_glm_hmm_this_K = test_ll_vals_across_iters[0]
                cvbt_thismodel_thisfold = calculate_cv_bit_trial(ll_glm_hmm_this_K, ll0,
                                                                n_test)
                train_cvbt_thismodel_thisfold = calculate_cv_bit_trial(
                    train_ll_vals_across_iters[0], ll0_train, n_train)
                
                cvbt_folds_model[model_idx, fold] = cvbt_thismodel_thisfold
                cvbt_train_folds_model[model_idx, fold] = train_cvbt_thismodel_thisfold
                    
                # Save best initialization to dictionary for later:
                key_for_dict = 'GLM_HMM_y_K_' + str(K) + '/fold_' + str(fold)
                best_init_cvbt_dict[key_for_dict] = int(init_ordering_by_train[0])

        elif model == "Lapse_Model":
            # One lapse parameter model:
            cvbt_folds_model[1, fold], cvbt_train_folds_model[
                1,
                fold], _, _ = return_lapse_nll(inpt, y, session,
                                                session_fold_lookup_table,
                                                fold, 1, results_dir, C)
            # Two lapse parameter model:
            cvbt_folds_model[2, fold], cvbt_train_folds_model[
                2,
                fold], _, _ = return_lapse_nll(inpt, y, session,
                                                session_fold_lookup_table,
                                                fold, 2, results_dir, C)
        else:
            raise NotImplementedError
        
    # Save best initialization directories across animals, folds and models
    print(cvbt_folds_model)
    print(cvbt_train_folds_model)
    json_dump = json.dumps(best_init_cvbt_dict)
    f = open(results_dir / "best_init_cvbt_dict_{}.json".format(model), "w") # need to save for GLM?
    f.write(json_dump)
    f.close()
    # Save cvbt_folds_model as numpy array for easy parsing across all
    # models and folds
    np.savez(results_dir / "cvbt_folds_model_{}.npz".format(model), cvbt_folds_model)
    np.savez(results_dir / "cvbt_train_folds_model_{}.npz".format(model), cvbt_train_folds_model)

    print('Best iter saved!')
    return cvbt_folds_model, cvbt_train_folds_model, best_init_cvbt_dict

def prepare_data_for_cv(inpt_y, inpt_rt, y, session, rt, stim_onset,
                        session_fold_lookup_table, fold):

    abort_idx = np.where(y == 3)[0]
    nonabort_idx, nonabort_mask = create_abort_mask(
        abort_idx, inpt_y.shape[0])
    # Load train and test data for session
    test_data, train_data, M_y, M_rt, n_test, n_train = get_train_test_dta(inpt_y, inpt_rt,
                                                                           y, rt, stim_onset, 
                                                                           nonabort_mask, session, 
                                                                           session_fold_lookup_table, fold)

    return test_data, train_data, M_y, M_rt, n_test, n_train
        
def get_train_test_dta(inpt_y, inpt_rt, y, rt, stim_onset, mask, session, session_fold_lookup_table, fold):
    '''
    Split inpt_y, inpt_rt, y, rt, stim_onset, session arrays into train and test arrays
    '''
    test_sessions = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] == fold), 0]
    train_sessions = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] != fold), 0]
    idx_test = [str(sess) in test_sessions for sess in session]
    idx_train = [str(sess) in train_sessions for sess in session]

    test_inpt_y, test_inpt_rt, test_y, test_rt, test_stim_onset, test_mask, test_session = \
                                                        inpt_y[idx_test, :], inpt_rt[idx_test, :], \
                                                        y[idx_test, :], rt[idx_test, :], \
                                                        stim_onset[idx_test, :], mask[idx_test, :], \
                                                        session[idx_test]
    train_inpt_y, train_inpt_rt, train_y, train_rt, train_stim_onset, train_mask, train_session = \
                                                        inpt_y[idx_train, :], inpt_rt[idx_train, :], \
                                                        y[idx_train, :], rt[idx_train, :], \
                                                        stim_onset[idx_train, :], mask[idx_train, :], \
                                                        session[idx_train]
    
    M_y = train_inpt_y.shape[1]
    M_rt = train_inpt_rt.shape[1]
    n_test = np.sum(test_mask == 1)
    n_train = np.sum(train_mask == 1)
    return [test_inpt_y, test_inpt_rt, test_y, test_rt, test_stim_onset, test_mask, test_session], \
            [train_inpt_y, train_inpt_rt, train_y, train_rt, train_stim_onset, train_mask, train_session], \
            M_y, M_rt, n_test, n_train


def calculate_baseline_test_ll(train_y, test_y, C):
    """
    Calculate baseline loglikelihood for CV bit/trial calculation. 
    While the choices follow multinominal conditional distribution because of outcome hierarchy,
    one can simply ignore the conditional probability and assume they follow a very simple
    multinominal distribution, as we observe all the oucomes in this definition of baseline 
    loglikelihood. loglikelihood L will be calculated with
    L = sum(n_i(log(p_i)))
    , where p_i is the proportion of trials in which the animal took choice i
    in the training set and n_i is the number of trials in which the animal took choice i
    in the test set

    :return: baseline loglikelihood for CV bit/trial calculation
    """
    _, train_class_totals = np.unique(train_y, return_counts=True)
    train_class_probs = train_class_totals / train_y.shape[0] # calculate proportions
    _, test_class_totals = np.unique(test_y, return_counts=True)
    ll0 = 0
    for c in range(C):
        ll0 += test_class_totals[c] * np.log(train_class_probs[c])
    return ll0

def calculate_glm_test_loglikelihood(glm_weights_file, test_y, test_inpt, M,
                                     C, outcome_dict):
    loglikelihood_train, glm_vectors = load_glm_vectors(glm_weights_file)
    # Calculate test loglikelihood
    new_glm = glm(M, C, outcome_dict, obs='Categorical') # multinomial distribution
    # Set parameters to fit parameters:
    new_glm.params = glm_vectors
    # Get loglikelihood of training data:
    loglikelihood_test = new_glm.log_marginal([test_y], [test_inpt], None, None)
    return loglikelihood_test

def calculate_cv_bit_trial(ll_model, ll_0, n_trials):
    cv_bit_trial = ((ll_model - ll_0) / n_trials) / np.log(2)
    return cv_bit_trial

def calculate_glm_hmm_test_loglikelihood(glm_hmm_dir, fname_header, test_datas, test_inputs,
                                         test_masks, K, D, M, C):
    """
    calculate test loglikelihood for GLM-HMM model.  Loop through all
    initializations for fold of interest, and check that final train LL is
    same for top initializations
    :return:
    """
    glm_hmm_outputs = scan_glm_hmm_output(glm_hmm_dir, fname_header=fname_header)
    glm_hmm_iters = glm_hmm_outputs[0]
    glm_hmm_files = glm_hmm_outputs[1]

    train_ll_vals_across_iters = []
    test_ll_vals_across_iters = []
    for this_file_path in glm_hmm_files:
        # Loop through initializations and calculate BIC:
        this_hmm_params, lls = load_glmhmm_data(this_file_path)
        train_ll_vals_across_iters.append(lls[-1])
        # Instantiate a new HMM and calculate test loglikelihood:
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs_multinominal",
                           observation_kwargs=dict(C=C),
                           transitions="standard")
        this_hmm.params = this_hmm_params
        test_ll = this_hmm.log_likelihood(test_datas,
                                          inputs=test_inputs,
                                          masks=test_masks)
        

        test_ll_vals_across_iters.append(test_ll)
    # Order initializations by train LL (don't train on test data!):
    train_ll_vals_across_iters = np.array(train_ll_vals_across_iters)
    test_ll_vals_across_iters = np.array(test_ll_vals_across_iters)
    # Order raw files by train LL
    file_ordering_by_train = np.argsort(-train_ll_vals_across_iters)
    # Get initialization number from raw_file ordering
    init_ordering_by_train = np.array(glm_hmm_iters)[file_ordering_by_train]

    return test_ll_vals_across_iters, init_ordering_by_train, \
           file_ordering_by_train

def calculate_lapse_test_loglikelihood(lapse_file, test_y, test_inpt, M,
                                       num_lapse_params):
    lapse_loglikelihood, lapse_glm_weights, _, lapse_p, _ = load_lapse_params(
        lapse_file)
    # Instantiate a model with these parameters
    new_lapse_model = lapse_model(M, num_lapse_params)
    if num_lapse_params == 1:
        new_lapse_model.params = [lapse_glm_weights, np.array([lapse_p])]
    else:
        new_lapse_model.params = [lapse_glm_weights, lapse_p]
    # Now calculate test loglikelihood
    loglikelihood_test = new_lapse_model.log_marginal(datas=[test_y],
                                                      inputs=[test_inpt],
                                                      masks=None,
                                                      tags=None)
    return loglikelihood_test

def return_lapse_nll(inpt, y, session, session_fold_lookup_table, fold,
                     num_lapse_params, results_dir_glm_lapse, C):
    test_inpt, test_y, test_nonviolation_mask, this_test_session, \
    train_inpt, train_y, train_nonviolation_mask, this_train_session, M, \
    n_test, n_train = prepare_data_for_cv(
        inpt, y, session, session_fold_lookup_table, fold)
    ll0 = calculate_baseline_test_ll(train_y[train_nonviolation_mask == 1, :],
                                     test_y[test_nonviolation_mask == 1, :], C)
    ll0_train = calculate_baseline_test_ll(
        train_y[train_nonviolation_mask == 1, :],
        train_y[train_nonviolation_mask == 1, :], C)
    if num_lapse_params == 1:
        lapse_file = results_dir_glm_lapse + '/Lapse_Model/fold_' + str(
            fold) + '/lapse_model_params_one_param.npz'
    elif num_lapse_params == 2:
        lapse_file = results_dir_glm_lapse + '/Lapse_Model/fold_' + str(
            fold) + '/lapse_model_params_two_param.npz'
    ll_lapse = calculate_lapse_test_loglikelihood(
        lapse_file,
        test_y[test_nonviolation_mask == 1, :],
        test_inpt[test_nonviolation_mask == 1, :],
        M,
        num_lapse_params=num_lapse_params)
    ll_train_lapse = calculate_lapse_test_loglikelihood(
        lapse_file,
        train_y[train_nonviolation_mask == 1, :],
        train_inpt[train_nonviolation_mask == 1, :],
        M,
        num_lapse_params=num_lapse_params)
    nll_lapse = calculate_cv_bit_trial(ll_lapse, ll0, n_test)
    nll_lapse_train = calculate_cv_bit_trial(ll_train_lapse, ll0_train,
                                             n_train)
    return nll_lapse, nll_lapse_train, ll_lapse, ll_train_lapse