# Create a matrix of size num_models x num_folds containing
# normalized loglikelihood for both train and test splits
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
import re
import pandas as pd
import ssm
from data_io import load_session_fold_lookup, load_data

def get_best_iter(model: str,
                  C: int,
                  num_folds: int,
                  data_dir: Path,
                  results_dir: Path,
                  outcome_dict=None,
                  K_vals: list =None):

    # Load data
    inpt, y, session, _, _ = load_data(data_dir / 'all_animals_concat.npz')
    y = y.astype('int')

    session_fold_lookup_table = load_session_fold_lookup(
        data_dir / 'all_animals_concat_session_fold_lookup.npz')

    # Create vectors to save output
    if model == "GLM" or model == "Lapse_Model":
        num_models = 1
    elif model == "GLM_HMM":
        num_models = len(K_vals) # maximum number of latent states
        D = 1  # number of output dimensions
    cvbt_folds_model = np.zeros((num_models, num_folds))
    cvbt_train_folds_model = np.zeros((num_models, num_folds))

    # Save best initialization for each model-fold combination
    best_init_cvbt_dict = {}
    print("Retrieving best iter results for model = {}; num_folds = {}".format(str(model), str(num_folds)))
    for fold in tqdm(range(num_folds)):
        # Use fold != fold for trainig and fold == fold for test dataset
        test_inpt, test_y, test_nonviolation_mask, this_test_session, \
        train_inpt, train_y, train_nonviolation_mask, this_train_session, M,\
        n_test, n_train = prepare_data_for_cv(
            inpt, y, session, session_fold_lookup_table, fold)

        ll0 = calculate_baseline_test_ll(
            train_y[train_nonviolation_mask == 1, :],
            test_y[test_nonviolation_mask == 1, :], C)
        ll0_train = calculate_baseline_test_ll(
            train_y[train_nonviolation_mask == 1, :],
            train_y[train_nonviolation_mask == 1, :], C)
        
        if model == "GLM":
            # Load parameters. Initerization does not matter for GLM
            glm_weights_file = results_dir / 'GLM' / ('fold_' + str(fold)) / 'variables_of_interest_iter_9.npz'            
            
            # Instantiate a new GLM object with these parameters
            ll_glm = calculate_glm_test_loglikelihood(
                glm_weights_file, test_y[test_nonviolation_mask == 1, :],
                test_inpt[test_nonviolation_mask == 1, :], M, C, outcome_dict)
            ll_glm_train = calculate_glm_test_loglikelihood(
                glm_weights_file, train_y[train_nonviolation_mask == 1, :],
                train_inpt[train_nonviolation_mask == 1, :], M, C, outcome_dict)
            
            cvbt_folds_model[0, fold] = calculate_cv_bit_trial(
                ll_glm, ll0, n_test)
            cvbt_train_folds_model[0, fold] = calculate_cv_bit_trial(
                ll_glm_train, ll0_train, n_train)

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
            
        elif model == "GLM_HMM":
            for model_idx, K in enumerate(K_vals):
                print("K = " + str(K))
                cvbt_folds_model[model_idx, fold], \
                cvbt_train_folds_model[
                    model_idx, fold], _, _, init_ordering_by_train = \
                    return_glmhmm_nll(
                        np.hstack((inpt, np.ones((len(inpt), 1)))), y,
                        session, session_fold_lookup_table, fold,
                        K, D, C, results_dir)
                # Save best initialization to dictionary for later:
                key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(fold)
                best_init_cvbt_dict[key_for_dict] = int(init_ordering_by_train[0])
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

def get_train_test_dta(inpt, y, mask, session, session_fold_lookup_table,
                       fold):
    '''
    Split inpt, y, mask, session arrays into train and test arrays
    :param inpt:
    :param y:
    :param mask:
    :param session:
    :param session_fold_lookup_table:
    :param fold:
    :return:
    '''
    test_sessions = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] == fold), 0]
    train_sessions = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] != fold), 0]
    idx_test = [str(sess) in test_sessions for sess in session]
    idx_train = [str(sess) in train_sessions for sess in session]
    test_inpt, test_y, test_mask, this_test_session = inpt[idx_test, :], y[
                                                                         idx_test,
                                                                         :], \
                                                      mask[idx_test], session[
                                                          idx_test]
    train_inpt, train_y, train_mask, this_train_session = inpt[idx_train,
                                                          :], y[idx_train,
                                                              :], \
                                                          mask[idx_train], \
                                                          session[idx_train]
    return test_inpt, test_y, test_mask, this_test_session, train_inpt, \
           train_y, train_mask, this_train_session

def prepare_data_for_cv(inpt, y, session, session_fold_lookup_table, fold):

    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, nonviolation_mask = create_violation_mask(
        violation_idx, inpt.shape[0])
    # Load train and test data for session
    test_inpt, test_y, test_nonviolation_mask, this_test_session, \
    train_inpt, train_y, train_nonviolation_mask, this_train_session = \
        get_train_test_dta(
            inpt, y, nonviolation_mask, session, session_fold_lookup_table,
            fold)
    M = train_inpt.shape[1]
    n_test = np.sum(test_nonviolation_mask == 1)
    n_train = np.sum(train_nonviolation_mask == 1)
    return test_inpt, test_y, test_nonviolation_mask, this_test_session, \
           train_inpt, train_y, train_nonviolation_mask, this_train_session, \
           M, n_test, n_train


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
    new_glm = glm(M, C, outcome_dict)
    # Set parameters to fit parameters:
    new_glm.params = glm_vectors
    # Get loglikelihood of training data:
    loglikelihood_test = new_glm.log_marginal([test_y], [test_inpt], None,
                                              None)
    return loglikelihood_test


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


def calculate_glm_hmm_test_loglikelihood(glm_hmm_dir, test_datas, test_inputs,
                                         test_nonviolation_masks, K, D, M, C):
    """
    calculate test loglikelihood for GLM-HMM model.  Loop through all
    initializations for fold of interest, and check that final train LL is
    same for top initializations
    :return:
    """
    this_file_name = glm_hmm_dir + '/iter_*/glm_hmm_raw_parameters_*.npz'
    raw_files = glob.glob(this_file_name, recursive=True)
    train_ll_vals_across_iters = []
    test_ll_vals_across_iters = []
    for file in raw_files:
        # Loop through initializations and calculate BIC:
        this_hmm_params, lls = load_glmhmm_data(file)
        train_ll_vals_across_iters.append(lls[-1])
        # Instantiate a new HMM and calculate test loglikelihood:
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C),
                           transitions="standard")
        this_hmm.params = this_hmm_params
        test_ll = this_hmm.log_likelihood(test_datas,
                                          inputs=test_inputs,
                                          masks=test_nonviolation_masks)
        test_ll_vals_across_iters.append(test_ll)
    # Order initializations by train LL (don't train on test data!):
    train_ll_vals_across_iters = np.array(train_ll_vals_across_iters)
    test_ll_vals_across_iters = np.array(test_ll_vals_across_iters)
    # Order raw files by train LL
    file_ordering_by_train = np.argsort(-train_ll_vals_across_iters)
    raw_file_ordering_by_train = np.array(raw_files)[file_ordering_by_train]
    # Get initialization number from raw_file ordering
    init_ordering_by_train = [
        int(re.findall(r'\d+', file)[-1])
        for file in raw_file_ordering_by_train
    ]
    return test_ll_vals_across_iters, init_ordering_by_train, \
           file_ordering_by_train


def return_glmhmm_nll(inpt, y, session, session_fold_lookup_table, fold, K, D,
                      C, results_dir_glm_hmm):
    '''
    For a given fold, return NLL for both train and test datasets for
    GLM-HMM model with K, D, C.  Requires reading in best
    parameters over all initializations for GLM-HMM (hence why
    results_dir_glm_hmm is required as an input)
    :param inpt:
    :param y:
    :param session:
    :param session_fold_lookup_table:
    :param fold:
    :param K:
    :param D:
    :param C:
    :param results_dir_glm_hmm:
    :return:
    '''
    test_inpt, test_y, test_nonviolation_mask, this_test_session, \
    train_inpt, train_y, train_nonviolation_mask, this_train_session, M, \
    n_test, n_train = prepare_data_for_cv(
        inpt, y, session, session_fold_lookup_table, fold)
    ll0 = calculate_baseline_test_ll(train_y[train_nonviolation_mask == 1, :],
                                     test_y[test_nonviolation_mask == 1, :], C)
    ll0_train = calculate_baseline_test_ll(
        train_y[train_nonviolation_mask == 1, :],
        train_y[train_nonviolation_mask == 1, :], C)
    # For GLM-HMM set values of y for violations to 1.  This value doesn't
    # matter (as mask will ensure that these y values do not contribute to
    # loglikelihood calculation
    test_y[test_nonviolation_mask == 0, :] = 1
    train_y[train_nonviolation_mask == 0, :] = 1
    # For GLM-HMM, need to partition data by session
    test_inputs, test_datas, test_nonviolation_masks = \
        partition_data_by_session(
            test_inpt, test_y,
            np.expand_dims(test_nonviolation_mask, axis=1),
            this_test_session)
    train_inputs, train_datas, train_nonviolation_masks = \
        partition_data_by_session(
            train_inpt, train_y,
            np.expand_dims(train_nonviolation_mask, axis=1),
            this_train_session)
    dir_to_check = results_dir_glm_hmm + '/GLM_HMM_K_' + str(
        K) + '/fold_' + str(fold) + '/'
    test_ll_vals_across_iters, init_ordering_by_train, \
    file_ordering_by_train = calculate_glm_hmm_test_loglikelihood(
        dir_to_check, test_datas, test_inputs, test_nonviolation_masks, K, D,
        M, C)
    train_ll_vals_across_iters, _, _ = calculate_glm_hmm_test_loglikelihood(
        dir_to_check, train_datas, train_inputs, train_nonviolation_masks, K,
        D, M, C)
    test_ll_vals_across_iters = test_ll_vals_across_iters[
        file_ordering_by_train]
    train_ll_vals_across_iters = train_ll_vals_across_iters[
        file_ordering_by_train]
    ll_glm_hmm_this_K = test_ll_vals_across_iters[0]
    cvbt_thismodel_thisfold = calculate_cv_bit_trial(ll_glm_hmm_this_K, ll0,
                                                     n_test)
    train_cvbt_thismodel_thisfold = calculate_cv_bit_trial(
        train_ll_vals_across_iters[0], ll0_train, n_train)
    return cvbt_thismodel_thisfold, train_cvbt_thismodel_thisfold, \
           ll_glm_hmm_this_K, \
           train_ll_vals_across_iters[0], init_ordering_by_train


def calculate_cv_bit_trial(ll_model, ll_0, n_trials):
    cv_bit_trial = ((ll_model - ll_0) / n_trials) / np.log(2)
    return cv_bit_trial


def get_file_name_for_best_model_fold(cvbt_folds_model, K, overall_dir,
                                      best_init_cvbt_dict):
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
    best_fold = np.where(cvbt_folds_model[loc_best, :] == max(cvbt_folds_model[
                                                              loc_best, :]))[
        0][0]
    base_path = overall_dir + '/GLM_HMM_K_' + str(K) + '/fold_' + str(
        best_fold)
    key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    best_iter = best_init_cvbt_dict[key_for_dict]
    raw_file = base_path + '/iter_' + str(
        best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return raw_file

def create_cv_frame_for_plotting(cv_file):
    cvbt_folds_model = load_cv_arr(cv_file)
    glm_lapse_model = cvbt_folds_model[:3, ]
    idx = np.array([0, 3, 4, 5, 6])
    cvbt_folds_model = cvbt_folds_model[idx, :]
    # Identify best cvbt:
    mean_cvbt = np.mean(cvbt_folds_model, axis=1)
    loc_best = np.where(mean_cvbt == max(mean_cvbt))[0]
    best_val = max(mean_cvbt)
    # Create dataframe for plotting
    num_models = cvbt_folds_model.shape[0]
    num_folds = cvbt_folds_model.shape[1]
    # Create pandas dataframe:
    data_for_plotting_df = pd.DataFrame({
        'model':
            np.repeat(np.arange(num_models), num_folds),
        'cv_bit_trial':
            cvbt_folds_model.flatten()
    })
    return data_for_plotting_df, loc_best, best_val, glm_lapse_model