import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import ssm
from data_labels import create_abort_mask, partition_data_by_session_L2
from data_io import load_glm_vectors, get_file_dir, load_glmhmm_data, scan_glmhmm_output, get_best_map_params, get_best_l2_params
from data_postprocessing_utils import partition_data_by_session, plot_best_MAP_params

sys.path.append(str(get_file_dir() / '1_fit_glm'))
from GLM import glm

class KFoldCV(object):
  """
  Class for performing k-fold cross-validation using GLM or GLM-HMM output
  Lapse model is not included at the moment.
  """
  def __init__(self,
                 model: str,
                 num_folds: int,
                 K_vals: list = None,
                 global_fit: bool = True,
                 Alpha_vals: list = None,
                 Sigma_vals: list = None,
                 Lambda_vals: list = None,
                 results_dir: Path = None,
                 animal: str = None):

                 self.model = model
                 self.num_folds = num_folds
                 self.K_vals = K_vals
                 self.global_fit = global_fit
                 self.Alpha_vals = Alpha_vals
                 self.Sigma_vals = Sigma_vals
                 self.Lambda_vals = Lambda_vals if Lambda_vals is not None else None
                 self.animal = animal
                 if (global_fit == True) & ('HMM' in self.model):
                    if isinstance(Alpha_vals, list) or isinstance(Sigma_vals, list):
                        if len(Alpha_vals) > 1 or len(Sigma_vals) > 1 :
                            raise ValueError("MLE only supports alpha = 1 and sigma = inf.")
                    if not self.animal is None :
                        raise ValueError("Global fitting does not require animals to be specified.")               
                 # Create vectors to save output
                 self._reset()

                 self.results_dir = results_dir


  def _reset(self):
        if self.model == "GLM_y" or self.model == "Lapse_Model":
            num_models = num_alpha = num_sigma = 1
        elif self.model == "GLM_HMM_y":
            num_models = len(self.K_vals) # maximum number of latent states
            num_alpha = len(self.Alpha_vals)
            num_sigma = len(self.Sigma_vals)
        self.D = 1  # number of output dimensions (only needed in GLM-HMM)
        if self.Lambda_vals is not None: # L2 regularizationpenalization
            num_lambda = len(self.Lambda_vals)

            self.cvbt_folds_model = np.zeros((num_lambda, num_models, self.num_folds))
            self.cvbt_train_folds_model = np.zeros((num_lambda, num_models, self.num_folds))
            self.best_init_cvbt_dict = {} # To save best initialization for each model-fold combination
        else: # no penalization
            self.cvbt_folds_model = np.zeros((num_alpha, num_sigma, num_models, self.num_folds))
            self.cvbt_train_folds_model = np.zeros((num_alpha, num_sigma, num_models, self.num_folds))
            self.best_init_cvbt_dict = {} # To save best initialization for each model-fold combination


  def save_best_iter(self,
                       inpt_y: np.array, 
                       inpt_rt: np.array, 
                       y: np.array, 
                       session: list, 
                       rt: np.array, 
                       stim_onset: np.array,
                       session_fold_lookup_table,
                       C: int,
                       outcome_dict = None,
                       map_params: np.array = None):
        '''
        Create a matrix of size num_alpha x num_sigma x num_models x num_folds 
        containing normalized loglikelihood for train and test splits
        '''

        y = y.astype('int')
        if self.global_fit:
             tuning = 'global_fitting'
        else:
             tuning = 'individual_fitting'

        print("Retrieving best iter results for model = {}; num_folds = {}".format(str(self.model), str(self.num_folds)))
        for fold in tqdm(range(self.num_folds)):
            test_data, train_data, M_y, M_rt, n_test, n_train = \
                prepare_data_for_cv(inpt_y, inpt_rt, y, session, rt, stim_onset, 
                                    session_fold_lookup_table, fold,
                                    paramter_tuning=False)
            
            [test_inpt_y, test_inpt_rt, test_y, test_rt, test_stim_onset, test_mask, test_session] = test_data
            [train_inpt_y, train_inpt_rt, train_y, train_rt, train_stim_onset, train_mask, train_session] = train_data

            if len(test_inpt_y) > 0:
                ll0 = calculate_baseline_test_ll(
                    train_y[np.where(train_mask == 1)[0], :],
                    test_y[np.where(test_mask == 1)[0], :], C)
                ll0_train = calculate_baseline_test_ll(
                    train_y[np.where(train_mask == 1)[0], :],
                    train_y[np.where(train_mask == 1)[0], :], C)

                if self.model == "GLM_y":
                    _, _, = self.cross_validate_glm(fold,
                                                    test_inpt_y, train_inpt_y,
                                                    test_inpt_rt, train_inpt_rt,
                                                    test_y, train_y,
                                                    test_mask, train_mask,
                                                    test_rt, train_rt,
                                                    test_stim_onset, train_stim_onset,
                                                    ll0, ll0_train,
                                                    n_test, n_train,
                                                    M_y, C, outcome_dict)
                elif self.model == "GLM_HMM_y":
                    _, _, _ = self.cross_validate_glmhmm(fold,
                                                    test_inpt_y, train_inpt_y,
                                                    test_inpt_rt, train_inpt_rt,
                                                    test_y, train_y,
                                                    test_mask, train_mask,
                                                    test_rt, train_rt,
                                                    test_stim_onset, train_stim_onset,
                                                    test_session, train_session,
                                                    ll0, ll0_train,
                                                    n_test, n_train,
                                                    M_y, C, outcome_dict,
                                                    tuning=tuning,
                                                    map_params=map_params)
                    
                else:
                    raise NotImplementedError
            else:
              # meaning there is no test data for this fold
              self.cvbt_folds_model[:, :, fold] = None
              self.cvbt_train_folds_model[:, :, fold] = None          
              # self.cvbt_folds_model[:, :, :, fold] = None
              # self.cvbt_train_folds_model[:, :, :, fold] = None    
        # Save best initialization directories across animals, folds and models
        json_dump = json.dumps(self.best_init_cvbt_dict)
        f = open(self.results_dir / "best_init_cvbt_dict_{}.json".format(self.model), "w") # need to save for GLM?
        f.write(json_dump)
        f.close()
        # Save cvbt_folds_model as numpy array for easy parsing across all
        # models and folds
        np.savez(self.results_dir / "cvbt_folds_model_{}.npz".format(self.model), self.cvbt_folds_model)
        np.savez(self.results_dir / "cvbt_train_folds_model_{}.npz".format(self.model), self.cvbt_train_folds_model)

        print('Best init saved!')

  def cross_validate_glm(self,
                            fold: int,
                            test_inpt_y, train_inpt_y,
                            test_inpt_rt, train_inpt_rt,
                            test_y, train_y,
                            test_mask, train_mask,
                            test_rt, train_rt,
                            test_stim_onset, train_stim_onset,
                            ll0, ll0_train,
                            n_test, n_train,
                            M_y: int, C: int, outcome_dict = None):
        # Load parameters. Initerization does not matter for GLM
        glm_weights_file = self.results_dir / 'GLM' / ('fold_' + str(fold)) / (self.model + '_variables_of_interest_iter_0.npz')            
        
        # Instantiate a new GLM object with these parameters
        ll_glm = calculate_glm_test_loglikelihood(
            glm_weights_file, test_y[np.where(test_mask == 1)[0], :],
            test_inpt_y[np.where(test_mask == 1)[0], :], M_y, C, outcome_dict)
        ll_glm_train = calculate_glm_test_loglikelihood(
            glm_weights_file, train_y[np.where(train_mask == 1)[0], :],
            train_inpt_y[np.where(train_mask == 1)[0], :], M_y, C, outcome_dict)
        
        self.cvbt_folds_model[0, 0, 0, fold] = calculate_cv_bit_trial(
            ll_glm, ll0, n_test)
        self.cvbt_train_folds_model[0, 0, 0, fold] = calculate_cv_bit_trial(
            ll_glm_train, ll0_train, n_train)

        return self.cvbt_folds_model, self.cvbt_train_folds_model
  
  def cross_validate_glmhmm(self,
                               fold: int, # training
                               test_inpt_y, train_inpt_y,
                               test_inpt_rt, train_inpt_rt,
                               test_y, train_y,
                               test_mask, train_mask,
                               test_rt, train_rt,
                               test_stim_onset, train_stim_onset,
                               test_session, train_session,
                               ll0, ll0_train,
                               n_test, n_train,
                               M_y: int, C: int, outcome_dict = None,
                               tuning: str = '',
                               fold_tuning: int = None, # tuning
                               map_params: np.array = None):

            if fold_tuning is None:
                fold_tuning = fold
            test_inpt_y = np.hstack((test_inpt_y, np.ones((len(test_inpt_y), 1))))
            train_inpt_y = np.hstack((train_inpt_y, np.ones((len(train_inpt_y), 1))))

            # For GLM-HMM set values of y for violations to 2.  This value doesn't
            # matter (as mask will ensure that these y values do not contribute to
            # loglikelihood calculation
            test_y[np.where(test_mask == 0)[0], :] = 2
            train_y[np.where(train_mask == 0)[0], :] = 2

            # For GLM-HMM, need to partition data by session
            if tuning == 'l2_tuning':
                pass # see below
            else:
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

            for model_idx, K in enumerate(self.K_vals):
                if tuning == 'global_fitting':
                    dir_to_check = self.results_dir / ('GLM_HMM_y_K_' + str(K)) / ('fold_' + str(fold))
                    # Instantiate alpha and signa as well
                    lambda_value, _ = get_best_l2_params(map_params, animal='global', fold=fold, K=K)
                    self.Lambda_vals = [lambda_value]    
                elif tuning == 'map_tuning' or tuning == 'l2_tuning':
                     dir_to_check = self.results_dir / ('GLM_HMM_y_K_' + str(K)) / ('fold_' + str(fold)) / ('tuningfold_' + str(fold_tuning))
                elif tuning == 'individual_fitting':
                    dir_to_check = self.results_dir / ('GLM_HMM_y_K_' + str(K)) / ('fold_' + str(fold))
                    # Instantiate alpha and signa as well
                    sigma, alpha, _ = get_best_map_params(map_params, animal=self.animal, fold=fold, K=K)
                    self.Alpha_vals = [alpha]
                    self.Sigma_vals  = [sigma]           
                    # best_alpha = np.where map_params
                    if (len(self.Alpha_vals) > 1) or (len(self.Sigma_vals) > 1):
                         raise ValueError('Are you tuning parameters?')
                else:
                     raise NotImplementedError

                if tuning == 'l2_tuning' or tuning == 'global_fitting':
                    sigma = self.Sigma_vals[0]
                    alpha = self.Alpha_vals[0]
                    for L_idx, lambda_value in enumerate(self.Lambda_vals):
                        this_test_inputs, this_test_datas, this_test_masks = \
                            partition_data_by_session_L2(
                                test_inpt_y, test_y,
                                test_mask,
                                test_session,
                                lambda_value)
                        this_train_inputs, this_train_datas, this_train_masks = \
                            partition_data_by_session_L2(
                                train_inpt_y, train_y,
                                train_mask,
                                train_session,
                                lambda_value)
                
                        test_ll_vals_across_iters, init_ordering_by_train, file_ordering_by_train = \
                            self.get_best_glmhmm_init(dir_to_check,
                            this_test_datas, this_test_inputs, this_test_masks, 
                            'GLM_HMM_y_raw_parameters_itr_', sigma, alpha, K, self.D, M_y, C, 
                            True, lambda_value)

                        train_ll_vals_across_iters, _, _ = \
                            self.get_best_glmhmm_init(dir_to_check,
                            this_train_datas, this_train_inputs, this_train_masks,
                            'GLM_HMM_y_raw_parameters_itr_', sigma, alpha, K, self.D, M_y, C,
                            True, lambda_value)

                        # Sort and find the best lls
                        test_ll_vals_across_iters = test_ll_vals_across_iters[
                            file_ordering_by_train]
                        train_ll_vals_across_iters = train_ll_vals_across_iters[
                            file_ordering_by_train]

                        cvbt_thismodel_thisfold = calculate_cv_bit_trial(test_ll_vals_across_iters[0], ll0,
                                                                        n_test)
                        train_cvbt_thismodel_thisfold = calculate_cv_bit_trial(
                            train_ll_vals_across_iters[0], ll0_train, n_train)
                        
                        self.cvbt_folds_model[L_idx, model_idx, fold_tuning] = cvbt_thismodel_thisfold
                        self.cvbt_train_folds_model[L_idx, model_idx, fold_tuning] = train_cvbt_thismodel_thisfold
                            
                        # Save best initialization number to dictionary:
                        key_for_dict = 'GLM_HMM_y_K_' + str(K) + '/fold_' + str(fold) \
                                        + '/alpha_' + str(alpha) + '/sigma_' + str(sigma) + '/lambda_' + str(lambda_value) \
                                        + '/fold_tuning_' + str(fold_tuning) 
                        self.best_init_cvbt_dict[key_for_dict] = int(init_ordering_by_train[0])
                else: # no regularization
                    for A_idx, alpha in enumerate(self.Alpha_vals):
                        for S_idx, sigma in enumerate(self.Sigma_vals):
                            test_ll_vals_across_iters, init_ordering_by_train, file_ordering_by_train = \
                                self.get_best_glmhmm_init(dir_to_check,
                                this_test_datas, this_test_inputs, this_test_masks, 
                                'GLM_HMM_y_raw_parameters_itr_', sigma, alpha, K, self.D, M_y, C)

                            train_ll_vals_across_iters, _, _ = \
                                self.get_best_glmhmm_init(dir_to_check,
                                this_train_datas, this_train_inputs, this_train_masks,
                                'GLM_HMM_y_raw_parameters_itr_', sigma, alpha, K, self.D, M_y, C)
    
                            # Sort and find the best lls
                            test_ll_vals_across_iters = test_ll_vals_across_iters[
                                file_ordering_by_train]
                            train_ll_vals_across_iters = train_ll_vals_across_iters[
                                file_ordering_by_train]

                            cvbt_thismodel_thisfold = calculate_cv_bit_trial(test_ll_vals_across_iters[0], ll0,
                                                                            n_test)
                            train_cvbt_thismodel_thisfold = calculate_cv_bit_trial(
                                train_ll_vals_across_iters[0], ll0_train, n_train)
                            
                            self.cvbt_folds_model[A_idx, S_idx, model_idx, fold_tuning] = cvbt_thismodel_thisfold
                            self.cvbt_train_folds_model[A_idx, S_idx, model_idx, fold_tuning] = train_cvbt_thismodel_thisfold
                                
                            # Save best initialization number to dictionary:
                            key_for_dict = 'GLM_HMM_y_K_' + str(K) + '/fold_' + str(fold) \
                                            + '/alpha_' + str(alpha) + '/sigma_' + str(sigma) \
                                            + '/fold_tuning_' + str(fold_tuning) 
                            self.best_init_cvbt_dict[key_for_dict] = int(init_ordering_by_train[0])

            return self.cvbt_folds_model, self.cvbt_train_folds_model, \
                    self.best_init_cvbt_dict

  def get_best_glmhmm_init(self,
                              dir_to_check,
                              this_datas, this_inputs, this_masks,
                              fname_header,
                              sigma, alpha, K, D, M, C,
                              IsRegularization=False, lambda_value = None):
        """
        Find the best initialization for GLM-HMM model given K, sigma, alpha
        """
        if IsRegularization:
            test_ll_vals_across_iters, train_ll_vals_across_iters, \
            alpha_vals_across_iters, sigma_vals_across_iters, \
            lambda_vals_across_iters, glmhmm_iters, glmhmm_files = \
                calculate_glmhmm_test_loglikelihood_L2(dir_to_check, 
                                                    fname_header, 
                                                    this_datas, this_inputs, this_masks,
                                                    K, D, M, C)
            # print(test_ll_vals_across_iters)
            # Find initialization for the params of interests
            idx = np.intersect1d(np.where(alpha_vals_across_iters == alpha), 
                                np.where(sigma_vals_across_iters == sigma),
                                np.where(lambda_vals_across_iters == lambda_value))
            test_ll_vals_across_iters_i = test_ll_vals_across_iters[idx]
            train_ll_vals_across_iters_i = train_ll_vals_across_iters[idx]
            glmhmm_iters_i = glmhmm_iters[idx]
        else:
            test_ll_vals_across_iters, train_ll_vals_across_iters, \
            alpha_vals_across_iters, sigma_vals_across_iters, \
            glmhmm_iters, glmhmm_files = \
                calculate_glmhmm_test_loglikelihood(dir_to_check, 
                                                    fname_header, 
                                                    this_datas, this_inputs, this_masks,
                                                    K, D, M, C)
            
            # Find initialization for the params of interests
            idx = np.intersect1d(np.where(alpha_vals_across_iters == alpha), 
                                np.where(sigma_vals_across_iters == sigma))
            test_ll_vals_across_iters_i = test_ll_vals_across_iters[idx]
            train_ll_vals_across_iters_i = train_ll_vals_across_iters[idx]
            glmhmm_iters_i = glmhmm_iters[idx]
        # Order raw files by train LL (don't train on test data!):
        file_ordering_by_train = np.argsort(-train_ll_vals_across_iters_i)
        # Get initialization number from raw_file ordering
        init_ordering_by_train = np.array(glmhmm_iters_i)[file_ordering_by_train]

        return test_ll_vals_across_iters_i, init_ordering_by_train, \
            file_ordering_by_train

  def save_best_MAPparams(self,
                          inpt_y: np.array, 
                          inpt_rt: np.array, 
                          y: np.array, 
                          session: list, 
                          rt: np.array, 
                          stim_onset: np.array,
                          session_fold_lookup_table,
                          this_fold_training: int,
                          C: int,
                          outcome_dict = None,
                          save_output=False):
        '''
        Create a matrix of size num_alpha x num_sigma x num_models x num_folds 
        containing normalized loglikelihood for train and test splits 
        within training dataset
        '''

        y = y.astype('int')

        print("Retrieving best iter results for model = {}; fold = {}; num_folds = {}".format(str(self.model), str(this_fold_training), str(self.num_folds)))
        for fold_tuning in tqdm(range(self.num_folds)):
            test_data, train_data, M_y, M_rt, n_test, n_train = \
                prepare_data_for_cv(inpt_y, inpt_rt, y, session, rt, stim_onset, 
                                    session_fold_lookup_table, this_fold_training, 
                                    fold_tuning, paramter_tuning=True)
            
            [test_inpt_y, test_inpt_rt, test_y, test_rt, test_stim_onset, test_mask, test_session] = test_data
            [train_inpt_y, train_inpt_rt, train_y, train_rt, train_stim_onset, train_mask, train_session] = train_data

            if len(test_inpt_y) > 0:
                ll0 = calculate_baseline_test_ll(
                    train_y[np.where(train_mask == 1)[0], :],
                    test_y[np.where(test_mask == 1)[0], :], C)
                ll0_train = calculate_baseline_test_ll(
                    train_y[np.where(train_mask == 1)[0], :],
                    train_y[np.where(train_mask == 1)[0], :], C)

                if self.model == "GLM_HMM_y":
                    _, _, _ = self.cross_validate_glmhmm(this_fold_training,
                                                        test_inpt_y, train_inpt_y,
                                                        test_inpt_rt, train_inpt_rt,
                                                        test_y, train_y,
                                                        test_mask, train_mask,
                                                        test_rt, train_rt,
                                                        test_stim_onset, train_stim_onset,
                                                        test_session, train_session,
                                                        ll0, ll0_train,
                                                        n_test, n_train,
                                                        M_y, C, outcome_dict,
                                                        tuning='map_tuning',
                                                        fold_tuning=fold_tuning)
                    
                else:
                    raise NotImplementedError
            else:
                 # meaning there is no test data for this fold
                 self.cvbt_folds_model[:, :, :, fold_tuning] = None
                 self.cvbt_train_folds_model[:, :, :, fold_tuning] = None
            
        print("Calculating best MAP parameters...")
        best_params = []
        for model_idx, K in enumerate(self.K_vals):
             this_cvbt_folds_model = self.cvbt_folds_model[:,:,model_idx,:] # test dataset
             mean_cvbt = np.mean(this_cvbt_folds_model, axis=2) # fold is stored in the last axis
             this_cvbt_train_folds_model = self.cvbt_train_folds_model[:,:,model_idx,:]
             mean_train_cvbt = np.mean(this_cvbt_train_folds_model, axis=2)
             assert mean_cvbt.ndim == 2, 'Wrong shape of cvbt_folds_model'
        
             best_idx = np.unravel_index(mean_cvbt.argmax(), mean_cvbt.shape)
             best_alpha_idx = best_idx[0]
             best_alpha = self.Alpha_vals[best_alpha_idx]
             best_sigma_idx = best_idx[1]
             best_sigma = self.Sigma_vals[best_sigma_idx]

             this_best_params = [K, best_alpha, best_sigma, best_alpha_idx, best_sigma_idx]
             best_params.append(this_best_params)

             if save_output:
                saving_dir = self.results_dir / ('GLM_HMM_y_K_' + str(K)) / ('fold_' + str(this_fold_training))
                # Save best initialization directories across animals, folds and models
                json_dump = json.dumps(self.best_init_cvbt_dict)
                f = open(saving_dir / "map_tuning_cvbt_dict_{}.json".format(self.model), "w") # need to save for GLM?
                f.write(json_dump)
                f.close()
                # Save cvbt_folds_model as numpy array for easy parsing across all
                # models and folds
                np.savez(saving_dir / "map_tuning_cvbt_folds_model_{}.npz".format(self.model), mean_cvbt)
                np.savez(saving_dir / "map_tuning_cvbt_train_folds_model_{}.npz".format(self.model), mean_train_cvbt)
                plot_best_MAP_params(mean_cvbt, 
                                      self.Alpha_vals, self.Sigma_vals,
                                      saving_dir, save_title='map_tuning_model_{}'.format(self.model))


        self._reset()
        return best_params

  def save_best_L2params(self,
                          inpt_y: np.array, 
                          inpt_rt: np.array, 
                          y: np.array, 
                          session: list, 
                          rt: np.array, 
                          stim_onset: np.array,
                          session_fold_lookup_table,
                          this_fold_training: int, # training
                          C: int,
                          outcome_dict = None,
                          save_output=False):
        '''
        Create a matrix of size num_lambda x num_models x num_folds 
        containing normalized loglikelihood for train and test splits 
        within training dataset
        '''

        y = y.astype('int')

        print("Retrieving best iter results for model = {}; fold = {}; num_folds = {}".format(str(self.model), str(this_fold_training), str(self.num_folds)))
        for fold_tuning in tqdm(range(self.num_folds)): # tuning
            test_data, train_data, M_y, M_rt, n_test, n_train = \
                prepare_data_for_cv(inpt_y, inpt_rt, y, session, rt, stim_onset, 
                                    session_fold_lookup_table, this_fold_training, 
                                    fold_tuning, paramter_tuning=True)
            
            [test_inpt_y, test_inpt_rt, test_y, test_rt, test_stim_onset, test_mask, test_session] = test_data
            [train_inpt_y, train_inpt_rt, train_y, train_rt, train_stim_onset, train_mask, train_session] = train_data

            if len(test_inpt_y) > 0:
                ll0 = calculate_baseline_test_ll(
                    train_y[np.where(train_mask == 1)[0], :],
                    test_y[np.where(test_mask == 1)[0], :], C)
                ll0_train = calculate_baseline_test_ll(
                    train_y[np.where(train_mask == 1)[0], :],
                    train_y[np.where(train_mask == 1)[0], :], C)

                if self.model == "GLM_HMM_y":
                    _, _, _ = self.cross_validate_glmhmm(this_fold_training,
                                                        test_inpt_y, train_inpt_y,
                                                        test_inpt_rt, train_inpt_rt,
                                                        test_y, train_y,
                                                        test_mask, train_mask,
                                                        test_rt, train_rt,
                                                        test_stim_onset, train_stim_onset,
                                                        test_session, train_session,
                                                        ll0, ll0_train,
                                                        n_test, n_train,
                                                        M_y, C, outcome_dict,
                                                        tuning='l2_tuning',
                                                        fold_tuning=fold_tuning)
                    
                else:
                    raise NotImplementedError
            else:
                 # meaning there is no test data for this fold
                 self.cvbt_folds_model[:, :, fold_tuning] = None
                 self.cvbt_train_folds_model[:, :, fold_tuning] = None

        print("Calculating best L2 parameters...")
        best_params = []
        for model_idx, K in enumerate(self.K_vals):
             this_cvbt_folds_model = self.cvbt_folds_model[:,model_idx,:] # test dataset
             print(this_cvbt_folds_model)
             mean_cvbt = np.mean(this_cvbt_folds_model, axis=1) # fold is stored in the last axis
             this_cvbt_train_folds_model = self.cvbt_train_folds_model[:,model_idx,:]
             mean_train_cvbt = np.mean(this_cvbt_train_folds_model, axis=1)
             print(mean_train_cvbt)
             assert mean_cvbt.ndim == 1,'Wrong shape of cvbt_folds_model'
        
             best_idx = np.unravel_index(mean_cvbt.argmax(), mean_cvbt.shape)
             print(best_idx)
             best_lambda_idx = best_idx[0]
             best_lambda = self.Lambda_vals[best_lambda_idx]

             this_best_params = [K, best_lambda, best_lambda_idx]
             best_params.append(this_best_params)

             if save_output:
                saving_dir = self.results_dir / ('GLM_HMM_y_K_' + str(K)) / ('fold_' + str(this_fold_training))
                # Save best initialization directories across animals, folds and models
                json_dump = json.dumps(self.best_init_cvbt_dict)
                f = open(saving_dir / "l2_tuning_cvbt_dict_{}.json".format(self.model), "w") # need to save for GLM?
                f.write(json_dump)
                f.close()
                # Save cvbt_folds_model as numpy array for easy parsing across all
                # models and folds
                np.savez(saving_dir / "l2_tuning_cvbt_folds_model_{}.npz".format(self.model), mean_cvbt)
                np.savez(saving_dir / "l2_tuning_cvbt_train_folds_model_{}.npz".format(self.model), mean_train_cvbt)
                # plot_best_l2_params(mean_cvbt, 
                #                       self.Alpha_vals, self.Sigma_vals,
                #                       saving_dir, save_title='l2_tuning_model_{}'.format(self.model))


        self._reset()
        return best_params

def prepare_data_for_cv(inpt_y, inpt_rt, y, session, rt, stim_onset,
                        session_fold_lookup_table, fold, 
                        fold_tuning=None, paramter_tuning=False):

    abort_idx = np.where(y == 3)[0]
    nonabort_idx, nonabort_mask = create_abort_mask(
        abort_idx, inpt_y.shape[0])
    # Load train and test data for session
    test_data, train_data, M_y, M_rt, n_test, n_train = get_train_test_dta(inpt_y, inpt_rt,
                                                                             y, rt, stim_onset, 
                                                                             nonabort_mask, session, 
                                                                             session_fold_lookup_table, 
                                                                             fold, fold_tuning, 
                                                                             paramter_tuning=paramter_tuning)

    return test_data, train_data, M_y, M_rt, n_test, n_train
        
def get_train_test_dta(inpt_y, inpt_rt, y, rt, stim_onset, mask, session, 
                       session_fold_lookup_table, fold,
                       fold_tuning, paramter_tuning=False):
    '''
    Split inpt_y, inpt_rt, y, rt, stim_onset, session arrays into train and test arrays
    '''
    # Use fold != fold for trainig and fold == fold for test dataset
    if not paramter_tuning:
        test_sessions = session_fold_lookup_table[np.where(
            session_fold_lookup_table[:, 1] == fold), 0]
        train_sessions = session_fold_lookup_table[np.where(
            session_fold_lookup_table[:, 1] != fold), 0]
    else:
        test_split_idx = np.logical_and(session_fold_lookup_table[:, 1] != fold, \
                                        session_fold_lookup_table[:, 2] == fold_tuning)
        test_sessions = session_fold_lookup_table[test_split_idx, 0]
    
        train_split_idx = np.logical_and(session_fold_lookup_table[:, 1] != fold, \
                                        session_fold_lookup_table[:, 2] != fold_tuning)
        train_sessions = session_fold_lookup_table[train_split_idx, 0]
    
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
    
    M_y = train_inpt_y.shape[1] # test or train does not matter
    M_rt = train_inpt_rt.shape[1] # test or train does not matter
    n_test = np.sum(test_mask == 1)
    n_train = np.sum(train_mask == 1)
    return [test_inpt_y, test_inpt_rt, test_y, test_rt, test_stim_onset, test_mask, test_session], \
            [train_inpt_y, train_inpt_rt, train_y, train_rt, train_stim_onset, train_mask, train_session], \
            M_y, M_rt, n_test, n_train


def calculate_baseline_test_ll(train_y, test_y, C):
    """
    Calculate baseline loglikelihood for cross-validation. 
    While the choices follow multinomial conditional distribution because of outcome hierarchy,
    one can simply ignore the conditional probability and assume they follow a very simple
    multinominal distribution, as we observe all the oucomes in this definition of baseline 
    loglikelihood. loglikelihood L will be calculated with
    L = sum(n_i(log(p_i)))
    where p_i is the proportion of trials in which the animal took choice i
    in the training set and n_i is the number of trials in which the animal took choice i
    in the test set

    :return: baseline loglikelihood for cross-validation
    """
    _, train_class_totals = np.unique(train_y, return_counts=True)
    train_class_probs = train_class_totals / train_y.shape[0] # calculate proportions
    _, test_class_totals = np.unique(test_y, return_counts=True)
    ll0 = 0
    for c in range(C):
        ll0 += test_class_totals[c] * np.log(train_class_probs[c])
    return ll0

def calculate_glm_test_loglikelihood(glm_weights_file, test_y, test_inpt, 
                                      M, C, outcome_dict):
    _, glm_vectors = load_glm_vectors(glm_weights_file)
    # Calculate test loglikelihood
    new_glm = glm(M, C, outcome_dict, obs='Categorical') # multinomial distribution
    # Set parameters to fit parameters:
    new_glm.Wk = glm_vectors
    # Get loglikelihood of training data:
    loglikelihood_test = new_glm.log_marginal([test_y], [test_inpt], None, None)
    # loglikelihood_train = new_glm.log_marginal([test_y], [test_inpt], [test_mask], None)
    return loglikelihood_test

def calculate_cv_bit_trial(ll_model, ll_0, n_trials):
    """
    See Eq. 22 of Ashwood et al., 2022, Nat, Neurosci.
    """
    cv_bit_trial = ((ll_model - ll_0) / n_trials) / np.log(2)
    return cv_bit_trial

def calculate_glmhmm_test_loglikelihood(glm_hmm_dir, fname_header, test_datas, test_inputs,
                                         test_masks, K, D, M, C):
    """
    Calculate test loglikelihood for GLM-HMM model. Loop through all
    initializations for fold of interest.
    """
    glmhmm_outputs = scan_glmhmm_output(glm_hmm_dir, fname_header=fname_header)
    glmhmm_iters = glmhmm_outputs[0]
    glmhmm_files = glmhmm_outputs[1]

    train_ll_vals_across_iters = []
    test_ll_vals_across_iters = []
    alpha_vals_across_iters = []
    sigma_vals_across_iters = []
    for file in glmhmm_files:
        # Loop through initializations and calculate BIC:
        this_hmm_params, lls, alpha, sigma, global_fit = \
            load_glmhmm_data(file)
        train_ll_vals_across_iters.append(lls[-1])
        # Instantiate a new HMM and calculate test loglikelihood:
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs_multinominal",
                           observation_kwargs=dict(C=C),
                           transitions="standard")
        if K ==1:
             this_hmm.observations.params = this_hmm_params
        elif K > 1:
            this_hmm.params = this_hmm_params
        test_ll = this_hmm.log_likelihood(test_datas,
                                           inputs=test_inputs,
                                           masks=test_masks)
        

        test_ll_vals_across_iters.append(test_ll)
        # Save MAP/MLE hyperparameters as well
        alpha_vals_across_iters.append(alpha)
        sigma_vals_across_iters.append(sigma)

    return np.array(test_ll_vals_across_iters), np.array(train_ll_vals_across_iters), \
            np.array(alpha_vals_across_iters), np.array(sigma_vals_across_iters), \
                np.array(glmhmm_iters), np.array(glmhmm_files)

def calculate_glmhmm_test_loglikelihood_L2(glm_hmm_dir, fname_header, test_datas, test_inputs,
                                         test_masks, K, D, M, C):
    """
    Calculate test loglikelihood for GLM-HMM model. Loop through all
    initializations for fold of interest.
    """
    glmhmm_outputs = scan_glmhmm_output(glm_hmm_dir, fname_header=fname_header)
    glmhmm_iters = glmhmm_outputs[0]
    glmhmm_files = glmhmm_outputs[1]

    train_ll_vals_across_iters = []
    test_ll_vals_across_iters = []
    alpha_vals_across_iters = []
    sigma_vals_across_iters = []
    lambda_vals_across_iters = []
    for file in glmhmm_files:
        # Loop through initializations and calculate BIC:
        this_hmm_params, lls, alpha, sigma, global_fit, l, reguralization = \
            load_glmhmm_data(file, True)
        train_ll_vals_across_iters.append(lls[-1])
        # Instantiate a new HMM and calculate test loglikelihood:
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs_multinominal",
                           observation_kwargs=dict(C=C),
                           transitions="standard")
        if K ==1:
             this_hmm.observations.params = this_hmm_params
        elif K > 1:
            this_hmm.params = this_hmm_params
        test_ll = this_hmm.log_likelihood(test_datas,
                                           inputs=test_inputs,
                                           masks=test_masks)
        

        test_ll_vals_across_iters.append(test_ll)
        # Save MAP/MLE hyperparameters as well
        alpha_vals_across_iters.append(alpha)
        sigma_vals_across_iters.append(sigma)
        lambda_vals_across_iters.append(l)

    return np.array(test_ll_vals_across_iters), np.array(train_ll_vals_across_iters), \
            np.array(alpha_vals_across_iters), np.array(sigma_vals_across_iters), \
            np.array(lambda_vals_across_iters), np.array(glmhmm_iters), np.array(glmhmm_files)