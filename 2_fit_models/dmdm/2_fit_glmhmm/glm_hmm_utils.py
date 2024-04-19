# Functions to assist with GLM-HMM model fitting
import sys
import ssm
import autograd.numpy as np
import numpy as onp

def fit_glm_hmm(datas, inputs, masks, K, D, M, C, N_em_iters,
                transition_alpha, prior_sigma, global_fit,
                reguralization, l, params_for_initialization, save_title) -> None:
    '''
    Instantiate and fit GLM-HMM model
    :param datas:
    :param inputs:
    :param masks:
    :param K:
    :param D:
    :param M:
    :param C:
    :param N_em_iters:
    :param global_fit:
    :param glm_vectors:
    :param save_title:
    :return:
    '''
    if global_fit == True:
        # Prior variables
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs_multinominal",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize observation weights as GLM weights with some noise:
        glm_vectors_repeated = np.tile(params_for_initialization, (K, 1, 1))
        glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(
            0, 0.2, glm_vectors_repeated.shape)
        this_hmm.observations.params = glm_vectors_with_noise
    else:
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs_multinominal",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        print(params_for_initialization)
        # Initialize HMM-GLM with global parameters:
        if K == 1:
            this_hmm.observations.params = params_for_initialization
        elif K > 1:
            this_hmm.params = params_for_initialization
        # Get log_prior of transitions:
    print("=== fitting GLM-HMM ========")
    sys.stdout.flush()
    # Fit this HMM and calculate marginal likelihood
    lls = this_hmm.fit(datas,
                       inputs=inputs,
                       masks=masks,
                       method="em",
                       num_iters=N_em_iters,
                       initialize=False,
                       tolerance=10 ** -4)

    # Save raw parameters of HMM, loglikelihood during entire training, as well as the MAP parameters
    if K == 1:
        recovered_params = this_hmm.observations.params
        # Avoid np.savez error. When K = 1 there is no transition.
    elif K > 1:
        recovered_params = this_hmm.params
    else:
        raise ValueError('K should be >= 1')
    
    file_header = '_a' + str(int(transition_alpha*100)) + '_s' +  str(int(prior_sigma*100)) + '_l' +  str(int(l*100))
    np.savez(str(save_title) + file_header + '.npz', 
             recovered_params, lls, transition_alpha, prior_sigma, global_fit, l, reguralization)

    return None