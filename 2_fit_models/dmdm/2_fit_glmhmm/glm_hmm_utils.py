# Functions to assist with GLM-HMM model fitting
import sys
import ssm
import autograd.numpy as np
import numpy as onp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def fit_glm_hmm(datas, inputs, masks, K, D, M, C, N_em_iters,
                transition_alpha, prior_sigma, global_fit,
                params_for_initialization, save_title):
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
        # Initialize HMM-GLM with global parameters:
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
    # Save raw parameters of HMM, as well as loglikelihood during training
    np.savez(save_title, this_hmm.params, lls)

    return None

def create_cv_frame_for_plotting(cvbt_folds_model):
    # Identify best cvbt:
    mean_cvbt = onp.mean(cvbt_folds_model, axis=1)
    loc_best = onp.where(mean_cvbt == max(mean_cvbt))[0]
    best_val = max(mean_cvbt)
    # Create dataframe for plotting
    num_models = cvbt_folds_model.shape[0]
    num_folds = cvbt_folds_model.shape[1]
    # Create pandas dataframe:
    data_for_plotting_df = pd.DataFrame({
        'model':
            onp.repeat(onp.arange(num_models), num_folds),
        'cv_bit_trial':
            cvbt_folds_model.flatten()
    })
    return data_for_plotting_df, loc_best, best_val

def plot_states(weight_vectors,
                log_transition_matrix,
                figure_directory,
                K,
                save_title='best_params_cross_validation_K_',
                labels_for_plot=[],
                cols_K = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"],
                cols = ["#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc","#96f97b"]):

    M = weight_vectors.shape[2] - 1
    
    fig = plt.figure(figsize=((K+1)*10, 10),
                        dpi=80,
                        facecolor='w',
                        edgecolor='k')
    plt.subplots_adjust(left=0.1,
                        bottom=0.24,
                        right=0.95,
                        top=0.7,
                        wspace=0.8,
                        hspace=0.5)
    
    plt.subplot(1, K + 1, 1)
    transition_matrix = onp.exp(log_transition_matrix)
    plt.imshow(transition_matrix, vmin=0, vmax=1)
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            text = plt.text(j,
                            i,
                            onp.around(transition_matrix[i, j],
                                        decimals=3),
                            ha="center",
                            va="center",
                            color="k",
                            fontsize=30)
    plt.ylabel("Previous State", fontsize=30)
    plt.xlabel("Next State", fontsize=30)
    plt.colorbar()
    plt.xlim(-0.5, K - 0.5)
    plt.ylim(-0.5, K - 0.5)
    plt.xticks(range(0, K), ('1', '2', '3', '4', '4', '5', '6', '7',
                                '8', '9', '10')[:K],
                fontsize=30)
    plt.yticks(range(0, K), ('1', '2', '3', '4', '4', '5', '6', '7',
                                '8', '9', '10')[:K],
                fontsize=30)
    plt.title("Retrieved", fontsize=40)

    choice_label_mapping = {0: 'miss', 1: 'Hit', 2: 'FA', 3: 'abort'}
    for k in range(K):
        Ws = weight_vectors[k]
        K_prime = Ws.shape[0] # C
        plt.subplot(1, K + 1, 2 + k)
        for j in range(K_prime): # each category 
            l = plt.plot(range(M + 1), 
                         Ws[j], # plot weights with orginal signs
                         marker='o',
                         label=choice_label_mapping[j],
                         lw=4)

        plt.xticks(list(range(0, len(labels_for_plot))),
                    labels_for_plot,
                    rotation='20',
                    fontsize=24)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        # plt.ylim((-3, 14))
        plt.ylabel("Weight", fontsize=30)
        plt.xlabel("Covariate", fontsize=30, labelpad=20)
        plt.title("y GLM: State Zk =" + str(k + 1), fontsize=40)

    fig.tight_layout()
    fig.savefig(figure_directory / (save_title + str(K) + '.png'))
    plt.axis('off')
    plt.close(fig)


def plot_model_comparison(cv,
                          cv_train,
                          K_vals,
                          figure_directory,
                          save_title='best_params_performance_',
                          cols = ["#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc","#96f97b"]):

    data_for_plotting_df, loc_best, best_val = \
        create_cv_frame_for_plotting(cv)
    train_data_for_plotting_df, train_loc_best, train_best_val, \
          = create_cv_frame_for_plotting(cv_train)
    
    fig = plt.figure(figsize=(10, 10),
                        dpi=80,
                        facecolor='w',
                        edgecolor='k')

    g = sns.lineplot(data_for_plotting_df['model'],
        data_for_plotting_df['cv_bit_trial'],
        err_style="bars",
        mew=0,
        color=cols[0],
        marker='o',
        ci=68,
        label="test",
        alpha=1,
        lw=4)
    sns.lineplot(
        train_data_for_plotting_df['model'],
        train_data_for_plotting_df['cv_bit_trial'],
        err_style="bars",
        mew=0,
        color=cols[1],
        marker='o',
        ci=68,
        label="train",
        alpha=1,
        lw=4)
    plt.xlabel("Model", fontsize=30)
    plt.ylabel("Normalized LL", fontsize=30)
    plt.xticks(range(len(K_vals)),
                [str(k) + ' states' for k in K_vals],
                rotation=45,
                fontsize=24)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right', fontsize=30)
    plt.tick_params(axis='y')
    # plt.yticks([0.2, 0.3, 0.4, 0.5], fontsize=30)
    # plt.ylim((0.2, 0.55))
    plt.title("Model Comparison", fontsize=40)

    fig.savefig(figure_directory / (save_title + '.png'))
    plt.axis('off')
    plt.close(fig)