import sys
import numpy as onp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sys.path.append('../2_fit_models/dmdm/') # a lazy trick to search parent dir
from data_labels import create_abort_mask, partition_data_by_session
from data_postprocessing_utils import get_marginal_posterior
from data_io import load_best_map_params

def create_cv_frame_for_plotting(cvbt_folds_model):
    """
    Very similar to data_io.get_file_name_for_best_glmhmm_fold
    But the aim here is to get the dataframe instead of the file name.
    If we know alpha and sigma, and state K, we should be able to retrieve the
    best loglikelihood and which fold has that best loglikelihood again.
    """
    assert cvbt_folds_model.ndim == 4, 'Wrong shape of cvbt_folds_model'
    cvbt_folds_model_tuned = cvbt_folds_model[0,0,:,:]
    assert cvbt_folds_model_tuned.ndim == 2, 'Shape of cvbt_folds_model wrongly processed'
    # Identify best cvbt:
    mean_cvbt = onp.mean(cvbt_folds_model_tuned, axis=1)
    loc_best_K = onp.where(mean_cvbt == max(mean_cvbt))[0]
    val_best_K = max(mean_cvbt)
    # Create dataframe for plotting
    num_models = cvbt_folds_model_tuned.shape[0]
    num_folds = cvbt_folds_model_tuned.shape[1]
    # Create pandas dataframe:
    data_for_plotting_df = pd.DataFrame({
        'model':
            onp.repeat(onp.arange(num_models), num_folds),
        'cv_bit_trial':
            cvbt_folds_model_tuned.flatten()
    })

    return data_for_plotting_df, loc_best_K, val_best_K

def get_file_name_for_best_glmhmm_fold(cvbt_folds_model, model_idx, K, 
                                        alpha_idx, alpha, sigma_idx, sigma, overall_dir: Path,
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
    # loc_best = 0
    best_fold = onp.where(cvbt_folds_model[alpha_idx, sigma_idx, model_idx, :] == \
                         max(cvbt_folds_model[alpha_idx, sigma_idx, model_idx, :]))[0][0]
    base_path = overall_dir / (model +'_K_' + str(K)) / ('fold_' + str(best_fold))
    key_for_dict = model +'_K_' + str(K) + '/fold_' + str(best_fold) \
                        + '/alpha_' + str(alpha) + '/sigma_' + str(sigma)
    best_iter = best_init_cvbt_dict[key_for_dict]

    fname_tail = '_a' + str(int(alpha*100)) + '_s' +  str(int(sigma*100)) + '.npz'
    fpath = base_path / ('iter_' + str(best_iter)) / (fname_header + str(best_iter) + fname_tail)
    return fpath, best_fold

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
    plt.imshow(transition_matrix, vmin=0, vmax=1, cmap='hot')
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
                          global_fit,
                          K_vals,
                          figure_directory,
                          save_title='best_params_performance',
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

def plot_state_occupancy(inpt_y, inpt_rt, y, session, rt, stim_onset,
                         K, hmm_params,
                         animal_name,
                         figure_directory,
                         save_title='best_params_state_occupancy',
                         cols = ["#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc","#96f97b"]):

    # get state occupancies:
    if K == 1:
        raise ValueError('K needs to be > 1')
    inpt_y = onp.hstack((inpt_y, onp.ones((len(inpt_y),1))))
    y = y.astype('int')
    abort_idx = onp.where(y == 3)[0]
    nonviolation_idx, mask = create_abort_mask(abort_idx, inpt_y.shape[0])

    y[onp.where(y == 3), :] = 2
    inputs, datas, masks = partition_data_by_session(
        inpt_y, y, mask, session)

    C = hmm_params[2].shape[1] 
    posterior_probs = get_marginal_posterior(inputs, datas, masks, C,
                                             hmm_params, K, range(K))
    states_max_posterior = onp.argmax(posterior_probs, axis=1)

    fig = plt.figure(figsize=(10, 10),
                    dpi=80,
                    facecolor='w',
                    edgecolor='k')   
    plt.hist(states_max_posterior)
    plt.ylabel("# trials", fontsize=30)
    plt.xlabel("State", fontsize=30)
    plt.xticks(range(K),
                ['State ' + str(k + 1) for k in range(K)],
                rotation=45,
                fontsize=24)
    plt.yticks(fontsize=30)
    plt.title("State occuupancies", fontsize=40)
    fig.suptitle(animal_name, fontsize=40)

    fig.savefig(figure_directory / (save_title + str(K) + '.png'))
    plt.axis('off')
    plt.close(fig)


def plot_state_prob(inpt_y, inpt_rt, y, session, rt, stim_onset,
                         K, hmm_params,
                         animal_name,
                         figure_directory,
                         save_title='best_params_state_occupancy',
                         cols = ["#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc","#96f97b"]):
    
    fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
    sess_id = 0 #session id; can choose any index between 0 and num_sess-1
    for k in range(num_states):
        plt.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2,
                color=cols[k])
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize = 10)
    plt.xlabel("trial #", fontsize = 15)
    plt.ylabel("p(state)", fontsize = 15)

def plot_state_dwelltime(inpt_y, inpt_rt, y, session, rt, stim_onset,
                         K, hmm_params,
                         animal_name,
                         figure_directory,
                         save_title='best_params_state_occupancy',
                         cols = ["#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc","#96f97b"]):
    return None