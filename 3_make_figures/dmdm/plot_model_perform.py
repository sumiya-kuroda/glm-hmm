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


def create_cv_frame_for_plotting_l2(cvbt_folds_model):
    """
    Very similar to data_io.get_file_name_for_best_glmhmm_fold
    But the aim here is to get the dataframe instead of the file name.
    If we know alpha and sigma, and state K, we should be able to retrieve the
    best loglikelihood and which fold has that best loglikelihood again.
    """
    assert cvbt_folds_model.ndim == 3, 'Wrong shape of cvbt_folds_model'
    cvbt_folds_model_tuned = cvbt_folds_model[0,:,:]
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
                          y_label="Normalized LL",
                          save_title='best_params_performance_nll',
                          cols = ["#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc","#96f97b"]):

    data_for_plotting_df, loc_best, best_val = \
        create_cv_frame_for_plotting(cv)
    train_data_for_plotting_df, train_loc_best, train_best_val, \
          = create_cv_frame_for_plotting(cv_train)
    
    fig = plt.figure(figsize=(10, 10),
                        dpi=80,
                        facecolor='w',
                        edgecolor='k')

    g = sns.lineplot(x=data_for_plotting_df['model'],
        y=data_for_plotting_df['cv_bit_trial'],
        err_style="bars",
        errorbar=('se', 1.96),
        mew=0,
        color=cols[0],
        marker='o',
        ci=68,
        label="test",
        alpha=1,
        lw=4)
    sns.lineplot(
        x=train_data_for_plotting_df['model'],
        y=train_data_for_plotting_df['cv_bit_trial'],
        err_style="bars",
        errorbar=('se', 1.96),
        mew=0,
        color=cols[1],
        marker='o',
        ci=68,
        label="train",
        alpha=1,
        lw=4)
    plt.xlabel("Model", fontsize=30)
    plt.ylabel(y_label, fontsize=30)
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

    # data_for_plotting_df.to_csv(str(figure_directory / (save_title + '_test.csv')), index=False)  
    data_for_plotting_df.to_pickle(str(figure_directory / (save_title + '_test.pkl')))  
    # train_data_for_plotting_df.to_csv(str(figure_directory / (save_title + '_train.csv')), index=False)  
    train_data_for_plotting_df.to_pickle(str(figure_directory / (save_title + '_train.pkl')))  

def plot_model_comparison_l2(cv,
                          cv_train,
                          global_fit,
                          K_vals,
                          figure_directory,
                          y_label="Normalized LL",
                          save_title='best_params_performance',
                          cols = ["#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc","#96f97b"]):

    data_for_plotting_df, loc_best, best_val = \
        create_cv_frame_for_plotting_l2(cv)
    train_data_for_plotting_df, train_loc_best, train_best_val, \
          = create_cv_frame_for_plotting_l2(cv_train)
    
    fig = plt.figure(figsize=(10, 10),
                        dpi=80,
                        facecolor='w',
                        edgecolor='k')

    g = sns.lineplot(x=data_for_plotting_df['model'],
        y=data_for_plotting_df['cv_bit_trial'],
        err_style="bars",
        errorbar=('se', 1.96),
        mew=0,
        color=cols[0],
        marker='o',
        ci=68,
        label="test",
        alpha=1,
        lw=4)
    sns.lineplot(
        x=train_data_for_plotting_df['model'],
        y=train_data_for_plotting_df['cv_bit_trial'],
        err_style="bars",
        errorbar=('se', 1.96),
        mew=0,
        color=cols[1],
        marker='o',
        ci=68,
        label="train",
        alpha=1,
        lw=4)
    plt.xlabel("Model", fontsize=30)
    plt.ylabel(y_label, fontsize=30)
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


def plot_FArate(df: pd.DataFrame, ax, label, K=None):
    """
    Plot false alarm rate (early lick/press rate)
    """

    if K is None:
        earlylicks = (df[(df['fitted_trials'] == 1)]
                    .groupby(['session'])
                    .agg({'early_report': 'mean'})
                    .unstack()
                    )
    else:
        earlylicks = (df[(df['fitted_trials'] == 1) & (df['state'] == K)]
                    .groupby(['session'])
                    .agg({'early_report': 'mean'})
                    .unstack()
                    )
        
    earlylicks_mean = earlylicks.mean(axis=0)
    earlylicks_prc = earlylicks.quantile([0.025, 0.975]) # this gives actual values

    earlylicks_err = np.array([[earlylicks_mean - earlylicks_prc.values[0]], 
                               [earlylicks_prc.values[1] - earlylicks_mean]])

    ax.errorbar(0, earlylicks_mean, yerr=earlylicks_err, 
                color='k', marker='o',
                markersize='2', capsize=2) # this asks for the diff


def plot_state_Wk(weight_vectors,
                  ax):

    assert weight_vectors.ndim == 2, 'weight_vectors has wrong dim!'
    C = weight_vectors.shape[0]
    M = weight_vectors.shape[1] - 1

    choice_label_mapping = {0: 'miss', 1: 'Hit', 2: 'FA', 3: 'abort'}
    for j in range(C): # each category 
        l = ax.plot(range(M + 1), 
                      weight_vectors[j,:], # plot weights with orginal signs
                      marker='o',
                      markersize='2',
                      label=choice_label_mapping[j],
                      lw=2)
    ax.axhline(y=0, color="k", alpha=0.5, ls="--")

def plot_state_Wk_diff(weight_vectors,
                  ax):

    assert weight_vectors.ndim == 2, 'weight_vectors has wrong dim!'
    C = weight_vectors.shape[0]
    M = weight_vectors.shape[1] - 1

    choice_label_mapping = {0: 'miss', 1: 'Hit', 2: 'FA', 3: 'abort'}
    l = ax.plot(range(M + 1), 
                    weight_vectors[1,:] - weight_vectors[0,:], # plot weights with orginal signs
                    marker='o',
                    markersize='2',
                    label="hit - miss",
                    lw=2)
    l = ax.plot(range(M + 1), 
                    weight_vectors[2,:] - weight_vectors[1,:], # plot weights with orginal signs
                    marker='o',
                    markersize='2',
                    label="FA - hit",
                    lw=2)
    l = ax.plot(range(M + 1), 
                    weight_vectors[2,:] - weight_vectors[0,:], # plot weights with orginal signs
                    marker='o',
                    markersize='2',
                    label="FA - miss",
                    lw=2)
    ax.axhline(y=0, color="k", alpha=0.5, ls="--")


def plot_state_dwelltime(dwell_time_df: pd.DataFrame,
                         ax,
                         bins=onp.arange(0, 90, 5)):
        ax.hist(dwell_time_df.duration.values,
                 bins=bins,
                 histtype='bar',
                 rwidth=0.8)
    
def calculate_predictive_accuracy(this_inputs, this_datas, this_masks, this_hmm, 
                                   y, idx_to_include):

    # Get expected states:
    expectations = [
    this_hmm.expected_states(data=data,
                            input=input,
                            mask=mask)[0]
    for data, input, mask in zip(this_datas, this_inputs, this_masks)
    ]
    # Convert this now to one array:
    posterior_probs = onp.concatenate(expectations, axis=0)

    prob_choices = [
    onp.exp(this_hmm.observations.calculate_logits(input=input))
    for data, input, mask in zip(this_datas, this_inputs, this_masks)
    ]
    prob_choices = onp.concatenate(prob_choices, axis=0)

    final_prob_0 = onp.sum(onp.multiply(posterior_probs, prob_choices[:, :, 0]), axis=1)
    final_prob_1 = onp.sum(onp.multiply(posterior_probs, prob_choices[:, :, 1]), axis=1)
    final_prob_2 = onp.sum(onp.multiply(posterior_probs, prob_choices[:, :, 2]), axis=1)

    predicted_label_0 = onp.around(final_prob_0, decimals=0).astype('int')
    predicted_label_1 = onp.around(final_prob_1, decimals=0).astype('int')
    predicted_label_2 = onp.around(final_prob_2, decimals=0).astype('int')

    predicted_label = onp.add(predicted_label_1, predicted_label_2*2)
    # Examine at appropriate idx
    predictive_acc = onp.sum(
    y[idx_to_include,
    0] == predicted_label[idx_to_include]) / len(idx_to_include)
    return predictive_acc