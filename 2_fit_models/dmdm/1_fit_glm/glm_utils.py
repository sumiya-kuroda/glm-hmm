import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from GLM import glm
from pathlib import Path

def fit_glm_runml(inputs, datas, M, C, masks, outcome_dict):
    # Initialize GLM and fit:
    new_glm = glm(M, C, outcome_dict, obs='RUNML')
    weights_progress = new_glm.fit_glm(datas, inputs, masks=masks, tags=None, tol=1e-2)
    lls_progress = new_glm.recover_lls(datas, inputs, masks, [None], weights_progress)

    # Get final lloglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, None, None)
    # Get final weights of training data:
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights, lls_progress

def fit_glm(inputs, datas, M, C, masks=None, outcome_dict=None,
            regularization=None,l2_penalty=0):
    # Initialize GLM and fit:
    new_glm = glm(M, C, outcome_dict, obs='Categorical',regularization=regularization, l2_penalty=l2_penalty)
    # new_glm = glm(M, C, outcome_dict, tau=[1,1], dist='RUNML')
    weights_progress = new_glm.fit_glm(datas, inputs, masks=masks, tags=None, tol=1e-2) 
    lls_progress = new_glm.recover_lls(datas, inputs, masks, [None], weights_progress)

    # Get final loglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, masks, None)
    # Get final weights of training data:
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights, lls_progress

# https://www.reddit.com/r/learnmath/comments/kw7bc6/can_someone_explain_what_a_diagonal_gaussian_is/
def fit_RT_glm(inputs, datas, stim_onset, M, masks=None):
    # Initialize GLM and fit:
    new_glm = glm(M, 0, None, obs='DiagonalGaussian')
    weights_progress = new_glm.fit_glm(datas, inputs, masks=masks, tags=stim_onset, 
                                       optimizer="rmsprop")
    lls_progress = weights_progress

    # Get final lloglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, masks, stim_onset)
    # Get final weights of training data:
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights, lls_progress

# https://stackoverflow.com/questions/44465242/getting-the-legend-label-for-a-line-in-matplotlib
def get_label_for_line(line):
    leg = line.axes.get_legend()
    ind = line.axes.get_lines().index(line)
    return leg.texts[ind].get_text()

def plot_input_vectors(Ws,
                       figure_directory,
                       title='true',
                       save_title="true",
                       labels_for_plot=[]):
    K = Ws.shape[0]
    K_prime = Ws.shape[1] # C
    M = Ws.shape[2] - 1 # exclude bias just for clarification purpose

    choice_label_mapping = {0: 'miss', 1: 'Hit', 2: 'FA', 3: 'abort'}

    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)

    for j in range(K):
        lines = []
        for k in range(K_prime): # each category 
            l = plt.plot(range(M + 1), 
                         Ws[j][k], # plot weights with orginal signs
                         marker='o',
                         label=choice_label_mapping[k])
            plt.ylim((-70, 70))
            plt.xlim(-1, M+1)
            lines.append(l)
        
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.xticks(list(range(0, len(labels_for_plot))),
                    labels_for_plot,
                    rotation='90',
                    fontsize=12)
        plt.legend()
    fig.text(0.04,
             0.5,
             "y Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)

    fig.savefig(figure_directory / ('y_glm_weights_' + save_title + '.png'))
    plt.axis('off')
    plt.close(fig)


def plot_logOR_hit_vs_miss(Ws,
                           figure_directory,
                           title='true',
                           save_title="true",
                           labels_for_plot=[]):
    K = Ws.shape[0]
    K_prime = Ws.shape[1] # C
    M = Ws.shape[2] - 1 # exclude bias just for clarification purpose

    # hit = 1, FA = 2, miss = 0, abort = 3
    choice_label_mapping = {0: 'miss', 1: 'Hit', 2: 'FA', 3: 'abort'}

    outcome_list = [0, 1]

    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)

    for j in range(K):
        lines = []
        for k in outcome_list: # each category 
            l = plt.plot(range(M + 1), 
                         Ws[j][1] - Ws[j][0], # plot weights with orginal signs
                         marker='o',
                         label=choice_label_mapping[k])
            plt.ylim((-4, 5))
            plt.xlim(-1, M+1)
        
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.xticks(list(range(0, len(labels_for_plot))),
                    labels_for_plot,
                    rotation='90',
                    fontsize=12)
    fig.text(0.04,
             0.5,
             "y Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)

    fig.savefig(figure_directory / ('glm_hit_vs_miss_' + save_title + '.png'))
    plt.axis('off')
    plt.close(fig)

def plot_logOR_hit_vs_FA(Ws,
                           figure_directory,
                           title='true',
                           save_title="true",
                           labels_for_plot=[]):
    K = Ws.shape[0]
    K_prime = Ws.shape[1] # C
    M = Ws.shape[2] - 1 # exclude bias just for clarification purpose

    # hit = 1, FA = 2, miss = 0, abort = 3
    choice_label_mapping = {0: 'miss', 1: 'Hit', 2: 'FA', 3: 'abort'}

    outcome_list = [0, 2]

    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)

    for j in range(K):
        lines = []
        for k in outcome_list: # each category 
            l = plt.plot(range(M + 1), 
                         Ws[j][1] - Ws[j][2], # plot weights with orginal signs
                         marker='o',
                         label=choice_label_mapping[k])
            plt.ylim((-4, 5))
            plt.xlim(-1, M+1)
        
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.xticks(list(range(0, len(labels_for_plot))),
                    labels_for_plot,
                    rotation='90',
                    fontsize=12)
    fig.text(0.04,
             0.5,
             "y Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)

    fig.savefig(figure_directory / ('glm_hit_vs_FA_' + save_title + '.png'))
    plt.axis('off')
    plt.close(fig)

def plot_logOR_FA_vs_abort(Ws,
                           figure_directory,
                           title='true',
                           save_title="true",
                           labels_for_plot=[]):
    K = Ws.shape[0]
    K_prime = Ws.shape[1] # C
    M = Ws.shape[2] - 1 # exclude bias just for clarification purpose

    # hit = 1, FA = 2, miss = 0, abort = 3
    choice_label_mapping = {0: 'miss', 1: 'Hit', 2: 'FA', 3: 'abort'}

    outcome_list = [2, 3]

    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)

    for j in range(K):
        lines = []
        for k in outcome_list: # each category 
            l = plt.plot(range(M + 1), 
                         Ws[j][2] - Ws[j][3], # plot weights with orginal signs
                         marker='o',
                         label=choice_label_mapping[k])
            plt.ylim((-4, 5))
            plt.xlim(-1, M+1)
        
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.xticks(list(range(0, len(labels_for_plot))),
                    labels_for_plot,
                    rotation='90',
                    fontsize=12)
    fig.text(0.04,
             0.5,
             "y Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)

    fig.savefig(figure_directory / ('glm_FA_vs_abort_' + save_title + '.png'))
    plt.axis('off')
    plt.close(fig)


def plot_rt_weights(Ws,
                       figure_directory,
                       title='true',
                       save_title="true",
                       labels_for_plot=[]):
    K = Ws.shape[0]
    M = Ws.shape[1] - 1 # exclude bias just for clarification purpose


    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)

    for j in range(K):
        l = plt.plot(range(M + 1), 
                        Ws[j], # plot weights with orginal signs
                        marker='o')
        plt.ylim((-10, 10))
        plt.xlim(-1, M+1)
        
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.xticks(list(range(0, len(labels_for_plot))),
                    labels_for_plot,
                    rotation='90',
                    fontsize=12)
    fig.text(0.04,
             0.5,
             "RT Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("RT GLM Weights: " + title, y=0.99, fontsize=14)

    fig.savefig(figure_directory / ('RT_glm_weights_' + save_title + '.png'))
    plt.axis('off')
    plt.close(fig)


def plot_lls(fit_ll,
             figure_directory,
             save_title="true"):

    fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(fit_ll, label="EM")
    plt.legend(loc="lower right")
    plt.xlabel("Iteration")
    plt.xlim(0, len(fit_ll))
    plt.ylabel("Log Probability")

    fig.savefig(figure_directory / ('glm_lls_' + save_title + '.png'))
    plt.axis('off')
    plt.close(fig)

