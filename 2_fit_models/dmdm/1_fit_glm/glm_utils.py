import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from GLM import glm
import os
from pathlib import Path

npr.seed(65)

def get_file_dir():
    return Path(os.path.dirname(os.path.realpath(__file__)))

def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    rt = data[3]
    stimT = data[4]
    return inpt, y, session, rt, stimT

def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table


def fit_glm_runml(inputs, datas, M, C, outcome_dict):
    new_glm = glm(M, C, outcome_dict, dist='RUNML')
    new_glm.fit_glm(datas, inputs, masks=None, tags=None)
    # Get loglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, None, None)
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights

def fit_glm(inputs, datas, M, C, outcome_dict):
    new_glm = glm(M, C, outcome_dict, dist='Categorical')
    # new_glm = glm(M, C, outcome_dict, tau=[1,1], dist='RUNML')
    new_glm.fit_glm(datas, inputs, masks=None, tags=None)
    # Get loglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, None, None)
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights


def load_animal_list(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list

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

    # hit = 1, FA = 2, miss = 0, abort = 3
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
             "Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)

    fig.savefig(figure_directory / ('glm_weights_' + save_title + '.png'))
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
             "Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)

    fig.savefig(figure_directory / ('glm_hit_vs_miss_' + save_title + '.png'))
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
             "Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)

    fig.savefig(figure_directory / ('glm_FA_vs_abort_' + save_title + '.png'))
    plt.axis('off')
    plt.close(fig)