# Functions to assist with post-processing of GLM-HMM fits
import numpy as np
import ssm
import matplotlib.pyplot as plt

def partition_data_by_session(inpt, y, mask, session):
    '''
    Partition inpt, y, mask by session
    :param inpt: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or
    not
    :param session: list of size T containing session ids
    :return: list of inpt arrays, data arrays and mask arrays, where the
    number of elements in list = number of sessions and each array size is
    number of trials in session
    '''
    inputs = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [
        session[index] for index in sorted(indexes)
    ]  # ensure that unique sessions are ordered as they are in
    # session (so we can map inputs back to inpt)
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(inpt[idx, :])
        datas.append(y[idx, :])
        masks.append(mask[idx,:])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks


def permute_transition_matrix(transition_matrix, permutation):
    transition_matrix = transition_matrix[np.ix_(permutation, permutation)]
    return transition_matrix


def calculate_state_permutation(hmm_params, K):
    '''
    If K = 3, calculate the permutation that results in states being ordered
    as engaged/bias left/bias right
    Else: order states so that they are ordered by engagement
    :param hmm_params:
    :return: permutation
    '''
    if K == 1:
        permutation = None
        weight_vectors = hmm_params
        log_transition_matrix = np.zeros((1, 1))
        init_state_dist = np.zeros((1))

    elif K > 1:
        glm_weights = hmm_params[2]
        K = glm_weights.shape[0]

        permutation = np.argsort(-glm_weights[:, 0, 0])
        assert len(permutation) == K, "permutation is incorrect size"
        assert check_all_indices_present(permutation, K), "not all indices " \
                                                        "present in " \
                                                        "permutation: " \
                                                        "permutation = " + \
                                                        str(permutation)
        weight_vectors = hmm_params[2][permutation]
        log_transition_matrix = permute_transition_matrix(
            hmm_params[1][0], permutation)
        init_state_dist = hmm_params[0][0][permutation]

    # if K == 3:
    #     # want states ordered as engaged/bias left/bias right
    #     M = glm_weights.shape[2] - 1
    #     # bias coefficient is last entry in dimension 2
    #     engaged_loc = \
    #         np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
    #     reduced_weights = np.copy(glm_weights)
    #     # set row in reduced weights corresponding to engaged to have a bias
    #     # that will not cause it to have largest bias
    #     reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
    #     bias_left_loc = \
    #         np.where(
    #             (reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[
    #             0][0]
    #     state_order = [engaged_loc, bias_left_loc]
    #     bias_right_loc = np.arange(3)[np.where(
    #         [range(3)[i] not in state_order for i in range(3)])][0]
    #     permutation = np.array([engaged_loc, bias_left_loc, bias_right_loc])
    # elif K == 4:
    #     # want states ordered as engaged/bias left/bias right
    #     M = glm_weights.shape[2] - 1
    #     # bias coefficient is last entry in dimension 2
    #     engaged_loc = \
    #         np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
    #     reduced_weights = np.copy(glm_weights)
    #     # set row in reduced weights corresponding to engaged to have a bias
    #     # that will not
    #     reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
    #     bias_right_loc = \
    #         np.where(
    #             (reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[
    #             0][0]
    #     bias_left_loc = \
    #         np.where(
    #             (reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[
    #             0][0]
    #     state_order = [engaged_loc, bias_left_loc, bias_right_loc]
    #     other_loc = np.arange(4)[np.where(
    #         [range(4)[i] not in state_order for i in range(4)])][0]
    #     permutation = np.array(
    #         [engaged_loc, bias_left_loc, bias_right_loc, other_loc])
    # else:
    #     # order states by engagement: with the most engaged being first.
    #     # Note: argsort sorts inputs from smallest to largest (hence why we
    #     # convert to -ve glm_weights)
    #     permutation = np.argsort(-glm_weights[:, 0, 0])
    # assert that all indices are present in permutation exactly once:

    return init_state_dist, log_transition_matrix, weight_vectors, permutation


def check_all_indices_present(permutation, K):
    for i in range(K):
        if i not in permutation:
            return False
    return True

def plot_best_MAP_params(averaged_normalized_lls:np.array, 
                         transition_alpha: list,
                         prior_sigma: list,
                         figure_directory,
                         save_title='map_parameters'):
    
    fig = plt.figure(figsize=(len(transition_alpha)*10, len(prior_sigma)*10),
                        dpi=80,
                        facecolor='w',
                        edgecolor='k')

    plt.imshow(averaged_normalized_lls, cmap='hot')
    for i in range(averaged_normalized_lls.shape[0]):
        for j in range(averaged_normalized_lls.shape[1]):
            text = plt.text(j,
                            i,
                            np.around(averaged_normalized_lls[i, j],
                                        decimals=3),
                            ha="center",
                            va="center",
                            color="k",
                            fontsize=30)
    plt.ylabel("sigma", fontsize=30)
    plt.xlabel("alpha", fontsize=30)
    plt.colorbar()
    plt.xlim(-0.5, len(transition_alpha) - 0.5)
    plt.ylim(-0.5, len(prior_sigma) - 0.5)
    plt.xticks(range(0, len(transition_alpha)), transition_alpha, fontsize=30)
    plt.yticks(range(0, len(prior_sigma)), prior_sigma, fontsize=30)
    plt.title("MAP hyperparameters", fontsize=40)

    fig.savefig(figure_directory / (save_title + '.png'))
    plt.axis('off')
    plt.close(fig)

def get_marginal_posterior(inputs, datas, masks, C,
                           hmm_params, K, permutation):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    this_hmm = ssm.HMM(K, D, M,
                       observations="input_driven_obs_multinominal",
                       observation_kwargs=dict(C=C),
                       transitions="standard")
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [this_hmm.expected_states(data=data, input=input,
                                             mask=mask)[0]
                    for data, input, mask
                    in zip(datas, inputs, masks)]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, permutation]

    return posterior_probs
