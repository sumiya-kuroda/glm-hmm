# Functions to assist with post-processing of GLM-HMM fits
import sys

import numpy as np
import ssm

from data_io import load_glmhmm_data, load_cv_arr, load_glm_vectors, load_lapse_params, get_file_dir

sys.path.append(str(get_file_dir() / '1_fit_glm'))
from GLM import glm


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
        masks.append(mask[idx])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks


def permute_transition_matrix(transition_matrix, permutation):
    transition_matrix = transition_matrix[np.ix_(permutation, permutation)]
    return transition_matrix


def calculate_state_permutation(hmm_params):
    '''
    If K = 3, calculate the permutation that results in states being ordered
    as engaged/bias left/bias right
    Else: order states so that they are ordered by engagement
    :param hmm_params:
    :return: permutation
    '''
    glm_weights = hmm_params[2]
    K = glm_weights.shape[0]

    permutation = np.argsort(-glm_weights[:, 0, 0])

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
    assert len(permutation) == K, "permutation is incorrect size"
    assert check_all_indices_present(permutation, K), "not all indices " \
                                                      "present in " \
                                                      "permutation: " \
                                                      "permutation = " + \
                                                      str(permutation)
    return permutation


def check_all_indices_present(permutation, K):
    for i in range(K):
        if i not in permutation:
            return False
    return True


def get_marginal_posterior(inputs, datas, masks, hmm_params, K, permutation):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    this_hmm = ssm.HMM(K, D, M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2),
                       transitions="standard")
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [this_hmm.expected_states(data=data, input=input,
                                             mask=np.expand_dims(mask,
                                                                 axis=1))[0]
                    for data, input, mask
                    in zip(datas, inputs, masks)]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, permutation]
    return posterior_probs
