#  Functions to assist with post-processing of GLM-HMM fits
import numpy as np

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

def create_abort_mask(abort_idx, T: int):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1
    = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in abort_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = np.expand_dims(mask + 0, axis=1)
    assert len(nonviolation_idx) + len(
        abort_idx) == T, "violation and non-violation idx do not include " \
                             "" \
                             "" \
                             "" \
                             "" \
                             "all dta!"
    return nonviolation_idx, mask

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