import numpy as np

def partition_data_by_session(inpt, y, mask, session):
    '''
    Partition inpt, y, mask by session
    :param inpt np.array: arr of size TxM
    :param y np.array: arr of size T x D
    :param mask np.array: Boolean arr of size T x 1 indicating if element is violation or not
    :param session list: list of size T containing session ids

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
        masks.append(mask[idx, :])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks

def create_abort_mask(abort_idx, T: int):
    """
    Return indices of nonviolations (non-abort) and also a Boolean mask for inclusion (1
    = nonviolation; 0 = violation)
    :param abort_idx np.array: indices of abort trials
    :param T int: length of the data, i.e., the total number of trials
    """
    mask = np.array([i not in abort_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = np.expand_dims(mask + 0, axis=1)
    assert len(nonviolation_idx) + len(abort_idx) == T, \
        "violation and non-violation idx do not include all data!"
    return nonviolation_idx, mask

def partition_data_by_session_L2(inpt, y, mask, session, penalization_factor):
    '''
    Partition inpt, y, mask by session
    :param inpt np.array: arr of size TxM
    :param y np.array: arr of size T x D
    :param mask np.array: Boolean arr of size T x 1 indicating if element is violation or not
    :param session list: list of size T containing session ids

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

        raw_inpt = inpt[idx, :]
        M = raw_inpt.shape[1]
        penal_array = np.sqrt(penalization_factor) * np.identity(M)
        L2_input = np.concatenate((raw_inpt,penal_array),axis=0).astype(np.int)
        inputs.append(L2_input)

        # L2_data = np.concatenate((y[idx, :], np.zeros((M,1)))).astype(np.int)
        # datas.append(L2_data)
        L2_data = np.concatenate((y[idx, :], np.full((M, 1), -1))).astype(np.int)
        datas.append(L2_data)
        # datas.append(y[idx, :])

        L2_mask = np.concatenate((mask[idx, :], np.ones((M,1)))).astype(np.int)
        masks.append(L2_mask)
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks