import numpy as np
import numpy.random as npr
import json
import os
from pathlib import Path
from design_mat_utils import create_design_mat_y, create_design_mat_rt

def scan_sessions(path):
    matches = []
    for dirpath, _, filenames in os.walk(str(path)):
        for filename in [f for f in filenames if f.endswith('outcome.npy')]:
            matches.append(str(Path(dirpath)))
    return matches

def get_animal_name(eid):
    # Get session id:
    raw_session_id = eid.split('Subjects/')[1]
    # Get animal:
    animal = raw_session_id.split('/')[0]
    return animal

def load_animal_list(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list

def load_animal_eid_dict(file):
    with open(file, 'r') as f:
        animal_eid_dict = json.load(f)
    return animal_eid_dict

def get_all_unnormalized_data_this_session(eid, path_to_dataset):
    # Load raw data
    animal, session_id, changesize, hazardblock, outcome, reactiontimes, stimT \
        = get_raw_data(eid, path_to_dataset)

    # Use sessions that have both early and late blocks
    trials_to_study = np.where((hazardblock == 0) | (hazardblock == 1))[0]

    # Create design mat for both y and rt. Continuous variables are unnormalized. Edit design_mat_utils.py
    # to change the entries. No bias intercept included here.
    unnormalized_inpt_y, outcome_noref, continuous_column_y = \
        create_design_mat_y(changesize[trials_to_study],
                            outcome[trials_to_study],
                            reactiontimes[trials_to_study],
                            stimT[trials_to_study],
                            hazardblock[trials_to_study])

    unnormalized_inpt_rt, rt_1d, stim_onset_1d, continuous_column_rt = \
        create_design_mat_rt(changesize[trials_to_study],
                             outcome[trials_to_study],
                             reactiontimes[trials_to_study],
                             stimT[trials_to_study],
                             hazardblock[trials_to_study])

    session = [session_id for i in range(changesize[trials_to_study].shape[0])]
    y = np.expand_dims(outcome_noref, axis=1)
    rt = np.expand_dims(rt_1d, axis=1)
    stim_onset = np.expand_dims(stim_onset_1d, axis=1)

    unnormalized_inpts = [unnormalized_inpt_y, unnormalized_inpt_rt]
    needs_to_be_normalized = [continuous_column_y, continuous_column_rt]

    return animal, session, y, rt, stim_onset, unnormalized_inpts, needs_to_be_normalized

def get_raw_data(eid, path_to_dataset):
    print(eid)
    # Get session id:
    raw_session_id = eid.split('Subjects/')[1]
    # Get animal:
    animal = raw_session_id.split('/')[0]
    # Replace '/' with dash in session ID
    session_id = raw_session_id.replace('/', '-')

    # Get trial data
    changesize = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.changesize.npy')[0]
    hazardblock = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.hazardblock.npy')[0]
    outcome = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.outcome.npy')[0]
    reactiontimes = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.reactiontimes.npy')[0]
    stimT = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.stimT.npy')[0]

    return animal, session_id, changesize, hazardblock, outcome, reactiontimes, stimT

def create_train_test_sessions(session, num_folds=5):
    """
    Create a session-fold lookup table to split training dataset and validation dataset
    from the entire dataset, as well as training dataset and tuning dataset 
    from the training dataset. They use the same number of folds.
    """
     
    num_sessions = len(np.unique(session))
    # Map sessions to folds:
    unshuffled_folds_1 = np.repeat(np.arange(num_folds),
                                   int(np.ceil(num_sessions / num_folds)))
    unshuffled_folds_2 = np.tile(np.arange(num_folds),
                                 int(np.ceil(num_sessions / num_folds))) # [0,0,0,...]
    unshuffled_folds = np.vstack([unshuffled_folds_1[:num_sessions], 
                                  unshuffled_folds_2[:num_sessions]]) # [0,1,2,...]
    shuffled_folds = npr.permutation(unshuffled_folds.T).T # np.permutation only supports colum-wise permutation
    assert len(np.unique(
        shuffled_folds)) == num_folds, "require at least one session per fold for " \
                               "each animal!"
    # Look up table of shuffle-folds:
    sess_id = np.array(np.unique(session), dtype='str')
    shuffled_folds = np.array(shuffled_folds, dtype='O')
    session_fold_lookup_table = np.transpose(
        np.vstack([sess_id, shuffled_folds]))
    return session_fold_lookup_table