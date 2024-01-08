import numpy as np
import numpy.random as npr
from scipy.stats import multinomial
import json
import os
from pathlib import Path
import scipy.io as io
import pandas as pd

def scan_sessions(path):
    matches = []
    for dirpath, _, filenames in os.walk(str(path)):
        for filename in [f for f in filenames if f.endswith('outcome.npy')]:
            matches.append(str(Path(dirpath)))
    return matches

def get_animal_name(eid):
    # get session id:
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

    # Create design mat = matrix of size T x 6, with entries for
    # change size/change timing/past choice/past change timing/past choice timing/bias block
    unnormalized_inpt, outcome_noref = create_design_mat(changesize,
                                                         hazardblock,
                                                         outcome,
                                                         reactiontimes,
                                                         stimT)
    y = np.expand_dims(outcome_noref, axis=1)
    session = [session_id for i in range(changesize.shape[0])]
    # You can add some criteria here and change codes to 
    # something like changesize[trials_to_study]

    return animal, unnormalized_inpt, y, session, reactiontimes, stimT

def get_raw_data(eid, path_to_dataset):
    print(eid)
    # get session id:
    raw_session_id = eid.split('Subjects/')[1]
    # Get animal:
    animal = raw_session_id.split('/')[0]
    # replace '/' with dash in session ID
    session_id = raw_session_id.replace('/', '-')

    # Get trial data
    changesize = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.changesize.npy')[0]
    hazardblock = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.hazardblock.npy')[0]
    outcome = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.outcome.npy')[0]
    reactiontimes = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.reactiontimes.npy')[0]
    stimT = np.load(path_to_dataset / Path(eid) / '_dmdm_trials.stimT.npy')[0]

    return animal, session_id, changesize, hazardblock, outcome, reactiontimes, stimT

def create_design_mat(stim, hazardblock, outcome, reactiontimes, stimT):
    ''' 
    Create unnormalized input: with first column is changesize,
    second column as change onset. third column as previous choice, 
    fourth column as previous change onset, fifth column as previous reaction time, 
    sixth column as previous hazardblock
    ''' 

    # changesize:
    design_mat = np.zeros((len(stim), 6))
    design_mat[:, 0] = stim

    # change onset:
    design_mat[:, 1] = stimT

    # previous choice vector:
    # hit = 1, FA = 2, miss = 0, abort = 3
    # Ref trials are treated as FA here.
    outcome_noref = ref2FA(outcome)
    # Hits in no change trials (where stim = 0) should also be regarded as FA, as from the mouse
    # perspective they are licking prior to changes.
    previous_choice, locs_FA_abort = create_previous_choice_vector(outcome_noref, stim)
    design_mat[:, 2] = previous_choice

    # previous change onset and reactiontime:
    previous_stimT, previous_rt = create_previous_RT_vector(stimT, reactiontimes, locs_FA_abort)
    design_mat[:, 3] = previous_stimT
    design_mat[:, 4] = previous_rt

    # previous hazard block
    design_mat[:, 5] = np.hstack([np.array(hazardblock[0]), hazardblock])[:-1]

    return design_mat, outcome_noref

def ref2FA(choice):
    new_choice = choice.copy()
    new_choice[new_choice == 4] = 2 # ref = 4, FA = 2
    return new_choice

def create_previous_choice_vector(choice, stim):
    ''' choice: choice vector of size T
        previous_choice : vector of size T with previous choice made by animal
                          output is in {0, 1, 2, 3}, where hit with changes = 1, FA = 2, 
                          miss = 0, abort = 3. We also make no change hit = 2 here.
        locs_FA_abort: locations of FA or abort happened.
    '''
    locs_nochangehit = np.intersect1d(np.where(stim == 0)[0], np.where(choice == 1)[0])
    locs_FA_abort = np.where((choice == 2) | (choice == 3))[0]

    choice_updated = choice.copy()
    for i, loc in enumerate(locs_nochangehit):
        choice_updated[loc] = 2
    
    previous_choice = np.hstack([np.array(choice_updated[0]), choice_updated])[:-1]
    return previous_choice, locs_FA_abort

def create_previous_RT_vector(stimT, reactiontimes, locs_FA_abort):
    ''' stimT: change onset vector of size T
        reactiontimes: reaction times from baseline onset
        locs_FA_abort: locations of FA or abort happened.
        previous_stimT : vector of size T with previous change onset observed by animal.
                         0 for FA and abort trials.
        previous_rt : vector of size T with previous reactiontimes. 0 for miss.
    '''
    stimT_updated = stimT.copy()
    for i, loc in enumerate(locs_FA_abort):
        stimT_updated[loc] = 0
    previous_stimT = np.hstack([np.array(stimT_updated[0]), stimT_updated])[:-1]

    rt_nonan = np.nan_to_num(reactiontimes, nan=0)
    previous_rt = np.hstack([np.array(rt_nonan[0]), rt_nonan])[:-1]
    return previous_stimT, previous_rt

def create_train_test_sessions(session, num_folds=5):
    # create a session-fold lookup table
    num_sessions = len(np.unique(session))
    # Map sessions to folds:
    unshuffled_folds = np.repeat(np.arange(num_folds),
                                 np.ceil(num_sessions / num_folds))
    shuffled_folds = npr.permutation(unshuffled_folds)[:num_sessions]
    assert len(np.unique(
        shuffled_folds)) == 5, "require at least one session per fold for " \
                               "each animal!"
    # Look up table of shuffle-folds:
    sess_id = np.array(np.unique(session), dtype='str')
    shuffled_folds = np.array(shuffled_folds, dtype='O')
    session_fold_lookup_table = np.transpose(
        np.vstack([sess_id, shuffled_folds]))
    return session_fold_lookup_table

def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    y = y.astype('int')
    session = data[2]
    return inpt, y, session