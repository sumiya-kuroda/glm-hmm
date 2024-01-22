import numpy as np
import numpy.random as npr
from scipy.stats import multinomial
import json
import os
from pathlib import Path
from sklearn import preprocessing
from ssm.util import one_hot

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

    # Subset early blocks only:
    trials_to_study = np.where((hazardblock == 0) | (hazardblock == 1))[0]

    # Create design mat = matrix of size T x 6, with entries for
    # change size/change timing/past choice/past change timing/past choice timing/bias block
    unnormalized_inpt, outcome_noref = create_design_mat(changesize[trials_to_study],
                                                         outcome[trials_to_study],
                                                         reactiontimes[trials_to_study],
                                                         stimT[trials_to_study],
                                                         hazardblock[trials_to_study])
    
    session = [session_id for i in range(changesize[trials_to_study].shape[0])]
    y = np.expand_dims(outcome_noref, axis=1)

    return animal, unnormalized_inpt, y, session, reactiontimes[trials_to_study], stimT[trials_to_study]

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

def create_design_mat(stim, outcome, reactiontimes, stimT, hazard):
    ''' 
    Create unnormalized input: with first column is changesize,
    second column as change onset. third column as previous choice, 
    fourth column as previous change onset.
    ''' 

    # Remap variables:
    # We first treat ref trials as FA here to allow only four outcomes (Hit/FA/Miss/Abort).
    # Hits in no change trials (where stim = 0) should also be regarded as FA, as from the mouse
    # perspective they are licking prior to changes. 
    # We then make change size and onset for FA and abort trials to be zero, 
    # because from mouse perspectives, they did not realize that there was a change.
    outcome_noref = ref2FA(outcome)
    choice_updated, stim_updated, stimT_updated = remap_vals(outcome_noref, 
                                                             stim, 
                                                             stimT, 
                                                             reactiontimes)
    
    # Change size:
    design_mat = np.zeros((len(stim_updated), 6))
    design_mat[:, 0] = stim_updated # unnormalized

    # Change onset:
    design_mat[:, 1] = stimT_updated # unnormalized
    # design_mat[:, 1] = hazard

    # previous Change onset
    # previous_stimT = np.hstack([np.array(stimT_updated[0]), stimT_updated])[:-1]
    # design_mat[:, 1] = preprocessing.scale(previous_stimT)

    # previous choice vector:
    previous_choice, rewarded = create_previous_choice_vector(choice_updated)
    design_mat[:, 2:6] = previous_choice
    # design_mat[:, 4] = rewarded

    # previous change onset and reactiontime:
    # previous_stimT, previous_rt = create_previous_RT_vector(stimT, reactiontimes, locs_FA_abort)
    # design_mat[:, 3] = previous_stimT
    # design_mat[:, 4] = previous_rt

    return design_mat, outcome_noref

def ref2FA(choice):
    new_choice = choice.copy()
    new_choice[new_choice == 4] = 2 # ref = 4, FA = 2
    return new_choice

def remap_vals(choice, stim, stimT, rt, delay=0.5):
    ''' choice: choice vector of size T. 
                By default, hit = 1, FA = 2, miss = 0, abort = 3
        stimT: change size vector of size T
        stimT: change onset vector of size T
    '''

    # Treat hits in no change trials (where stim = 0) as FA
    locs_nochangehit = np.intersect1d(np.where(stim == 0)[0], np.where(choice == 1)[0])
    choice_updated = choice.copy()
    for i, loc in enumerate(locs_nochangehit):
        choice_updated[loc] = 2 # FA = 2

    # Make change sizes of FA and abort trials to be zero (no change FA trials are already zero)
    # Also, make a change onset of FA and abort trials to be reactiontimes - delay
    locs_FA_abort = np.where((choice_updated == 2) | (choice_updated == 3))[0]
    stim_updated = stim.copy()
    stimT_updated  = stimT.copy()
    for i, loc in enumerate(locs_FA_abort):
        stim_updated[loc] = 0 
        stimT_updated[loc] = rt[loc] - delay

    # Make a change onset of miss trials to be a full length of that trial
    locs_miss = np.where(choice_updated == 0)[0]
    for i, loc in enumerate(locs_miss):
        stimT_updated[loc] = stimT_updated[loc] + 2.15

    return choice_updated, stim_updated, stimT_updated

def create_previous_choice_vector(choice):
    # The original choice vectors are
    # hit with changes = 1, FA = 2, miss = 0, abort = 3.
    # We will one-hot encode this vector as well as create a new vector about
    # whther mice got rewarded or punished
    previous_choice = np.hstack([np.array(choice[0]), choice])[:-1]
    # C = len(np.unique(choice)) but some sessions do not have aborts
    one_hot_prev_choice = one_hot(previous_choice, 4)

    choice_mapping = {1: 1, 2: -1, 0: 0, 3: 0}
    prev_rewarded = [choice_mapping[old_choice] for old_choice in choice]

    return one_hot_prev_choice, prev_rewarded

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
        shuffled_folds)) == num_folds, "require at least one session per fold for " \
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