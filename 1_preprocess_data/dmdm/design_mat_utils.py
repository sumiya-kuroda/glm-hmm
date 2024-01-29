import numpy as np
from ssm.util import one_hot

def create_design_mat_y(stim, outcome, reactiontimes, stimT, hazard):
    ''' 
    Create unnormalized input for y: with first column is changesize,
    second column as change onset. third column to sizth column as previous choice
    ''' 

    # Remap variables:
    # We first treat ref trials as FA here to allow only four outcomes (Hit/FA/Miss/Abort).
    # Hits in no change trials (where stim = 0) should also be regarded as FA, as from the mouse
    # perspective they are licking prior to changes. 
    # We then make change size and onset for FA and abort trials to be zero, 
    # because from mouse perspectives, they did not realize that there was a change.
    outcome_noref = ref2FA(outcome)
    choice_updated, stim_updated, stimT_updated, _ = remap_vals(outcome_noref, 
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

    continuous_column = [0, 1]

    return design_mat, outcome_noref, continuous_column

def create_design_mat_rt(stim, outcome, reactiontimes, stimT, hazard):
    ''' 
    Create unnormalized input for rt: with first column is changesize,
    second column as change onset. third column to sizth column as previous choice.
    seventh column as previous change onset. eigth column as previous reactiontimes.
    ''' 

    # Remap variables:
    # We first treat ref trials as FA here to allow only four outcomes (Hit/FA/Miss/Abort).
    # Hits in no change trials (where stim = 0) should also be regarded as FA, as from the mouse
    # perspective they are licking prior to changes. 
    # We then make change size and onset for FA and abort trials to be zero, 
    # because from mouse perspectives, they did not realize that there was a change.
    outcome_noref = ref2FA(outcome)


    choice_updated, stim_updated, stimT_updated, rt_updated = \
        remap_vals(outcome_noref, stim, stimT, 
                   reactiontimes, 
                   miss_onset_delay = 0)
    
    # Change size:
    design_mat = np.zeros((len(stimT), 8))
    design_mat[:, 0] = stim_updated # unnormalized

    # Change onset:
    design_mat[:, 1] = stimT_updated # unnormalized

    # previous choice vector:
    previous_choice, rewarded = create_previous_choice_vector(choice_updated)
    design_mat[:, 2:6] = previous_choice
    # design_mat[:, 4] = rewarded

    # previous Change onset
    previous_stimT = np.hstack([np.array(stimT_updated[0]), stimT_updated])[:-1]
    design_mat[:, 6] = previous_stimT # unnormalized

    # previous reactiontimes
    previous_rt = np.hstack([np.array(rt_updated[0]), rt_updated])[:-1]
    design_mat[:, 7] = previous_rt # unnormalized
    # WSLS

    # previous_stimT = np.hstack([np.array(stimT_updated[0]), stimT_updated])[:-1]
    # design_mat[:, 1] = preprocessing.scale(previous_stimT)

    # previous change onset and reactiontime:
    # previous_stimT, previous_rt = create_previous_RT_vector(stimT, reactiontimes, locs_FA_abort)
    # design_mat[:, 3] = previous_stimT
    # design_mat[:, 4] = previous_rt

    continuous_column = [0, 1, 6, 7]

    return design_mat, rt_updated, stimT_updated, continuous_column

def ref2FA(choice):
    new_choice = choice.copy()
    new_choice[new_choice == 4] = 2 # ref = 4, FA = 2
    return new_choice

def remap_vals(choice, stim, stimT, rt, delay=0.5, miss_onset_delay = 2.15):
    ''' 
    choice: choice vector of size T. 
            By default, hit = 1, FA = 2, miss = 0, abort = 3
    stim: change size vector of size T
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
        if stimT_updated[loc] <0:
            stimT_updated[loc] = 0

    # Make a change onset of miss trials to be a full length of that trial
    locs_miss = np.where(choice_updated == 0)[0]
    rt_updated  = rt.copy()
    for i, loc in enumerate(locs_miss):
        stimT_updated[loc] = stimT_updated[loc] + miss_onset_delay
        rt_updated[loc] = stimT_updated[loc] + 2.15

    return choice_updated, stim_updated, stimT_updated, rt_updated

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
    ''' 
    stimT: change onset vector of size T
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