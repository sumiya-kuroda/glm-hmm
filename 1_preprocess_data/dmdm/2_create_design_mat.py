#!/usr/bin/env python

import numpy as np
import numpy.random as npr
from sklearn import preprocessing
import os
import json
from collections import defaultdict
from preprocessing_utils import load_animal_list, load_animal_eid_dict, \
    get_all_unnormalized_data_this_session, create_train_test_sessions
from pathlib import Path
import defopt

npr.seed(65)

def main(dname, *, req_num_sessions = 30, num_folds=5):
    """
    Continue preprocessing of dmdm dataset and create design matrix for GLM(-HMM)
    
    :param str dname: name of dataset needs to be preprocessed
    :param int req_num_sessions: Required number of sessions for each animal
    :param int num_folds: Number of folds for k-fold cross-validation
    """
    if req_num_sessions < num_folds:
        raise ValueError('Required number of sessions must be greater than the number of folds')
    
    dirname = Path(os.path.dirname(os.path.abspath(__file__)))
    dmdm_data_path =  dirname.parents[1] / "data" / "dmdm" / dname
    processed_dmdm_data_path =  dmdm_data_path / "data_for_cluster"

    # Create directories for saving data:
    Path(processed_dmdm_data_path).mkdir(parents=True, exist_ok=True)
    # Also create a subdirectory for storing each individual animal's data:
    Path(processed_dmdm_data_path / "data_by_animal").mkdir(parents=True, exist_ok=True)

    # Load animal list/results of partial processing:
    animal_list = load_animal_list(
        dmdm_data_path / 'partially_processed' / 'animal_list.npz')
    animal_eid_dict = load_animal_eid_dict(
        dmdm_data_path / 'partially_processed' / 'animal_eid_dict.json')

    # Remove animals with few sessions
    for animal in animal_list:
        num_sessions = len(animal_eid_dict[animal])
        if num_sessions < req_num_sessions:
            animal_list = np.delete(animal_list, np.where(animal_list == animal))
            
    # Identify idx in master array where each animal's data starts and ends:
    animal_start_idx = {}
    animal_end_idx = {}
    final_animal_eid_dict = defaultdict(list)
    # WORKHORSE: iterate through each animal and each animal's set of eids;
    # obtain unnormalized data. Write out each animal's data and then also
    # write to master array
    for z, animal in enumerate(animal_list):
        sess_counter = 0
        for eid in animal_eid_dict[animal]:

            animal, session, y, reactiontimes, stim_onset,\
                unnormalized_inpts, needs_to_be_normalized = \
                get_all_unnormalized_data_this_session(eid, dmdm_data_path)
            assert len(unnormalized_inpts) == 2, "Require both y and rt design mat!"

            unnormalized_inpt_y = unnormalized_inpts[0]
            unnormalized_inpt_rt = unnormalized_inpts[1]

            if sess_counter == 0:
                animal_unnormalized_y_inpt = np.copy(unnormalized_inpt_y)
                animal_unnormalized_rt_inpt = np.copy(unnormalized_inpt_rt)
                animal_y = np.copy(y)
                animal_session = session
                animal_rt = np.copy(reactiontimes)
                animal_stim_onset = np.copy(stim_onset)
            else:
                animal_unnormalized_y_inpt = np.vstack(
                    (animal_unnormalized_y_inpt, unnormalized_inpt_y))
                animal_unnormalized_rt_inpt = np.vstack(
                    (animal_unnormalized_rt_inpt, unnormalized_inpt_rt))
                animal_y = np.concatenate((animal_y, y))
                animal_session = np.concatenate((animal_session, session))
                animal_rt = np.concatenate((animal_rt, reactiontimes))
                animal_stim_onset = np.concatenate((animal_stim_onset, stim_onset))
            sess_counter += 1
            final_animal_eid_dict[animal].append(eid)

        # Write out animal's unnormalized data matrix:
        np.savez(
            processed_dmdm_data_path / 'data_by_animal' / (animal + '_unnormalized.npz'),
            animal_unnormalized_y_inpt, animal_unnormalized_rt_inpt,
            animal_y, animal_session, animal_rt, animal_stim_onset)
        # Write out animal's train/test session:
        animal_session_fold_lookup = create_train_test_sessions(animal_session,
                                                                num_folds)
        np.savez(
            processed_dmdm_data_path / 'data_by_animal' / (animal + "_session_fold_lookup.npz"),
            animal_session_fold_lookup)
        
        # Now create or append data to master array across all animals:
        if z == 0:
            master_y_inpt = np.copy(animal_unnormalized_y_inpt)
            master_rt_inpt = np.copy(animal_unnormalized_rt_inpt)
            animal_start_idx[animal] = 0
            animal_end_idx[animal] = master_y_inpt.shape[0] - 1
            master_y = np.copy(animal_y)
            master_session = animal_session
            master_session_fold_lookup_table = animal_session_fold_lookup
            master_rt = np.copy(animal_rt)
            master_stim_onset = np.copy(animal_stim_onset)
        else:
            animal_start_idx[animal] = master_y_inpt.shape[0]
            master_y_inpt = np.concatenate((master_y_inpt, animal_unnormalized_y_inpt))
            master_rt_inpt = np.concatenate((master_rt_inpt, animal_unnormalized_rt_inpt))
            animal_end_idx[animal] = master_y_inpt.shape[0] - 1
            master_y = np.concatenate((master_y, animal_y))
            master_session = np.concatenate((master_session, animal_session))
            master_session_fold_lookup_table = np.concatenate(
                (master_session_fold_lookup_table, animal_session_fold_lookup))
            master_rt = np.concatenate((master_rt, animal_rt))
            master_stim_onset = np.concatenate((master_stim_onset, animal_stim_onset))

    # Write out master data across animals
    assert np.shape(master_y_inpt)[0] == np.shape(master_y)[
        0], "inpt and y not same length"
    assert len(np.unique(master_session)) == \
           np.shape(master_session_fold_lookup_table)[
               0], "number of unique sessions and session fold lookup don't " \
                   "match"
    
    # Z-score continuous variables in design matrix:
    normalized_y_inpt = np.copy(master_y_inpt)
    normalized_rt_inpt = np.copy(master_rt_inpt)
    for i, mat in enumerate([normalized_y_inpt, normalized_rt_inpt]):
        for c in needs_to_be_normalized[i]:
            mat[:, c] = preprocessing.scale(mat[:, c], with_mean=False)
    
    np.savez(
        processed_dmdm_data_path / 'all_animals_concat.npz',
        normalized_y_inpt, normalized_rt_inpt, master_y, 
        master_session, master_rt, master_stim_onset)
    np.savez(
        processed_dmdm_data_path / 'all_animals_concat_unnormalized.npz',
        master_y_inpt, master_rt_inpt, master_y, 
        master_session, master_rt, master_stim_onset)
    np.savez(
        processed_dmdm_data_path / 'all_animals_concat_session_fold_lookup.npz',
        master_session_fold_lookup_table)
    np.savez(
        processed_dmdm_data_path / 'data_by_animal/' / 'animal_list.npz',
        animal_list)

    out_json = json.dumps(final_animal_eid_dict)
    f = open(str(processed_dmdm_data_path / "final_animal_eid_dict.json"), "w")
    f.write(out_json)
    f.close()

    # Now write out normalized data (when normalized across all animals) for
    # each animal:
    counter = 0
    for animal in animal_start_idx.keys():
        start_idx = animal_start_idx[animal]
        end_idx = animal_end_idx[animal]
        inpt_y = normalized_y_inpt[range(start_idx, end_idx + 1)]
        inpt_rt = normalized_rt_inpt[range(start_idx, end_idx + 1)]
        y = master_y[range(start_idx, end_idx + 1)]
        session = master_session[range(start_idx, end_idx + 1)]
        rt = master_rt[range(start_idx, end_idx + 1)]
        stim_onset = master_stim_onset[range(start_idx, end_idx + 1)]
        counter += inpt_y.shape[0]
        np.savez(processed_dmdm_data_path / 'data_by_animal' / (animal + '_processed.npz'),
                 inpt_y, inpt_rt, y, session, rt, stim_onset)
    assert counter == master_y_inpt.shape[0]

if __name__ == "__main__":
    defopt.run(main)