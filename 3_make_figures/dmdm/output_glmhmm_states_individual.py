#!/usr/bin/env python
import defopt
import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append('../../2_fit_models/dmdm')
from data_io import get_file_dir, load_animal_list
from plotting_utils import load_glmhmm_result, infer_individual_glmhmm_result, flatten_list

sys.path.append('../../1_preprocess_data/dmdm')
from preprocessing_utils import load_animal_eid_dict

def main(*, dname: str, animal: str, model: str, K:int, regularization: str, bestK:bool, suffix: str = None):
    """
    Begin processing dmdm data: identify animals in dataset 
    that enter biased blocks and list their session ids.
    
    :param str dname: name of dataset needs to be preprocessed
    :param str animal: name of animal
    :param str model: name of GLM-HMM model
    :param int K: number of states
    :param bool bestK: whether K is the best value
    :param str regularization: specify regularization method. If not using regularization, specify None
    :param str suffix: suffix of saving location if any
    """

    print('Saving individual GLMHMM results ...')
    data_dir =  get_file_dir().parents[1] / "data" / "dmdm" / dname / 'data_for_cluster'
    data_2_dir =  get_file_dir().parents[1] / "data" / "dmdm" / dname / 'data_for_cluster' / "data_by_animal"
    results_2_dir = get_file_dir().parents[1] / "results" / "dmdm_individual_fit" / dname

    figure_dir = get_file_dir().parents[1] / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    if regularization == 'None':
        regularization = None # change dtype
    if regularization is None:
        is_regularized = 'None'
    else:
        is_regularized = regularization

    if suffix is None:
        inference_input = dname
    else:
        inference_input = dname + '_' + suffix
    print('dname: {}; animal: {}; model: {}; regularization: {}, state_num: {}; data_input {}'.format(dname, animal, model, is_regularized, K, inference_input))

    if suffix is None:
        # Simply Load GLM-HMM results
        states_max_posterior, posterior_probs, _, inpt_rt, _, session, _, _, mask, hmm_params \
            = load_glmhmm_result(animal, K, model, results_2_dir, data_2_dir)
        
        # Load raw data eids
        animal_list = load_animal_list(
            data_dir / 'data_by_animal' / 'animal_list.npz')
        animal_eid_dict = load_animal_eid_dict(
            data_dir / 'final_animal_eid_dict.json')

        # Create trial IDs
        trialnum_all = []
        for z, animal in enumerate(animal_list):
            sess_counter = 0
            for eid in animal_eid_dict[animal]:

                raw_session_id = eid.split('Subjects/')[1]
                # Get animal:
                animal = raw_session_id.split('/')[0]
                # Replace '/' with dash in session ID
                session_id = raw_session_id.replace('/', '-')

                trialnum = np.load(data_dir.parent / Path(eid) / '_dmdm_trials.trial.npy')[0]
                trialnum2 = [session_id + '-' + str(int(t)) for t in trialnum]
                trialnum_all.append(trialnum2)
    else:
         # Infer GLM-HMM states
        data_2_dir_inference =  get_file_dir().parents[1] / "data" / "dmdm" / inference_input / 'data_for_cluster' 
        states_max_posterior, posterior_probs, _, inpt_rt, _, session, _, _, mask, hmm_params \
            = infer_individual_glmhmm_result(animal, K, model, results_2_dir, data_2_dir, data_2_dir_inference / 'data_by_animal')
        
        # Load raw data eids
        animal_list = load_animal_list(
            data_2_dir_inference / 'data_by_animal' / 'animal_list.npz')
        animal_eid_dict = load_animal_eid_dict(
            data_2_dir_inference / 'final_animal_eid_dict.json')
        
        # Create trial IDs
        trialnum_all = []
        for eid in animal_eid_dict[animal]:

            raw_session_id = eid.split('Subjects/')[1]
            # Get animal:
            animal = raw_session_id.split('/')[0]
            # Replace '/' with dash in session ID
            session_id = raw_session_id.replace('/', '-')

            trialnum = np.load(data_2_dir_inference.parent / Path(eid) / '_dmdm_trials.trial.npy')[0]
            trialnum2 = [session_id + '-' + str(int(t)) for t in trialnum]
            trialnum_all.append(trialnum2)
        
    # Save outputs in json file
    rdict = defaultdict(list)
    for zk in range(K):
        b = np.where(states_max_posterior == zk)
        x = np.array(flatten_list(trialnum_all))[b]
        rdict[zk] = x.tolist()

    out_json = json.dumps(rdict)
    if (suffix is None) and (bestK is not True):
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_{}.json".format(dname, K, is_regularized, animal)), "w")
    elif (suffix is None) and (bestK is True):
        f = open(str(figure_dir / "state_session_trial_{}_bestK_{}_{}.json".format(dname, is_regularized, animal)), "w")
    elif (suffix is not None) and (bestK is not True):
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_{}_{}.json".format(dname, K, is_regularized, animal, suffix)), "w")
    elif (suffix is not None) and (bestK is True):
        f = open(str(figure_dir / "state_session_trial_{}_bestK_{}_{}_{}.json".format(dname, is_regularized, animal, suffix)), "w")
    f.write(out_json)
    f.close()

    # highprob
    rdict = defaultdict(list)
    for zk in range(K):
        b = np.where(np.logical_and(states_max_posterior == zk, posterior_probs >= 0.8))[0]
        x = np.array(flatten_list(trialnum_all))[b]
        rdict[zk] = x.tolist()

    out_json = json.dumps(rdict)
    if (suffix is None) and (bestK is not True):
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_{}_highprob.json".format(dname, K, is_regularized, animal)), "w")
    elif (suffix is None) and (bestK is True):
        f = open(str(figure_dir / "state_session_trial_{}_bestK_{}_{}_highprob.json".format(dname, is_regularized, animal)), "w")
    elif (suffix is not None) and (bestK is not True):
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_{}_{}_highprob.json".format(dname, K, is_regularized, animal, suffix)), "w")
    elif (suffix is not None) and (bestK is True):
        f = open(str(figure_dir / "state_session_trial_{}_bestK_{}_{}_{}_highprob.json".format(dname, is_regularized, animal, suffix)), "w")
    f.write(out_json)
    f.close()

if __name__ == "__main__":
    defopt.run(main)